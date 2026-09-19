"""DQN 训练器和评估流程。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .agent import DQNAgent
from .config import DQNConfig
from .visualization import save_test_comparison, save_training_curves


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """一个训练回合的统计信息。"""

    episode: int
    episode_return: float
    episode_length: int
    mean_loss: float | None
    epsilon: float
    learning_rate: float
    update_steps: int


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """一次独立验证的统计信息。"""

    episode: int
    mean_return: float
    std_return: float
    mean_length: float
    success_rate: float | None


class DQNTrainer:
    """组织 DQN 的环境交互、训练、记录和模型保存。"""

    def __init__(self, config: DQNConfig, save_config: bool = True) -> None:
        """根据配置创建训练环境和 DQN Agent。

        参数：
            config：DQN 完整配置。
            save_config：是否立即把配置写入输出目录。纯评估模式应设置为
                ``False``，避免覆盖原训练实验的配置文件。
        """

        self.config = config
        self.output_dir = config.evaluation.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练环境不打开可视化窗口，避免训练过程中产生额外开销。
        self.environment = gym.make(config.env_id)
        state_dim = self._get_state_dim(self.environment)
        action_dim = self._get_action_dim(self.environment)

        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
        )
        self.history: list[EpisodeRecord] = []
        self.validation_history: list[ValidationRecord] = []
        self.best_validation_return = float("-inf")
        self.solved_score = self._resolve_solved_score(self.environment)

        if save_config:
            self._save_config()

    @staticmethod
    def _get_state_dim(environment: gym.Env) -> int:
        """从环境观测空间读取状态维度。"""

        if not isinstance(environment.observation_space, gym.spaces.Box):
            raise TypeError("当前 DQNTrainer 要求环境的 observation_space 是 Box。")
        if len(environment.observation_space.shape) != 1:
            raise ValueError("当前 DQNTrainer 要求状态是一维向量。")
        return int(environment.observation_space.shape[0])

    @staticmethod
    def _get_action_dim(environment: gym.Env) -> int:
        """从环境动作空间读取离散动作数量。"""

        if not isinstance(environment.action_space, gym.spaces.Discrete):
            raise TypeError("当前 DQNTrainer 要求环境的 action_space 是 Discrete。")
        return int(environment.action_space.n)

    def train(self, num_episodes: int | None = None) -> list[EpisodeRecord]:
        """运行 DQN 训练过程。

        参数：
            num_episodes：训练回合数。未提供时使用配置中的默认值，主要用于
                测试时缩短训练过程。

        返回：
            每个训练回合对应的 :class:`EpisodeRecord` 列表。
        """

        total_episodes = (
            self.config.training.num_episodes if num_episodes is None else num_episodes
        )
        if total_episodes <= 0:
            raise ValueError("num_episodes 必须大于 0。")

        self.agent.train()

        for episode_index in range(total_episodes):
            record = self._run_training_episode(episode_index)
            self.history.append(record)
            self._write_history()

            if (episode_index + 1) % self.config.evaluation.checkpoint_interval == 0:
                self.agent.save(self.output_dir / "last_model.pt")

            print(
                f"Episode {record.episode:04d} | "
                f"Return {record.episode_return:7.2f} | "
                f"Length {record.episode_length:4d} | "
                f"Epsilon {record.epsilon:.4f} | "
                f"Loss {self._format_metric(record.mean_loss)}"
            )

            # 验证场景不写入 Replay Buffer，也不进行梯度更新。固定验证种子可以
            # 公平比较不同训练阶段，并用验证平均奖励选择最佳模型。
            should_validate = (
                record.episode % self.config.evaluation.validation_interval == 0
                or record.episode == total_episodes
            )
            if should_validate:
                validation_record = self.validate(record.episode)
                self.validation_history.append(validation_record)
                self._write_validation_history()

                if validation_record.mean_return > self.best_validation_return:
                    self.best_validation_return = validation_record.mean_return
                    self.agent.save(self.output_dir / "best_model.pt")

                print(
                    "  Validation | "
                    f"Mean return {validation_record.mean_return:7.2f} | "
                    f"Std {validation_record.std_return:6.2f} | "
                    f"Success {self._format_percentage(validation_record.success_rate)}"
                )

        self.agent.save(self.output_dir / "last_model.pt")
        self._save_training_visualization()
        return self.history

    def _run_training_episode(self, episode_index: int) -> EpisodeRecord:
        """运行一个训练回合并收集统计信息。"""

        episode_seed = self.config.training.seed + episode_index
        state, _ = self.environment.reset(seed=episode_seed)
        self.environment.action_space.seed(episode_seed)

        episode_return = 0.0
        episode_length = 0
        losses: list[float] = []

        for step_index in range(self.config.training.max_steps_per_episode):
            action = self.agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = self.environment.step(action)

            # 如果达到 Trainer 自己设置的最大步数，也视为外部截断。
            reached_trainer_limit = (
                step_index + 1 >= self.config.training.max_steps_per_episode
                and not terminated
                and not truncated
            )
            effective_truncated = truncated or reached_trainer_limit

            self.agent.observe(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=effective_truncated,
            )
            metrics = self.agent.update()

            if metrics is not None:
                losses.append(float(metrics["loss"]))

            state = next_state
            episode_return += float(reward)
            episode_length = step_index + 1

            if terminated or truncated or reached_trainer_limit:
                break

        return EpisodeRecord(
            episode=episode_index + 1,
            episode_return=episode_return,
            episode_length=episode_length,
            mean_loss=float(np.mean(losses)) if losses else None,
            epsilon=self.agent.epsilon,
            learning_rate=float(self.agent.optimizer.param_groups[0]["lr"]),
            update_steps=self.agent.update_steps,
        )

    def validate(self, episode: int) -> ValidationRecord:
        """在固定且独立的验证场景上评估当前策略。

        参数：
            episode：触发本次验证时已经完成的训练回合数。

        返回：
            用于选择最佳模型的验证指标。验证期间禁用探索和参数更新。
        """

        result = self._run_greedy_policy(
            num_episodes=self.config.evaluation.validation_episodes,
            seed_offset=self.config.evaluation.validation_seed_offset,
            render=False,
        )
        return ValidationRecord(
            episode=episode,
            mean_return=float(result["mean_return"]),
            std_return=float(result["std_return"]),
            mean_length=float(result["mean_length"]),
            success_rate=result["success_rate"],
        )

    def evaluate(
        self,
        num_episodes: int | None = None,
        render: bool = False,
    ) -> dict[str, Any]:
        """在从未用于训练和选模的随机场景上执行最终测试。

        参数：
            num_episodes：最终测试回合数。未提供时使用配置值。
            render：是否把测试过程显示为窗口。正式统计通常保持 ``False``，
                需要观看策略表现时应调用 :meth:`demonstrate`。

        返回：
            包含逐回合奖励、均值、标准差、成功率、是否达到环境解决标准，
            以及可选随机策略基线的字典。
        """

        total_episodes = (
            self.config.evaluation.num_episodes if num_episodes is None else num_episodes
        )
        if total_episodes <= 0:
            raise ValueError("num_episodes 必须大于 0。")

        result = self._run_greedy_policy(
            num_episodes=total_episodes,
            seed_offset=self.config.evaluation.test_seed_offset,
            render=render,
        )
        if self.config.evaluation.compare_random_baseline:
            random_result = self._run_random_policy(
                num_episodes=total_episodes,
                seed_offset=self.config.evaluation.test_seed_offset,
            )
            result["random_baseline_returns"] = random_result["episode_returns"]
            result["random_baseline_mean_return"] = random_result["mean_return"]

        self._write_evaluation(result)
        if self.config.evaluation.save_plots:
            save_test_comparison(result, self.output_dir / "test_comparison.png")
        return result

    def demonstrate(self, num_episodes: int | None = None) -> dict[str, Any]:
        """打开 Gymnasium 窗口，直观展示训练后策略。

        参数：
            num_episodes：演示回合数。未提供时使用配置值。

        返回：
            演示回合的奖励统计。演示结果不用于选模或最终成绩。
        """

        total_episodes = (
            self.config.evaluation.demo_episodes if num_episodes is None else num_episodes
        )
        if total_episodes <= 0:
            raise ValueError("num_episodes 必须大于 0。")
        return self._run_greedy_policy(
            num_episodes=total_episodes,
            seed_offset=self.config.evaluation.demo_seed_offset,
            render=True,
        )

    def load_best_model(self) -> Path:
        """加载按照验证平均奖励选出的最佳模型。"""

        checkpoint_path = self.output_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"尚未生成最佳模型：{checkpoint_path}")
        self.agent.load(checkpoint_path, load_training_state=False)
        return checkpoint_path

    def _run_greedy_policy(
        self,
        num_episodes: int,
        seed_offset: int,
        render: bool,
    ) -> dict[str, Any]:
        """使用 epsilon=0 的贪心策略运行一组独立场景。"""

        evaluation_environment = gym.make(
            self.config.env_id,
            render_mode="human" if render else None,
        )
        returns: list[float] = []
        lengths: list[int] = []

        self.agent.eval()
        try:
            for episode_index in range(num_episodes):
                episode_seed = self.config.training.seed + seed_offset + episode_index
                state, _ = evaluation_environment.reset(seed=episode_seed)
                evaluation_environment.action_space.seed(episode_seed)
                episode_return = 0.0
                episode_length = 0

                for step_index in range(self.config.training.max_steps_per_episode):
                    action = self.agent.select_action(state, explore=False)
                    state, reward, terminated, truncated, _ = evaluation_environment.step(action)
                    episode_return += float(reward)
                    episode_length = step_index + 1
                    if terminated or truncated:
                        break

                returns.append(episode_return)
                lengths.append(episode_length)
        finally:
            evaluation_environment.close()
            self.agent.train()

        return self._build_evaluation_result(returns, lengths)

    def _run_random_policy(self, num_episodes: int, seed_offset: int) -> dict[str, Any]:
        """在相同测试种子上运行随机动作策略，作为最低基线。"""

        baseline_environment = gym.make(self.config.env_id)
        returns: list[float] = []
        lengths: list[int] = []
        try:
            for episode_index in range(num_episodes):
                episode_seed = self.config.training.seed + seed_offset + episode_index
                _, _ = baseline_environment.reset(seed=episode_seed)
                baseline_environment.action_space.seed(episode_seed)
                episode_return = 0.0
                episode_length = 0

                for step_index in range(self.config.training.max_steps_per_episode):
                    action = baseline_environment.action_space.sample()
                    _, reward, terminated, truncated, _ = baseline_environment.step(action)
                    episode_return += float(reward)
                    episode_length = step_index + 1
                    if terminated or truncated:
                        break

                returns.append(episode_return)
                lengths.append(episode_length)
        finally:
            baseline_environment.close()

        return self._build_evaluation_result(returns, lengths)

    def _build_evaluation_result(
        self,
        returns: list[float],
        lengths: list[int],
    ) -> dict[str, Any]:
        """根据逐回合结果计算统一的评估统计量。"""

        mean_return = float(np.mean(returns))
        success_rate = None
        is_solved = None
        if self.solved_score is not None:
            success_rate = float(np.mean(np.asarray(returns) >= self.solved_score))
            is_solved = mean_return >= self.solved_score

        return {
            "episode_returns": returns,
            "episode_lengths": lengths,
            "mean_return": mean_return,
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": success_rate,
            "solved_score": self.solved_score,
            "is_solved": is_solved,
        }

    def close(self) -> None:
        """关闭训练环境。"""

        self.environment.close()

    def _resolve_solved_score(self, environment: gym.Env) -> float | None:
        """读取配置或 Gymnasium 注册信息中的环境解决阈值。"""

        configured_score = self.config.evaluation.solved_score
        if configured_score is not None:
            return float(configured_score)
        if environment.spec is None or environment.spec.reward_threshold is None:
            return None
        return float(environment.spec.reward_threshold)

    def _save_config(self) -> None:
        """将实验配置保存为 JSON。"""

        with (self.output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.to_dict(), file, ensure_ascii=False, indent=2)

    def _write_history(self) -> None:
        """将当前训练历史写入 CSV 文件。"""

        if not self.history:
            return

        rows = [asdict(record) for record in self.history]
        with (self.output_dir / "training_metrics.csv").open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_validation_history(self) -> None:
        """将独立验证历史写入 CSV 文件。"""

        if not self.validation_history:
            return

        rows = [asdict(record) for record in self.validation_history]
        with (self.output_dir / "validation_metrics.csv").open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_evaluation(self, result: dict[str, Any]) -> None:
        """将独立测试结果写入 JSON 文件。"""

        with (self.output_dir / "test_evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    def _save_training_visualization(self) -> Path | None:
        """根据当前训练和验证历史保存综合诊断图。"""

        if not self.config.evaluation.save_plots or not self.history:
            return None
        return save_training_curves(
            history=self.history,
            validation_history=self.validation_history,
            output_path=self.output_dir / "training_curves.png",
            moving_average_window=self.config.evaluation.moving_average_window,
            solved_score=self.solved_score,
        )

    @staticmethod
    def _format_metric(metric: float | None) -> str:
        """格式化可能为空的训练指标。"""

        return "-" if metric is None else f"{metric:.6f}"

    @staticmethod
    def _format_percentage(value: float | None) -> str:
        """把可能为空的比例格式化为百分数。"""

        return "-" if value is None else f"{value:.1%}"
