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


class DQNTrainer:
    """组织 DQN 的环境交互、训练、记录和模型保存。"""

    def __init__(self, config: DQNConfig) -> None:
        """根据配置创建训练环境、评估配置和 DQN Agent。"""

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
        self.best_episode_return = float("-inf")

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

            if record.episode_return > self.best_episode_return:
                self.best_episode_return = record.episode_return
                self.agent.save(self.output_dir / "best_model.pt")

            if (episode_index + 1) % self.config.evaluation.checkpoint_interval == 0:
                self.agent.save(self.output_dir / "last_model.pt")

            print(
                f"Episode {record.episode:04d} | "
                f"Return {record.episode_return:7.2f} | "
                f"Length {record.episode_length:4d} | "
                f"Epsilon {record.epsilon:.4f} | "
                f"Loss {self._format_metric(record.mean_loss)}"
            )

        self.agent.save(self.output_dir / "last_model.pt")
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

    def evaluate(
        self,
        num_episodes: int | None = None,
        render: bool | None = None,
    ) -> dict[str, Any]:
        """使用贪心策略评估当前 Agent。

        参数：
            num_episodes：评估回合数。未提供时使用配置值。
            render：是否显示环境画面。未提供时使用配置值。

        返回：
            包含每回合奖励、平均奖励、奖励标准差和平均长度的字典。
        """

        total_episodes = (
            self.config.evaluation.num_episodes if num_episodes is None else num_episodes
        )
        should_render = self.config.evaluation.render if render is None else render
        if total_episodes <= 0:
            raise ValueError("num_episodes 必须大于 0。")

        evaluation_environment = gym.make(
            self.config.env_id,
            render_mode="human" if should_render else None,
        )
        returns: list[float] = []
        lengths: list[int] = []

        self.agent.eval()
        try:
            for episode_index in range(total_episodes):
                episode_seed = self.config.training.seed + 10_000 + episode_index
                state, _ = evaluation_environment.reset(seed=episode_seed)
                evaluation_environment.action_space.seed(episode_seed)
                episode_return = 0.0
                episode_length = 0

                for step_index in range(self.config.training.max_steps_per_episode):
                    action = self.agent.select_action(state, explore=False)
                    next_state, reward, terminated, truncated, _ = evaluation_environment.step(
                        action
                    )
                    state = next_state
                    episode_return += float(reward)
                    episode_length = step_index + 1

                    if terminated or truncated:
                        break

                returns.append(episode_return)
                lengths.append(episode_length)
        finally:
            evaluation_environment.close()
            self.agent.train()

        result = {
            "episode_returns": returns,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }
        self._write_evaluation(result)
        return result

    def close(self) -> None:
        """关闭训练环境。"""

        self.environment.close()

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

    def _write_evaluation(self, result: dict[str, Any]) -> None:
        """将评估结果写入 JSON 文件。"""

        with (self.output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_metric(metric: float | None) -> str:
        """格式化可能为空的训练指标。"""

        return "-" if metric is None else f"{metric:.6f}"
