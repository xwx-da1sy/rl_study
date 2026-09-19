"""DQN 实验的集中配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class NetworkConfig:
    """Q 网络结构配置。"""

    # 可选值："mlp" 或 "residual_mlp"。
    architecture: str = "mlp"

    # 三层隐藏层对应：4 -> 256 -> 128 -> 128 -> action_dim。
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    hidden_dim_3: int = 128


@dataclass(frozen=True)
class OptimizerConfig:
    """Online Network 优化器配置。"""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 10.0


@dataclass(frozen=True)
class SchedulerConfig:
    """学习率调度器配置。"""

    # None 表示根据训练回合数和每回合最大步数自动估算。
    t_max: int | None = None
    eta_min: float = 1e-5


@dataclass(frozen=True)
class ReplayBufferConfig:
    """经验回放池配置。"""

    capacity: int = 10_000
    batch_size: int = 64
    warmup_size: int = 1_000


@dataclass(frozen=True)
class ExplorationConfig:
    """ε-greedy 探索策略配置。"""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 10_000


@dataclass(frozen=True)
class TrainingConfig:
    """DQN 训练过程配置。"""

    seed: int = 42
    num_episodes: int = 500
    max_steps_per_episode: int = 500
    gamma: float = 0.99
    train_frequency: int = 1
    target_update_frequency: int = 500
    device: str = "auto"


@dataclass(frozen=True)
class EvaluationConfig:
    """模型评估和保存配置。"""

    num_episodes: int = 10
    render: bool = False
    checkpoint_interval: int = 50
    output_dir: Path = field(default_factory=lambda: Path("runs/cartpole_dqn"))


@dataclass(frozen=True)
class DQNConfig:
    """CartPole DQN 实验的完整配置。"""

    # 环境名称。状态维度和动作数量由环境空间自动读取，不在这里写死。
    env_id: str = "CartPole-v1"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        """检查配置中的关键参数是否合理。"""

        if not self.env_id:
            raise ValueError("env_id 不能为空。")
        if self.network.architecture not in {"mlp", "residual_mlp"}:
            raise ValueError("network.architecture 必须是 'mlp' 或 'residual_mlp'。")
        if (
            self.network.architecture == "residual_mlp"
            and self.network.hidden_dim_2 != self.network.hidden_dim_3
        ):
            raise ValueError("residual_mlp 要求 hidden_dim_2 等于 hidden_dim_3。")

        if any(
            hidden_dim <= 0
            for hidden_dim in (
                self.network.hidden_dim_1,
                self.network.hidden_dim_2,
                self.network.hidden_dim_3,
            )
        ):
            raise ValueError("Q 网络隐藏层宽度必须大于 0。")

        if self.optimizer.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0。")
        if self.optimizer.weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0。")
        if self.optimizer.gradient_clip_norm is not None and self.optimizer.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm 必须大于 0 或设置为 None。")
        if self.scheduler.t_max is not None and self.scheduler.t_max <= 0:
            raise ValueError("scheduler.t_max 必须大于 0 或设置为 None。")
        if self.scheduler.eta_min < 0:
            raise ValueError("scheduler.eta_min 不能小于 0。")
        if self.scheduler.eta_min > self.optimizer.learning_rate:
            raise ValueError("scheduler.eta_min 不能大于 learning_rate。")

        if self.replay_buffer.capacity <= 0:
            raise ValueError("经验回放池容量必须大于 0。")
        if self.replay_buffer.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if not 0 < self.replay_buffer.warmup_size <= self.replay_buffer.capacity:
            raise ValueError("warmup_size 必须大于 0 且不超过回放池容量。")

        if not 0 <= self.exploration.epsilon_end <= self.exploration.epsilon_start <= 1:
            raise ValueError("epsilon 必须满足 0 <= epsilon_end <= epsilon_start <= 1。")
        if self.exploration.decay_steps <= 0:
            raise ValueError("epsilon 衰减步数必须大于 0。")

        if not 0 < self.training.gamma <= 1:
            raise ValueError("gamma 必须满足 0 < gamma <= 1。")
        if self.training.num_episodes <= 0:
            raise ValueError("训练回合数必须大于 0。")
        if self.training.max_steps_per_episode <= 0:
            raise ValueError("每回合最大步数必须大于 0。")
        if self.training.train_frequency <= 0:
            raise ValueError("train_frequency 必须大于 0。")
        if self.training.target_update_frequency <= 0:
            raise ValueError("target_update_frequency 必须大于 0。")

        if self.evaluation.num_episodes <= 0:
            raise ValueError("评估回合数必须大于 0。")
        if self.evaluation.checkpoint_interval <= 0:
            raise ValueError("模型保存间隔必须大于 0。")

    def resolve_device(self) -> torch.device:
        """根据配置解析实际运行设备。

        返回：
            当配置为 ``"auto"`` 时，如果 CUDA 可用则返回 CUDA，否则返回 CPU；
            其他情况直接返回配置指定的 PyTorch 设备。
        """

        if self.training.device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.training.device

        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 环境无法使用 CUDA，请选择 cpu 或 auto。")
        return device

    def estimate_optimizer_updates(self) -> int:
        """估算本次训练最多进行多少次 Online Network 更新。

        返回：
            根据最大训练步数和训练频率估算出的优化器更新次数。这个值用于
            为余弦退火调度器设置默认的 ``T_max``。
        """

        total_environment_steps = self.training.num_episodes * self.training.max_steps_per_episode
        return max(
            1,
            (total_environment_steps + self.training.train_frequency - 1)
            // self.training.train_frequency,
        )

    def resolve_scheduler_t_max(self) -> int:
        """获取余弦退火调度器实际使用的 ``T_max``。"""

        if self.scheduler.t_max is not None:
            return self.scheduler.t_max
        return self.estimate_optimizer_updates()

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为适合日志记录或保存为 JSON 的字典。"""

        config_dict = asdict(self)
        config_dict["evaluation"]["output_dir"] = str(self.evaluation.output_dir)
        return config_dict


def create_default_config() -> DQNConfig:
    """创建一份 CartPole DQN 默认配置。"""

    return DQNConfig()
