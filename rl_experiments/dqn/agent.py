"""DQN 智能体。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import DQNConfig
from .network import DQNNetworks
from .optimizer import build_optimizer_and_scheduler, clip_online_gradients
from .replay_buffer import ReplayBuffer, TransitionBatch


class DQNAgent:
    """使用 DQN 算法进行决策和学习的智能体。

    Agent 负责：

    - 使用 Online Network 和 Target Network 估计动作价值；
    - 使用 ε-greedy 选择动作；
    - 保存和采样 Replay Buffer 中的经验；
    - 计算 TD target 并更新 Online Network；
    - 定期同步 Target Network；
    - 保存和加载训练检查点。

    环境的 episode 循环由 Trainer 负责，Agent 不直接创建或控制环境。

    参数：
        state_dim：环境状态的维度。
        action_dim：环境离散动作的数量。
        config：DQN 完整配置。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig,
    ) -> None:
        """创建 DQN Agent 及其全部学习组件。"""

        if state_dim <= 0:
            raise ValueError("state_dim 必须大于 0。")
        if action_dim <= 0:
            raise ValueError("action_dim 必须大于 0。")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.resolve_device()

        # 固定网络初始化随机性，保证相同配置可以复现实验。
        torch.manual_seed(config.training.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.training.seed)

        # 使用配置中的网络结构创建 Online/Target 两个独立网络。
        self.networks = DQNNetworks(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim_1=config.network.hidden_dim_1,
            hidden_dim_2=config.network.hidden_dim_2,
            hidden_dim_3=config.network.hidden_dim_3,
            architecture=config.network.architecture,
            device=self.device,
        )

        # Replay Buffer 负责保存经验和随机采样 Batch。
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer.capacity,
            seed=config.training.seed,
        )

        # 优化器和调度器只绑定 Online Network。
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            online_network=self.networks.online_network,
            config=config,
        )

        # 独立的随机数生成器用于 ε-greedy 探索。
        self._random = random.Random(config.training.seed)

        # 环境步数用于计算 epsilon，更新次数用于同步 Target Network。
        self.environment_steps = 0
        self.update_steps = 0

    @property
    def epsilon(self) -> float:
        """返回当前 ε-greedy 探索概率。"""

        exploration = self.config.exploration
        progress = min(
            self.environment_steps / exploration.decay_steps,
            1.0,
        )
        return exploration.epsilon_start + progress * (
            exploration.epsilon_end - exploration.epsilon_start
        )

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """根据当前状态选择一个离散动作。

        参数：
            state：环境返回的状态，形状通常为 ``[state_dim]``。
            explore：是否启用 ε-greedy 探索。评估时应传入 ``False``。

        返回：
            一个范围为 ``[0, action_dim)`` 的整数动作。
        """

        if explore and self._random.random() < self.epsilon:
            return self._random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        was_training = self.networks.online_network.training
        self.networks.online_network.eval()
        with torch.no_grad():
            q_values = self.networks.online_network(state_tensor)
        if was_training:
            self.networks.train_online_network()

        return int(q_values.argmax(dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """保存一条环境交互经验并推进环境步数。

        参数：
            state：执行动作前的状态。
            action：Agent 执行的动作。
            reward：环境返回的即时奖励。
            next_state：执行动作后的状态。
            terminated：环境是否真正终止。
            truncated：环境是否因为时间限制等原因截断。

        返回：
            ``None``。
        """

        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )
        self.environment_steps += 1

    def can_update(self) -> bool:
        """判断当前是否满足一次 DQN 更新的条件。"""

        if self.environment_steps % self.config.training.train_frequency != 0:
            return False
        if len(self.replay_buffer) < self.config.replay_buffer.warmup_size:
            return False
        return self.replay_buffer.can_sample(self.config.replay_buffer.batch_size)

    def update(self) -> dict[str, float | int] | None:
        """采样一个 Batch 并更新 Online Network。

        返回：
            如果当前经验不足或尚未到达训练频率，返回 ``None``；否则返回包含
            loss、梯度范数、epsilon、学习率和 Q 值统计的指标字典。
        """

        if not self.can_update():
            return None

        batch = self.replay_buffer.sample(self.config.replay_buffer.batch_size)
        metrics = self._update_from_batch(batch)
        return metrics

    def _update_from_batch(self, batch: TransitionBatch) -> dict[str, float | int]:
        """使用一个已经采样的 Batch 执行一次 TD 更新。"""

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch.next_states,
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            batch.terminated,
            dtype=torch.float32,
            device=self.device,
        )

        self.networks.train_online_network()

        # 只提取实际执行动作对应的 Q(s, a)。
        current_q_values = self.networks.online_network(states).gather(1, actions).squeeze(1)

        # Target Network 只计算目标值，不参与梯度传播。
        with torch.no_grad():
            next_q_values = self.networks.target_network(next_states).max(dim=1).values
            target_q_values = rewards + self.config.training.gamma * (
                1.0 - terminated
            ) * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = clip_online_gradients(
            online_network=self.networks.online_network,
            config=self.config,
        )
        self.optimizer.step()
        self.scheduler.step()

        self.update_steps += 1
        if self.update_steps % self.config.training.target_update_frequency == 0:
            self.networks.sync_target_network()

        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return {
            "loss": float(loss.item()),
            "gradient_norm": float(gradient_norm or 0.0),
            "mean_current_q": float(current_q_values.mean().item()),
            "mean_target_q": float(target_q_values.mean().item()),
            "epsilon": self.epsilon,
            "learning_rate": learning_rate,
            "update_steps": self.update_steps,
        }

    def train(self) -> None:
        """将 Online Network 切换到训练模式。"""

        self.networks.train_online_network()

    def eval(self) -> None:
        """将 Online Network 和 Target Network 切换到评估模式。"""

        self.networks.evaluate()

    def save(self, path: str | Path) -> None:
        """保存模型、优化器、调度器和训练进度。

        参数：
            path：检查点文件路径，通常使用 ``.pt`` 后缀。
        """

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "online_network": self.networks.online_network.state_dict(),
            "target_network": self.networks.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def load(
        self,
        path: str | Path,
        load_training_state: bool = True,
    ) -> dict[str, Any]:
        """加载模型参数和可选的训练状态。

        参数：
            path：检查点文件路径。
            load_training_state：是否同时恢复优化器、调度器和训练步数。

        返回：
            从检查点中读取的原始字典，便于调用方检查额外信息。
        """

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        self.networks.online_network.load_state_dict(checkpoint["online_network"])
        self.networks.target_network.load_state_dict(checkpoint["target_network"])

        if load_training_state:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.environment_steps = int(checkpoint["environment_steps"])
            self.update_steps = int(checkpoint["update_steps"])

        return checkpoint
