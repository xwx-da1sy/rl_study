"""DQN 工程的单文件冒烟测试。"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import DQNConfig
from .network import DQNNetworks, build_q_network
from .replay_buffer import ReplayBuffer
from .trainer import DQNTrainer


def check_replay_buffer() -> None:
    """检查经验写入、容量覆盖和 Batch 采样。"""

    replay_buffer = ReplayBuffer(capacity=4, seed=42)
    for index in range(6):
        replay_buffer.push(
            state=np.full(4, index, dtype=np.float32),
            action=index % 2,
            reward=float(index),
            next_state=np.full(4, index + 1, dtype=np.float32),
            terminated=False,
            truncated=False,
        )

    batch = replay_buffer.sample(batch_size=4)
    assert len(replay_buffer) == 4
    assert batch.states.shape == (4, 4)
    assert batch.actions.shape == (4,)


def check_networks() -> None:
    """检查普通网络、残差网络和 Online/Target 分离。"""

    states = torch.zeros(8, 4)
    for architecture in ("mlp", "residual_mlp"):
        network = build_q_network(architecture, state_dim=4, action_dim=2)
        assert network(states).shape == (8, 2)

    networks = DQNNetworks(state_dim=4, action_dim=2)
    assert networks.online_network is not networks.target_network
    assert all(
        torch.equal(online_parameter, target_parameter)
        for online_parameter, target_parameter in zip(
            networks.online_network.parameters(),
            networks.target_network.parameters(),
        )
    )
    assert not any(
        parameter.requires_grad for parameter in networks.target_network.parameters()
    )


def check_training_pipeline() -> None:
    """运行极短训练，检查 Agent、Trainer、保存和评估流程。"""

    base_config = DQNConfig()
    with tempfile.TemporaryDirectory() as temporary_directory:
        config = replace(
            base_config,
            training=replace(
                base_config.training,
                device="cpu",
                num_episodes=2,
                max_steps_per_episode=8,
                target_update_frequency=2,
            ),
            replay_buffer=replace(
                base_config.replay_buffer,
                capacity=32,
                batch_size=4,
                warmup_size=4,
            ),
            evaluation=replace(
                base_config.evaluation,
                num_episodes=2,
                checkpoint_interval=1,
                output_dir=Path(temporary_directory),
            ),
        )

        trainer = DQNTrainer(config)
        try:
            history = trainer.train()
            evaluation = trainer.evaluate()
        finally:
            trainer.close()

        assert len(history) == 2
        assert len(evaluation["episode_returns"]) == 2
        assert (Path(temporary_directory) / "last_model.pt").exists()
        assert (Path(temporary_directory) / "training_metrics.csv").exists()


def main() -> None:
    """依次运行 DQN 工程的核心冒烟测试。"""

    check_replay_buffer()
    check_networks()
    check_training_pipeline()
    print("DQN 工程冒烟测试全部通过。")


if __name__ == "__main__":
    main()
