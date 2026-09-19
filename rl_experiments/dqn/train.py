"""CartPole DQN 的正式程序入口。

示例：

    python -m rl_experiments.dqn
    python -m rl_experiments.dqn.train --episodes 10 --device cpu
    python -m rl_experiments.dqn.train --architecture residual_mlp
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DQNConfig, create_default_config
from .trainer import DQNTrainer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 CartPole DQN Agent。")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="训练回合数。未指定时使用配置中的默认值。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="每回合最大步数。未指定时使用配置中的默认值。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="运行设备。默认自动选择 CUDA 或 CPU。",
    )
    parser.add_argument(
        "--architecture",
        choices=("mlp", "residual_mlp"),
        default=None,
        help="Q 网络结构。默认使用配置中的架构。",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="训练结束后的评估回合数。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="评估时显示 CartPole 画面。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="训练结果保存目录。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DQNConfig:
    """根据命令行参数创建本次实验配置。"""

    config = create_default_config()

    training_config = config.training
    if args.episodes is not None:
        training_config = replace(training_config, num_episodes=args.episodes)
    if args.max_steps is not None:
        training_config = replace(
            training_config,
            max_steps_per_episode=args.max_steps,
        )
    if args.device is not None:
        training_config = replace(training_config, device=args.device)

    network_config = config.network
    if args.architecture is not None:
        network_config = replace(network_config, architecture=args.architecture)

    evaluation_config = config.evaluation
    if args.eval_episodes is not None:
        evaluation_config = replace(
            evaluation_config,
            num_episodes=args.eval_episodes,
        )
    if args.render:
        evaluation_config = replace(evaluation_config, render=True)
    if args.output_dir is not None:
        evaluation_config = replace(
            evaluation_config,
            output_dir=args.output_dir,
        )

    return replace(
        config,
        network=network_config,
        training=training_config,
        evaluation=evaluation_config,
    )


def main() -> None:
    """创建 Trainer，执行训练和评估，然后关闭环境。"""

    args = parse_args()
    config = build_config(args)
    trainer = DQNTrainer(config)

    try:
        print(f"开始训练：{config.env_id}")
        print(f"网络结构：{config.network.architecture}")
        print(f"运行设备：{config.resolve_device()}")
        print(f"结果目录：{config.evaluation.output_dir}")

        trainer.train()
        evaluation_result = trainer.evaluate()

        print("训练完成。")
        print(f"评估平均奖励：{evaluation_result['mean_return']:.2f}")
        print(f"评估奖励标准差：{evaluation_result['std_return']:.2f}")
        print(f"评估平均长度：{evaluation_result['mean_length']:.2f}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
