"""CartPole DQN 的正式程序入口。

示例：

    python -m rl_experiments.dqn
    python -m rl_experiments.dqn.train --episodes 10 --device cpu
    python -m rl_experiments.dqn.train --architecture residual_mlp
    python -m rl_experiments.dqn --eval-only --render
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
        "--seed",
        type=int,
        default=None,
        help="训练、验证和测试使用的基础随机种子。",
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
        help="最终独立测试回合数。",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help="每隔多少个训练回合进行一次独立验证。",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=None,
        help="每次独立验证运行的回合数。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="最终测试后打开 CartPole 窗口演示最佳模型。",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成训练曲线和测试对比图。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="训练结果保存目录。",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="跳过训练，直接加载检查点进行独立测试。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="评估模式加载的检查点；默认使用输出目录中的 best_model.pt。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DQNConfig:
    """根据命令行参数创建本次实验配置。"""

    config = create_default_config()

    training_config = config.training
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
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
    if args.validation_interval is not None:
        evaluation_config = replace(
            evaluation_config,
            validation_interval=args.validation_interval,
        )
    if args.validation_episodes is not None:
        evaluation_config = replace(
            evaluation_config,
            validation_episodes=args.validation_episodes,
        )
    if args.render:
        evaluation_config = replace(evaluation_config, render=True)
    if args.no_plots:
        evaluation_config = replace(evaluation_config, save_plots=False)
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
    trainer = DQNTrainer(config, save_config=not args.eval_only)

    try:
        print(f"网络结构：{config.network.architecture}")
        print(f"运行设备：{config.resolve_device()}")
        print(f"结果目录：{config.evaluation.output_dir}")

        if args.eval_only:
            checkpoint_path = (
                args.checkpoint
                if args.checkpoint is not None
                else config.evaluation.output_dir / "best_model.pt"
            )
            trainer.agent.load(checkpoint_path, load_training_state=False)
            best_model_path = checkpoint_path
            print(f"跳过训练，加载模型：{best_model_path}")
        else:
            print(f"开始训练：{config.env_id}")
            trainer.train()
            best_model_path = trainer.load_best_model()

        evaluation_result = trainer.evaluate(render=False)

        print("独立测试完成。")
        print(f"最终测试使用模型：{best_model_path}")
        print(f"评估平均奖励：{evaluation_result['mean_return']:.2f}")
        print(f"评估奖励标准差：{evaluation_result['std_return']:.2f}")
        print(f"评估平均长度：{evaluation_result['mean_length']:.2f}")
        if evaluation_result.get("random_baseline_mean_return") is not None:
            print(
                "随机策略平均奖励："
                f"{evaluation_result['random_baseline_mean_return']:.2f}"
            )
        if evaluation_result["solved_score"] is not None:
            print(f"环境解决标准：{evaluation_result['solved_score']:.2f}")
            print(f"是否达到解决标准：{evaluation_result['is_solved']}")
            print(f"测试成功率：{evaluation_result['success_rate']:.1%}")

        if config.evaluation.save_plots:
            print(f"训练曲线：{config.evaluation.output_dir / 'training_curves.png'}")
            print(f"测试对比图：{config.evaluation.output_dir / 'test_comparison.png'}")

        if config.evaluation.render:
            print("开始显示最佳模型的 CartPole 演示窗口。")
            trainer.demonstrate()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
