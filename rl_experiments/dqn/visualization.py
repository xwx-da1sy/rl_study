"""DQN 训练和测试结果可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

import matplotlib
import numpy as np

# 训练曲线直接保存为 PNG，不依赖桌面图形窗口。
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class EpisodeMetrics(Protocol):
    """绘制训练曲线所需的回合指标接口。"""

    episode: int
    episode_return: float
    mean_loss: float | None
    epsilon: float
    learning_rate: float


class ValidationMetrics(Protocol):
    """绘制验证曲线所需的指标接口。"""

    episode: int
    mean_return: float


def moving_average(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """计算一维序列的滑动平均。

    参数：
        values：需要平滑的数值序列。
        window：滑动窗口长度。

    返回：
        ``(位置, 平滑结果)``。当数据量小于窗口时返回两个空数组。
    """

    value_array = np.asarray(values, dtype=np.float64)
    if window <= 0:
        raise ValueError("window 必须大于 0。")
    if len(value_array) < window:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    kernel = np.ones(window, dtype=np.float64) / window
    averages = np.convolve(value_array, kernel, mode="valid")
    positions = np.arange(window, len(value_array) + 1)
    return positions, averages


def save_training_curves(
    history: Sequence[EpisodeMetrics],
    validation_history: Sequence[ValidationMetrics],
    output_path: str | Path,
    moving_average_window: int,
    solved_score: float | None,
) -> Path:
    """保存奖励、损失、探索率和学习率四张训练曲线。"""

    if not history:
        raise ValueError("history 不能为空。")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.asarray([record.episode for record in history])
    returns = np.asarray([record.episode_return for record in history])
    losses = np.asarray(
        [np.nan if record.mean_loss is None else record.mean_loss for record in history]
    )
    epsilons = np.asarray([record.epsilon for record in history])
    learning_rates = np.asarray([record.learning_rate for record in history])

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    reward_axis = axes[0, 0]
    reward_axis.plot(episodes, returns, color="#8fb9e1", alpha=0.55, label="Episode return")
    moving_positions, smoothed_returns = moving_average(returns, moving_average_window)
    if len(smoothed_returns) > 0:
        reward_axis.plot(
            moving_positions,
            smoothed_returns,
            color="#1565c0",
            linewidth=2.0,
            label=f"Moving average ({moving_average_window})",
        )
    if validation_history:
        reward_axis.plot(
            [record.episode for record in validation_history],
            [record.mean_return for record in validation_history],
            color="#ef6c00",
            marker="o",
            linewidth=1.5,
            label="Validation mean",
        )
    if solved_score is not None:
        reward_axis.axhline(
            solved_score,
            color="#2e7d32",
            linestyle="--",
            linewidth=1.5,
            label=f"Solved threshold ({solved_score:g})",
        )
    reward_axis.set(title="Training and validation returns", xlabel="Episode", ylabel="Return")
    reward_axis.legend(fontsize=8)

    loss_axis = axes[0, 1]
    valid_loss_mask = np.isfinite(losses)
    if valid_loss_mask.any():
        loss_axis.plot(episodes[valid_loss_mask], losses[valid_loss_mask], color="#c62828")
    else:
        loss_axis.text(0.5, 0.5, "Waiting for replay-buffer warm-up", ha="center", va="center")
    loss_axis.set(title="Mean TD loss", xlabel="Episode", ylabel="Huber loss")

    epsilon_axis = axes[1, 0]
    epsilon_axis.plot(episodes, epsilons, color="#6a1b9a")
    epsilon_axis.set(title="Exploration schedule", xlabel="Episode", ylabel="Epsilon")
    epsilon_axis.set_ylim(0.0, 1.05)

    learning_rate_axis = axes[1, 1]
    learning_rate_axis.plot(episodes, learning_rates, color="#00838f")
    learning_rate_axis.set(title="Learning-rate schedule", xlabel="Episode", ylabel="Learning rate")
    if np.all(learning_rates > 0):
        learning_rate_axis.set_yscale("log")

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    figure.suptitle("CartPole DQN training diagnostics", fontsize=14)
    figure.tight_layout()
    figure.savefig(target_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return target_path


def save_test_comparison(
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """保存 DQN 与随机策略在独立测试场景上的奖励对比图。"""

    learned_returns = np.asarray(result["episode_returns"], dtype=np.float64)
    random_returns = np.asarray(result.get("random_baseline_returns", []), dtype=np.float64)
    if len(learned_returns) == 0:
        raise ValueError("测试奖励不能为空。")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    episode_numbers = np.arange(1, len(learned_returns) + 1)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        episode_numbers,
        learned_returns,
        color="#1565c0",
        marker="o",
        linewidth=1.8,
        label=f"DQN (mean={learned_returns.mean():.1f})",
    )
    if len(random_returns) > 0:
        axis.plot(
            episode_numbers,
            random_returns,
            color="#9e9e9e",
            marker="x",
            linewidth=1.2,
            label=f"Random policy (mean={random_returns.mean():.1f})",
        )

    solved_score = result.get("solved_score")
    if solved_score is not None:
        axis.axhline(
            float(solved_score),
            color="#2e7d32",
            linestyle="--",
            label=f"Solved threshold ({float(solved_score):g})",
        )

    axis.set(
        title="Independent test episodes",
        xlabel="Test episode",
        ylabel="Return",
    )
    if len(episode_numbers) <= 25:
        axis.set_xticks(episode_numbers)
    else:
        # 测试回合较多时只显示少量均匀刻度，避免标签重叠。
        sparse_ticks = np.unique(
            np.linspace(1, len(episode_numbers), num=11, dtype=np.int64)
        )
        axis.set_xticks(sparse_ticks)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return target_path
