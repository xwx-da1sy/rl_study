"""DQN 的优化器、学习率调度器和梯度裁剪工具。"""

from __future__ import annotations

import torch
from torch import nn

from .config import DQNConfig


def build_optimizer_and_scheduler(
    online_network: nn.Module,
    config: DQNConfig,
    total_updates: int | None = None,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """为 Online Network 创建 AdamW 和余弦退火调度器。

    参数：
        online_network：需要被训练的 Online Network。Target Network 不应传入。
        config：DQN 完整配置。
        total_updates：实际计划进行的优化器更新次数。未提供时使用配置估算值。

    返回：
        ``(optimizer, scheduler)`` 二元组。每次 Online Network 更新后，应该
        调用一次 ``scheduler.step()``。
    """

    if total_updates is None:
        total_updates = config.resolve_scheduler_t_max()
    if total_updates <= 0:
        raise ValueError("total_updates 必须大于 0。")

    optimizer = torch.optim.AdamW(
        online_network.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=config.scheduler.eta_min,
    )
    return optimizer, scheduler


def clip_online_gradients(
    online_network: nn.Module,
    config: DQNConfig,
) -> float | None:
    """按照配置裁剪 Online Network 梯度。

    参数：
        online_network：刚刚完成反向传播的 Online Network。
        config：DQN 完整配置。

    返回：
        如果启用了梯度裁剪，返回裁剪前的梯度范数；如果配置为 ``None``，
        则不进行裁剪并返回 ``None``。
    """

    max_norm = config.optimizer.gradient_clip_norm
    if max_norm is None:
        return None

    gradient_norm = nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_norm=max_norm,
    )
    return float(gradient_norm)
