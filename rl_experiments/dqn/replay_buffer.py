"""DQN 使用的经验回放池。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Transition:
    """环境与智能体交互产生的一条经验。"""

    state: NDArray[np.float32]
    action: int
    reward: float
    next_state: NDArray[np.float32]
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """从经验回放池中随机采样得到的一批经验。

    数组的第一维都是 batch 维度。这里暂时返回 NumPy 数组，之后由 Agent
    根据运行设备统一转换为 PyTorch Tensor。
    """

    states: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_states: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]

    @property
    def batch_size(self) -> int:
        """返回当前 Batch 中包含的经验数量。"""

        return int(self.states.shape[0])

    def __iter__(self) -> Iterator[NDArray[np.generic]]:
        """按 DQN 更新常用的顺序迭代 Batch 字段。"""

        yield self.states
        yield self.actions
        yield self.rewards
        yield self.next_states
        yield self.terminated
        yield self.truncated


class ReplayBuffer:
    """固定容量的随机经验回放池。

    经验达到容量上限后，新的经验会覆盖最旧的经验。训练时从其中随机
    采样 Batch，打破连续环境样本之间的相关性。

    参数：
        capacity：最多保存的经验数量。
        seed：随机采样使用的种子。设置后可以复现采样顺序。
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        """创建一个空的固定容量经验回放池。"""

        if capacity <= 0:
            raise ValueError("ReplayBuffer 的容量必须大于 0。")

        self.capacity = capacity
        self._buffer: list[Transition | None] = [None] * capacity
        self._next_index = 0
        self._size = 0
        self._random = random.Random(seed)

    def push(
        self,
        state: NDArray[np.float32] | np.ndarray,
        action: int,
        reward: float,
        next_state: NDArray[np.float32] | np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """保存一条环境交互经验。

        参数：
            state：执行动作前的状态。
            action：智能体执行的离散动作。
            reward：环境对该动作返回的即时奖励。
            next_state：执行动作后得到的下一个状态。
            terminated：环境是否真正进入终止状态。
            truncated：环境是否因为时间限制等外部条件截断。

        返回：
            ``None``。当回放池已满时，新经验会覆盖最旧经验。
        """

        state_array = np.asarray(state, dtype=np.float32).copy()
        next_state_array = np.asarray(next_state, dtype=np.float32).copy()

        self._buffer[self._next_index] = Transition(
            state=state_array,
            action=int(action),
            reward=float(reward),
            next_state=next_state_array,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """随机采样一个 Batch。

        参数：
            batch_size：需要采样的经验数量。当前实现不重复采样。

        返回：
            包含状态、动作、奖励、下一个状态和终止标记的
            :class:`TransitionBatch`。

        异常：
            当 batch_size 不合法，或回放池中的经验不足时抛出 ``ValueError``。
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        if batch_size > self._size:
            raise ValueError(
                f"经验数量不足：当前有 {self._size} 条经验，无法采样 {batch_size} 条。"
            )

        indices = self._random.sample(range(self._size), batch_size)
        transitions = [self._buffer[index] for index in indices]

        # 由于 _size 始终表示有效经验数量，这里不会出现 None。
        valid_transitions = [transition for transition in transitions if transition is not None]
        if len(valid_transitions) != batch_size:
            raise RuntimeError("ReplayBuffer 内部存在无效经验。")

        return TransitionBatch(
            states=np.stack([transition.state for transition in valid_transitions]).astype(
                np.float32,
                copy=False,
            ),
            actions=np.asarray(
                [transition.action for transition in valid_transitions],
                dtype=np.int64,
            ),
            rewards=np.asarray(
                [transition.reward for transition in valid_transitions],
                dtype=np.float32,
            ),
            next_states=np.stack(
                [transition.next_state for transition in valid_transitions]
            ).astype(np.float32, copy=False),
            terminated=np.asarray(
                [transition.terminated for transition in valid_transitions],
                dtype=np.bool_,
            ),
            truncated=np.asarray(
                [transition.truncated for transition in valid_transitions],
                dtype=np.bool_,
            ),
        )

    def can_sample(self, batch_size: int) -> bool:
        """判断当前回放池是否至少包含一个指定大小的 Batch。"""

        return 0 < batch_size <= self._size

    def clear(self) -> None:
        """清空回放池中的全部经验。"""

        self._buffer = [None] * self.capacity
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        """返回当前有效经验数量。"""

        return self._size
