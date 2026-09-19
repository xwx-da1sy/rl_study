"""DQN 算法使用的神经网络。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QNetwork(nn.Module):
    """使用多层感知机估计所有动作 Q 值的网络。

    对于 CartPole，默认网络结构为 4 -> 256 -> 128 -> 128 -> 2：

        4 -> 256 -> 128 -> 128 -> 2

    DQN 会使用这个类实例化两次：一次作为在线网络，一次作为目标网络。
    虽然两者的网络结构相同，但它们是参数相互独立的两个实例。

    参数：
        state_dim：一次环境观测包含的数值数量。
        action_dim：环境中的离散动作数量。
        hidden_dim_1：第一层隐藏层的宽度，默认值为 256。
        hidden_dim_2：第二层隐藏层的宽度，默认值为 128。
        hidden_dim_3：第三层隐藏层的宽度，默认值为 128。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        hidden_dim_3: int = 128,
    ) -> None:
        """初始化 Q 值估计网络的所有层。"""

        super().__init__()

        # 第一层全连接层：将 4 维原始状态映射为 256 维特征。
        self.input_layer = nn.Linear(in_features=state_dim, out_features=hidden_dim_1)

        # 输入映射之后的非线性激活层。
        self.input_activation = nn.ReLU()

        # 第二层全连接层：将 256 维特征压缩为 128 维特征。
        self.hidden_layer_1 = nn.Linear(
            in_features=hidden_dim_1,
            out_features=hidden_dim_2,
        )

        # 第二层隐藏层之后的非线性激活层。
        self.hidden_activation_1 = nn.ReLU()

        # 第三层全连接层：继续提取和组合 128 维状态特征。
        self.hidden_layer_2 = nn.Linear(
            in_features=hidden_dim_2,
            out_features=hidden_dim_3,
        )

        # 第三层隐藏层之后的非线性激活层。
        self.hidden_activation_2 = nn.ReLU()

        # 为每个可用动作输出一个标量 Q 值。
        # 不使用 Softmax，因为 Q 值不是概率。
        self.output_layer = nn.Linear(in_features=hidden_dim_3, out_features=action_dim)

    def forward(self, states: Tensor) -> Tensor:
        """计算单个状态或一批状态对应的 Q 值。

        参数：
            states：形状为 ``[state_dim]`` 或 ``[batch_size, state_dim]`` 的张量。
                输入网络前会被转换为 float32。

        返回：
            对于单个状态，返回形状为 ``[action_dim]`` 的张量；对于一个 batch，
            返回形状为 ``[batch_size, action_dim]`` 的张量。每一列对应一个动作
            的估计 Q 值。
        """

        states = states.float()
        features = self.input_layer(states)
        features = self.input_activation(features)
        features = self.hidden_layer_1(features)
        features = self.hidden_activation_1(features)
        features = self.hidden_layer_2(features)
        features = self.hidden_activation_2(features)
        q_values = self.output_layer(features)
        return q_values


class ResidualBlock(nn.Module):
    """保持特征维度不变的全连接残差块。"""

    def __init__(self, feature_dim: int) -> None:
        """初始化残差块中的两层全连接层。"""

        super().__init__()

        # 残差主路径的第一层全连接层。
        self.layer_1 = nn.Linear(feature_dim, feature_dim)

        # 主路径中的非线性激活层。
        self.activation_1 = nn.ReLU()

        # 残差主路径的第二层全连接层。
        self.layer_2 = nn.Linear(feature_dim, feature_dim)

        # 残差相加之后的输出激活层。
        self.output_activation = nn.ReLU()

    def forward(self, features: Tensor) -> Tensor:
        """计算残差块输出。

        参数：
            features：形状为 ``[feature_dim]`` 或
                ``[batch_size, feature_dim]`` 的特征张量。

        返回：
            与输入形状相同的特征张量。
        """

        residual = features
        transformed = self.layer_1(features)
        transformed = self.activation_1(transformed)
        transformed = self.layer_2(transformed)
        return self.output_activation(transformed + residual)


class ResidualQNetwork(nn.Module):
    """使用全连接残差块估计动作 Q 值的网络。

    默认结构为：

        4 -> 256 -> ResidualBlock(256) -> 128
          -> ResidualBlock(128) -> 2

    参数：
        state_dim：一次环境观测包含的数值数量。
        action_dim：环境中的离散动作数量。
        hidden_dim_1：第一个残差阶段的特征宽度。
        hidden_dim_2：投影后的特征宽度。
        hidden_dim_3：第二个残差阶段的特征宽度。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        hidden_dim_3: int = 128,
    ) -> None:
        """初始化残差 Q 网络的所有层。"""

        super().__init__()

        # 输入层：将环境状态映射为高维特征。
        self.input_layer = nn.Linear(state_dim, hidden_dim_1)
        self.input_activation = nn.ReLU()

        # 第一个残差阶段保持 hidden_dim_1 维度不变。
        self.residual_block_1 = ResidualBlock(hidden_dim_1)

        # 投影层：将第一阶段特征转换到第二阶段的维度。
        self.projection_layer = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.projection_activation = nn.ReLU()

        # 第二个残差阶段保持 hidden_dim_3 维度不变。
        # 当前默认 hidden_dim_2 和 hidden_dim_3 都是 128。
        if hidden_dim_2 != hidden_dim_3:
            raise ValueError("ResidualQNetwork 要求 hidden_dim_2 等于 hidden_dim_3。")
        self.residual_block_2 = ResidualBlock(hidden_dim_3)

        # 输出层：为每个离散动作输出一个 Q 值。
        self.output_layer = nn.Linear(hidden_dim_3, action_dim)

    def forward(self, states: Tensor) -> Tensor:
        """计算单个状态或一批状态对应的 Q 值。"""

        states = states.float()
        features = self.input_layer(states)
        features = self.input_activation(features)
        features = self.residual_block_1(features)
        features = self.projection_layer(features)
        features = self.projection_activation(features)
        features = self.residual_block_2(features)
        return self.output_layer(features)


def build_q_network(
    architecture: str,
    state_dim: int,
    action_dim: int,
    hidden_dim_1: int = 256,
    hidden_dim_2: int = 128,
    hidden_dim_3: int = 128,
) -> nn.Module:
    """根据架构名称创建 Q 网络。

    参数：
        architecture：``"mlp"`` 或 ``"residual_mlp"``。
        state_dim：一次环境观测包含的数值数量。
        action_dim：环境中的离散动作数量。
        hidden_dim_1：第一阶段隐藏特征宽度。
        hidden_dim_2：第二阶段隐藏特征宽度。
        hidden_dim_3：第三阶段隐藏特征宽度。

    返回：
        一个可以输出所有动作 Q 值的神经网络。
    """

    network_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "hidden_dim_3": hidden_dim_3,
    }

    if architecture == "mlp":
        return QNetwork(**network_kwargs)
    if architecture == "residual_mlp":
        return ResidualQNetwork(**network_kwargs)
    raise ValueError("architecture 必须是 'mlp' 或 'residual_mlp'。")


class DQNNetworks:
    """管理 DQN 使用的 Online Network 和 Target Network。

    两个网络使用完全相同的 QNetwork 结构，但拥有各自独立的参数：

    - Online Network：每次梯度更新都会改变参数；
    - Target Network：只用于计算 TD target，按固定间隔同步参数。

    参数：
        state_dim：一次环境观测包含的数值数量。
        action_dim：环境中的离散动作数量。
        hidden_dim_1：第一层隐藏层的宽度。
        hidden_dim_2：第二层隐藏层的宽度。
        hidden_dim_3：第三层隐藏层的宽度。
        device：网络运行的设备，例如 ``"cpu"`` 或 ``"cuda"``。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        hidden_dim_3: int = 128,
        architecture: str = "mlp",
        device: torch.device | str = "cpu",
    ) -> None:
        """创建两个独立的 Q 网络，并让目标网络初始同步在线网络。"""

        self.device = torch.device(device)

        # Online Network：后续会由优化器更新参数。
        self.online_network = build_q_network(
            architecture=architecture,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            hidden_dim_3=hidden_dim_3,
        ).to(self.device)

        # Target Network：用于计算稳定的 TD target，不参与梯度更新。
        self.target_network = build_q_network(
            architecture=architecture,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            hidden_dim_3=hidden_dim_3,
        ).to(self.device)

        self.sync_target_network()

        # 关闭目标网络的梯度，避免它被意外传入反向传播和优化器。
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

    def sync_target_network(self) -> None:
        """将 Online Network 的参数复制到 Target Network。"""

        self.target_network.load_state_dict(self.online_network.state_dict())

    def train_online_network(self) -> None:
        """将 Online Network 切换到训练模式。"""

        self.online_network.train()

    def evaluate(self) -> None:
        """将两个网络切换到评估模式。"""

        self.online_network.eval()
        self.target_network.eval()
