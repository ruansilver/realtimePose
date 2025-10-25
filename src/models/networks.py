# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
神经网络组件

包含实时姿态预测所需的各种神经网络模块和架构。
"""

import collections
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


##################
# TDS FEATURIZER #
##################


class Permute(nn.Module):
    """排列张量维度的模块。
    
    例如:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "from_dims和to_dims的维度数量应该相同"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Permute目前只支持3、4、5和6维张量"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        """获取反向排列操作以恢复原始维度顺序"""
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """nn.BatchNorm1d的包装，适用于NTC格式"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    """一维卷积块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        """带填充的一维卷积，使输入输出长度匹配"""

        super().__init__()

        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)

        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConv2dBlock(nn.Module):
    """时间深度可分离二维卷积块
    
    基于论文 "Sequence-to-Sequence Speech Recognition with Time-Depth 
    Separable Convolutions" (Hannun et al, https://arxiv.org/abs/1904.02619)

    Args:
        channels (int): 输入和输出通道数。对于形状为(T, N, num_features)的输入，
            需要满足 channels * width = num_features
        width (int): 输入宽度。对于形状为(T, N, num_features)的输入，
            需要满足 channels * width = num_features  
        kernel_width (int): 时间卷积的核大小
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width必须为奇数"
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, T = inputs.shape  # BCT

        # BCT -> BcwT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)  # BcwT -> BCT

        # 下采样后的跳跃连接
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]

        # 在C维度上进行层归一化
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSFullyConnectedBlock(nn.Module):
    """全连接块
    
    基于论文 "Sequence-to-Sequence Speech Recognition with Time-Depth 
    Separable Convolutions" (Hannun et al, https://arxiv.org/abs/1904.02619)

    Args:
        num_features (int): 输入形状(T, N, num_features)中的num_features
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # 在C维度上进行层归一化
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSConvEncoder(nn.Module):
    """时间深度可分离卷积编码器
    
    由TDSConv2dBlock和TDSFullyConnectedBlock组成的序列，
    基于论文 "Sequence-to-Sequence Speech Recognition with Time-Depth 
    Separable Convolutions" (Hannun et al, https://arxiv.org/abs/1904.02619)

    Args:
        num_features (int): 输入形状(T, N, num_features)中的num_features
        block_channels (list): 整数列表，指示每个TDSConv2dBlock的通道数
        kernel_width (int): 时间卷积的核大小
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels}必须能整除num_features {num_features}"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TdsStage(nn.Module):
    """TDS阶段模块"""
    
    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        """由几个TdsBlock组成的阶段，前面有一个非可分离的子采样卷积
        
        初始（可选的子采样）卷积层将输入通道数映射到残差TDS块使用的相应内部宽度。
        
        遵循来自 https://arxiv.org/abs/1904.02619 的多阶段网络构建方法。
        """

        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        C = channels * feature_width

        self.out_channels = out_channels

        # Conv1d块
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                C,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        elif in_channels != C:
            # 检查in_channels是否与TDS通道和特征宽度一致
            raise ValueError(
                f"in_channels ({in_channels})必须等于channels * "
                f"feature_width ({channels} * {feature_width})，如果"
                " in_conv_kernel_width不是正数"
            )

        # TDS块
        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        # 线性投影
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TdsNetwork(nn.Module):
    """TDS网络"""
    
    def __init__(
        self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        """计算左上下文"""
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for tds_stage in tds_stages:
            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride

        return left


#############
# NEUROPOSE #
#############


class EncoderBlock(nn.Module):
    """编码器块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        max_pool_size: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        maxpool = nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
        self.network = nn.Sequential(conv, bn, relu, dropout, maxpool)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        num_convs: int,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        def _conv(in_channels: int, out_channels: int):
            """单个卷积块"""
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        modules = [*_conv(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            modules += _conv(out_channels, out_channels)

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.network(x)


class DecoderBlock(nn.Module):
    """解码器块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        upsampling: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        scale_factor = (float(upsampling[0]), float(upsampling[1]))
        upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.network = nn.Sequential(conv, bn, relu, dropout, upsample)
        self.out_channels = out_channels

    def forward(self, x):
        return self.network(x)


class NeuroPose(nn.Module):
    """NeuroPose网络"""
    
    def __init__(
        self,
        encoder_blocks: list[EncoderBlock],
        residual_blocks: list[ResidualBlock],
        decoder_blocks: list[DecoderBlock],
        linear_in_channels: int,
        out_channels: int = 22,
    ):
        super().__init__()
        self.network = nn.Sequential(*encoder_blocks, *residual_blocks, *decoder_blocks)
        self.linear = nn.Linear(linear_in_channels, out_channels)
        self.left_context = 0
        self.right_context = 0

    def forward(self, x):
        # NeuroPose在时间和空间上使用2D卷积，因此我们添加对应于网络特征的新通道维度
        x = x[:, None].swapaxes(-1, -2)  # BCT -> BCtc
        x = self.network(x)
        x = x.swapaxes(-2, -3).flatten(-2)  # BCtc -> BTC
        return self.linear(x).swapaxes(-1, -2)  # BTC -> BCT


############
# DECODERS #
############


class MLP(nn.Module):
    """基础MLP，支持可选的最终输出缩放"""

    def __init__(
        self,
        in_channels: int,
        layer_sizes: list[int],
        out_channels: int,
        layer_norm: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        sizes = [in_channels] + layer_sizes
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        # x的形状为(batch, channel)
        return self.mlp(x) * self.scale


class SequentialLSTM(nn.Module):
    """
    顺序LSTM，每次forward()调用只计算单个时间步，
    以兼容手动循环时间的方式。

    注意：需要在外部上下文中每个轨迹后手动重置状态！
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mlp_out = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(hidden_size, out_channels)
        )
        self.scale = scale

    def reset_state(self):
        """重置隐藏状态"""
        self.hidden = None

    def forward(self, x):
        """单个时间步的前向传播，其中x的形状为(batch, channel)"""

        if self.hidden is None:
            # 用零初始化隐藏状态
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (torch.zeros(*size).to(device), torch.zeros(*size).to(device))

        out, self.hidden = self.lstm(x[:, None], self.hidden)
        return self.mlp_out(out[:, 0]) * self.scale

    def _non_sequential_forward(self, x):
        """非顺序前向传播，其中x的形状为(batch, time, channel)"""
        return self.mlp_out(self.lstm(x)[0]) * self.scale
