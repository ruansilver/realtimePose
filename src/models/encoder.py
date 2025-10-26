"""TDS编码器：时序深度可分离卷积网络"""

import collections
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


class Permute(nn.Module):
    """维度置换模块
    
    示例: Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    """
    
    def __init__(self, from_dims: str, to_dims: str):
        super().__init__()
        assert len(from_dims) == len(to_dims), \
            "from_dims和to_dims维度数量必须相同"
        
        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError("仅支持3-6维张量的置换")
        
        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)
    
    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"


class BatchNorm1d(nn.Module):
    """NTC格式的BatchNorm1d包装"""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    """1D卷积块（padding=0，用于序列下采样）"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
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
    """TDS 2D时序卷积块
    
    参考: "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions" (Hannun et al, 2019)
    
    Args:
        channels: 输入输出通道数
        width: 特征宽度 (channels * width = num_features)
        kernel_width: 时序卷积核大小
    """
    
    def __init__(self, channels: int, width: int, kernel_width: int):
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
        
        # 跳跃连接（下采样后对齐）
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]
        
        # LayerNorm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        
        return x


class TDSFullyConnectedBlock(nn.Module):
    """TDS全连接块"""
    
    def __init__(self, num_features: int):
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
        
        # LayerNorm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        
        return x


class TDSConvEncoder(nn.Module):
    """TDS卷积编码器：堆叠多个TDS块"""
    
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ):
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)
        
        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels} 必须整除 num_features {num_features}"
            tds_conv_blocks.extend([
                TDSConv2dBlock(channels, feature_width, kernel_width),
                TDSFullyConnectedBlock(num_features),
            ])
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)


class TdsStage(nn.Module):
    """TDS阶段：Conv1d下采样 + TDS块 + 可选线性投影"""
    
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
            raise ValueError(
                f"in_channels ({in_channels}) 必须等于 channels * "
                f"feature_width ({channels} * {feature_width}) "
                "如果 in_conv_kernel_width 不为正"
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
    """完整TDS网络：多个Conv1d块 + 多个TDS阶段"""
    
    def __init__(
        self,
        conv_blocks: Sequence[Conv1dBlock],
        tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0
    
    def forward(self, x):
        return self.layers(x)
    
    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        """计算网络的左上下文长度（因果卷积的接收野）"""
        left, stride = 0, 1
        
        # Conv1d块的贡献
        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride
        
        # TDS阶段的贡献
        for tds_stage in tds_stages:
            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride
            
            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride
        
        return left


def build_tds_network_from_config(config: dict) -> TdsNetwork:
    """根据配置构建TDS网络
    
    Args:
        config: 网络配置字典
        
    Returns:
        TdsNetwork实例
    """
    # 构建Conv1d块
    conv_blocks = []
    for block_cfg in config['conv_blocks']:
        conv_blocks.append(Conv1dBlock(
            in_channels=block_cfg['in_channels'],
            out_channels=block_cfg['out_channels'],
            kernel_size=block_cfg['kernel_size'],
            stride=block_cfg['stride'],
        ))
    
    # 构建TDS阶段
    tds_stages = []
    for stage_cfg in config['tds_stages']:
        tds_stages.append(TdsStage(
            in_channels=stage_cfg['in_channels'],
            in_conv_kernel_width=stage_cfg['in_conv_kernel_width'],
            in_conv_stride=stage_cfg['in_conv_stride'],
            num_blocks=stage_cfg['num_blocks'],
            channels=stage_cfg['channels'],
            feature_width=stage_cfg['feature_width'],
            kernel_width=stage_cfg['kernel_width'],
            out_channels=stage_cfg.get('out_channels', None),
        ))
    
    return TdsNetwork(conv_blocks, tds_stages)

