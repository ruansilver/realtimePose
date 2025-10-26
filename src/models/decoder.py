"""LSTM解码器：单步递归预测"""

import torch
from torch import nn


class SequentialLSTM(nn.Module):
    """单步LSTM解码器
    
    每次forward()调用仅计算单个时间步，用于手动循环rollout。
    
    注意：需要在每个轨迹后手动调用reset_state()重置状态！
    
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        scale: 输出缩放系数
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
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels)
        )
        self.scale = scale
    
    def reset_state(self):
        """重置LSTM隐藏状态"""
        self.hidden = None
    
    def forward(self, x):
        """单步前向传播
        
        Args:
            x: 输入 (batch, channel)
            
        Returns:
            输出 (batch, out_channels)
        """
        if self.hidden is None:
            # 初始化隐藏状态为零
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (
                torch.zeros(*size).to(device), 
                torch.zeros(*size).to(device)
            )
        
        out, self.hidden = self.lstm(x[:, None], self.hidden)
        return self.mlp_out(out[:, 0]) * self.scale
    
    def _non_sequential_forward(self, x):
        """非递归前向传播（用于批量处理）
        
        Args:
            x: 输入 (batch, time, channel)
            
        Returns:
            输出 (batch, time, out_channels)
        """
        return self.mlp_out(self.lstm(x)[0]) * self.scale


def build_decoder_from_config(config: dict) -> SequentialLSTM:
    """根据配置构建解码器
    
    Args:
        config: 解码器配置字典
        
    Returns:
        SequentialLSTM实例
    """
    return SequentialLSTM(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        scale=config.get('scale', 1.0),
    )

