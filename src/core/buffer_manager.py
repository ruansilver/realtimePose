"""缓冲管理器：流式聚合EMG样本"""

import torch
import numpy as np


class InputBuffer:
    """输入缓冲区：累积EMG样本直到满窗
    
    Args:
        max_samples: 最大样本数（窗口大小）
    """
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.buffer = []
        self.total_samples = 0
    
    def append(self, samples: torch.Tensor | np.ndarray):
        """追加新样本
        
        Args:
            samples: EMG样本 (C, T) 或 (T, C)
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        
        # 确保格式为 (C, T)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        
        self.buffer.append(samples)
        self.total_samples += samples.shape[-1]
    
    def has_full_window(self) -> bool:
        """检查是否累积满窗"""
        return self.total_samples >= self.max_samples
    
    def get_window(self) -> torch.Tensor:
        """获取完整窗口并清空缓冲
        
        Returns:
            窗口数据 (C, max_samples)
        """
        if not self.has_full_window():
            raise ValueError(
                f"缓冲区样本数不足: {self.total_samples}/{self.max_samples}"
            )
        
        # 拼接所有缓冲样本
        full_buffer = torch.cat(self.buffer, dim=-1)
        
        # 提取窗口
        window = full_buffer[:, :self.max_samples]
        
        # 保留剩余样本
        remaining = full_buffer[:, self.max_samples:]
        self.buffer = [remaining] if remaining.shape[-1] > 0 else []
        self.total_samples = remaining.shape[-1] if remaining.shape[-1] > 0 else 0
        
        return window
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.total_samples = 0
    
    def __len__(self) -> int:
        """返回当前缓冲区样本数"""
        return self.total_samples

