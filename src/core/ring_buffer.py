"""环形缓冲器：2kHz流式数据管理"""

import torch
from typing import Optional


class RingBuffer:
    """2kHz环形缓冲器，用于流式推理的数据管理
    
    维护一个固定大小的环形缓冲区，支持连续数据追加和按上下文切片。
    专为流式推理设计，每20ms触发一次推理步骤。
    
    Args:
        max_samples: 缓冲区最大样本数
        num_channels: 数据通道数（EMG为16）
        sample_rate: 采样率（默认2000Hz）
        step_ms: 推理步长（默认20ms对应50Hz）
    """
    
    def __init__(
        self,
        max_samples: int,
        num_channels: int = 16,
        sample_rate: int = 2000,
        step_ms: int = 20
    ):
        self.max_samples = max_samples
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.step_ms = step_ms
        self.step_samples = int(sample_rate * step_ms / 1000)  # 20ms = 40 samples at 2kHz
        
        # 环形缓冲区
        self.buffer = torch.zeros((max_samples, num_channels), dtype=torch.float32)
        self.write_ptr = 0  # 写指针
        self.total_samples = 0  # 总接收样本数
        self.last_step_samples = 0  # 上次步进时的样本数
        
    def append(self, x: torch.Tensor) -> None:
        """追加新数据到缓冲区
        
        Args:
            x: 输入数据 (time, channel) 或 (channel,) 单帧
        """
        if x.ndim == 1:
            # 单帧数据，添加时间维度
            x = x.unsqueeze(0)
        
        assert x.shape[1] == self.num_channels, \
            f"通道数不匹配: 期望{self.num_channels}, 收到{x.shape[1]}"
        
        num_new_samples = x.shape[0]
        
        # 写入环形缓冲区
        for i in range(num_new_samples):
            self.buffer[self.write_ptr] = x[i]
            self.write_ptr = (self.write_ptr + 1) % self.max_samples
            self.total_samples += 1
    
    def can_step(self, step_ms: Optional[int] = None) -> bool:
        """检查是否可以进行下一步推理
        
        Args:
            step_ms: 步长毫秒数，如果None则使用默认值
            
        Returns:
            是否有足够新数据进行推理
        """
        if step_ms is None:
            step_ms = self.step_ms
        
        step_samples = int(self.sample_rate * step_ms / 1000)
        samples_since_last_step = self.total_samples - self.last_step_samples
        
        return samples_since_last_step >= step_samples
    
    def slice_for_context(self, left_context: int) -> torch.Tensor:
        """为推理获取上下文切片
        
        Args:
            left_context: 左上下文长度（样本数）
            
        Returns:
            上下文数据 (time, channel)
        """
        if self.total_samples < left_context:
            raise ValueError(
                f"缓冲区数据不足: 需要{left_context}样本, 当前{self.total_samples}"
            )
        
        # 计算读取范围
        available_samples = min(self.total_samples, self.max_samples)
        start_offset = max(0, available_samples - left_context)
        
        # 从环形缓冲区读取
        if self.write_ptr >= start_offset:
            # 连续读取
            end_ptr = self.write_ptr
            start_ptr = end_ptr - (available_samples - start_offset)
            result = self.buffer[start_ptr:end_ptr].clone()
        else:
            # 跨越边界读取
            samples_needed = available_samples - start_offset
            part1_size = self.max_samples - self.write_ptr
            part2_size = samples_needed - part1_size
            
            if part1_size > 0:
                part1 = self.buffer[self.max_samples - part1_size:].clone()
            else:
                part1 = torch.empty(0, self.num_channels)
            
            if part2_size > 0:
                part2 = self.buffer[:part2_size].clone()
                result = torch.cat([part1, part2], dim=0)
            else:
                result = part1
        
        # 截取所需长度
        if result.shape[0] > left_context:
            result = result[-left_context:]
        
        return result
    
    def mark_step_completed(self) -> None:
        """标记当前推理步骤已完成"""
        self.last_step_samples = self.total_samples
    
    def get_current_samples(self) -> int:
        """获取当前总样本数"""
        return self.total_samples
    
    def get_samples_since_last_step(self) -> int:
        """获取自上次步进以来的新样本数"""
        return self.total_samples - self.last_step_samples
    
    def reset(self) -> None:
        """重置缓冲区状态"""
        self.buffer.zero_()
        self.write_ptr = 0
        self.total_samples = 0
        self.last_step_samples = 0