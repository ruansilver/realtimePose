"""增量重采样器：2kHz→50Hz线性插值"""

import torch
from typing import Optional


class IncrementalResampler:
    """2kHz到50Hz的增量线性插值器
    
    用于流式推理中的实时重采样，维护插值状态以支持连续输出。
    采用线性插值确保与第一阶段整窗插值的数值一致性。
    
    Args:
        input_rate: 输入采样率（默认2000Hz）
        output_rate: 输出采样率（默认50Hz）
        num_channels: 数据通道数
    """
    
    def __init__(
        self,
        input_rate: int = 2000,
        output_rate: int = 50,
        num_channels: int = 64  # encoder输出特征维度
    ):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.num_channels = num_channels
        self.ratio = input_rate / output_rate  # 40.0 (2000/50)
        
        # 插值状态
        self.input_buffer = []  # 输入缓冲区
        self.next_output_time = 0.0  # 下一个输出时间点（以输入采样为单位）
        self.current_input_time = 0.0  # 当前输入时间
        
    def push(self, x: torch.Tensor) -> None:
        """推入新的输入数据
        
        Args:
            x: 输入特征 (time, channel) 或 (channel,) 单帧
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        assert x.shape[1] == self.num_channels, \
            f"通道数不匹配: 期望{self.num_channels}, 收到{x.shape[1]}"
        
        # 添加到输入缓冲区
        for i in range(x.shape[0]):
            self.input_buffer.append((self.current_input_time, x[i].clone()))
            self.current_input_time += 1.0
    
    def pull_next(self) -> Optional[torch.Tensor]:
        """拉取下一个重采样输出
        
        Returns:
            重采样后的特征 (channel,) 或 None（如果数据不足）
        """
        # 检查是否有足够数据进行插值
        if len(self.input_buffer) < 2:
            return None
        
        target_time = self.next_output_time
        
        # 查找插值所需的两个点
        left_idx = None
        right_idx = None
        
        for i, (time, _) in enumerate(self.input_buffer):
            if time <= target_time:
                left_idx = i
            elif time > target_time and right_idx is None:
                right_idx = i
                break
        
        # 如果找不到右边界点，说明数据不足
        if right_idx is None:
            return None
        
        # 执行线性插值
        left_time, left_data = self.input_buffer[left_idx]
        right_time, right_data = self.input_buffer[right_idx]
        
        if right_time == left_time:
            # 时间相同，直接返回
            result = left_data.clone()
        else:
            # 线性插值
            alpha = (target_time - left_time) / (right_time - left_time)
            result = left_data * (1 - alpha) + right_data * alpha
        
        # 清理不再需要的旧数据
        self._cleanup_old_data(target_time)
        
        # 更新下一个输出时间
        self.next_output_time += self.ratio
        
        return result
    
    def can_pull(self) -> bool:
        """检查是否可以拉取下一个输出
        
        Returns:
            是否有足够数据进行插值
        """
        if len(self.input_buffer) < 2:
            return False
        
        target_time = self.next_output_time
        
        # 检查是否有合适的右边界点
        for time, _ in self.input_buffer:
            if time > target_time:
                return True
        
        return False
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """清理不再需要的旧数据
        
        Args:
            current_time: 当前处理的时间点
        """
        # 保留当前时间点之前的至少一个数据点用于插值
        cutoff_time = current_time - self.ratio
        
        while len(self.input_buffer) > 2:
            time, _ = self.input_buffer[0]
            if time < cutoff_time:
                self.input_buffer.pop(0)
            else:
                break
    
    def reset(self) -> None:
        """重置重采样器状态"""
        self.input_buffer.clear()
        self.next_output_time = 0.0
        self.current_input_time = 0.0
    
    def get_pending_outputs(self) -> int:
        """获取可以拉取的输出数量
        
        Returns:
            当前可拉取的输出数量
        """
        count = 0
        temp_time = self.next_output_time
        
        while True:
            # 检查是否有足够数据用于temp_time的插值
            has_right_bound = False
            for time, _ in self.input_buffer:
                if time > temp_time:
                    has_right_bound = True
                    break
            
            if not has_right_bound or len(self.input_buffer) < 2:
                break
            
            count += 1
            temp_time += self.ratio
        
        return count


