"""TDS流式编码器：增量卷积实现"""

import torch
from torch import nn
from typing import Dict, List, Optional, Tuple
import math

from src.models.encoder import TdsNetwork


class StreamingTdsLayer:
    """单层TDS的流式包装器
    
    为含stride的卷积层实现overlap-save增量计算，
    维护必要的历史输入缓存以支持连续推理。
    """
    
    def __init__(self, layer: nn.Module, kernel_size: int, stride: int):
        self.layer = layer
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 计算所需的历史缓存长度
        self.history_length = kernel_size - 1
        self.input_buffer: Optional[torch.Tensor] = None
        
        # 输出对齐：每stride个输入产生一个输出
        self.input_count = 0
        self.pending_inputs = []
    
    def reset(self):
        """重置层状态"""
        self.input_buffer = None
        self.input_count = 0
        self.pending_inputs.clear()
    
    def can_produce_output(self) -> bool:
        """检查是否可以产生输出"""
        return len(self.pending_inputs) >= self.stride
    
    def push_input(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """推入新输入并尝试产生输出
        
        Args:
            x: 输入张量 (B, C, T)
            
        Returns:
            输出张量 (B, C', T') 或 None
        """
        batch_size, channels, time_steps = x.shape
        self.pending_inputs.append(x)
        
        # 检查是否可以产生输出
        if not self.can_produce_output():
            return None
        
        # 收集stride个输入进行卷积
        inputs_to_process = self.pending_inputs[:self.stride]
        self.pending_inputs = self.pending_inputs[self.stride:]
        
        # 拼接输入
        combined_input = torch.cat(inputs_to_process, dim=2)  # (B, C, stride)
        
        # 如果有历史缓存，将其添加到前面
        if self.input_buffer is not None:
            combined_input = torch.cat([self.input_buffer, combined_input], dim=2)
        
        # 执行卷积
        output = self.layer(combined_input)
        
        # 更新历史缓存
        if self.history_length > 0:
            self.input_buffer = combined_input[:, :, -self.history_length:].clone()
        
        return output


class StreamingTdsNetwork:
    """TDS网络的流式包装器
    
    对TdsNetwork进行增量包装，实现overlap-save算法。
    支持逐步推入输入数据，并在有足够数据时产生输出。
    
    Args:
        base_network: 基础TDS网络
        step_size: 每步推入的样本数（对应20ms@2kHz = 40样本）
    """
    
    def __init__(self, base_network: TdsNetwork, step_size: int = 40):
        self.base_network = base_network
        self.step_size = step_size
        
        # 分析网络结构，创建流式层
        self.streaming_layers = []
        self._analyze_network_structure()
        
        # 计算总的步长累积
        self.total_stride = self._calculate_total_stride()
        
        # 预热状态
        self.is_warmed_up = False
        self.warmup_inputs = []
        
    def _analyze_network_structure(self):
        """分析网络结构，为每层创建流式包装器"""
        # 遍历base_network的layers
        for layer in self.base_network.layers:
            if hasattr(layer, 'kernel_size') and hasattr(layer, 'stride'):
                # Conv1d层
                streaming_layer = StreamingTdsLayer(
                    layer, layer.kernel_size, layer.stride
                )
                self.streaming_layers.append(streaming_layer)
            elif hasattr(layer, 'layers'):
                # TdsStage - 查看其内部结构
                for sub_layer in layer.layers:
                    if hasattr(sub_layer, 'kernel_size') and hasattr(sub_layer, 'stride'):
                        streaming_layer = StreamingTdsLayer(
                            sub_layer, sub_layer.kernel_size, sub_layer.stride
                        )
                        self.streaming_layers.append(streaming_layer)
    
    def _calculate_total_stride(self) -> int:
        """计算网络的总步长"""
        total_stride = 1
        for layer in self.streaming_layers:
            total_stride *= layer.stride
        return total_stride
    
    def reset(self):
        """重置所有状态"""
        for layer in self.streaming_layers:
            layer.reset()
        self.is_warmed_up = False
        self.warmup_inputs.clear()
    
    def warmup(self, x: torch.Tensor):
        """预热网络以建立初始状态
        
        Args:
            x: 预热数据 (B, C, T)，通常是left_context长度的数据
        """
        # 直接使用基础网络进行预热
        with torch.no_grad():
            _ = self.base_network(x)
        
        # 为流式层建立初始缓存
        # 这里需要逐层传播来建立正确的缓存状态
        current_input = x
        for layer in self.streaming_layers:
            # 将预热数据分解为单步输入来建立缓存
            time_steps = current_input.shape[2]
            for t in range(0, time_steps, layer.stride):
                end_t = min(t + layer.stride, time_steps)
                step_input = current_input[:, :, t:end_t]
                if step_input.shape[2] == layer.stride:
                    current_input = layer.push_input(step_input)
                    if current_input is None:
                        break
            
            if current_input is None:
                break
        
        self.is_warmed_up = True
    
    def push_step(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """推入一步数据并尝试产生输出
        
        Args:
            x: 输入数据 (B, C, step_size)
            
        Returns:
            编码器输出 (B, C', T') 或 None（如果还不能产生输出）
        """
        if not self.is_warmed_up:
            raise RuntimeError("网络未预热，请先调用warmup()")
        
        if x.shape[2] != self.step_size:
            raise ValueError(f"输入大小不匹配: 期望{self.step_size}, 收到{x.shape[2]}")
        
        # 逐层传播
        current_input = x
        for layer in self.streaming_layers:
            current_input = layer.push_input(current_input)
            if current_input is None:
                return None
        
        return current_input
    
    def get_warmup_length(self) -> int:
        """获取预热所需的输入长度"""
        return self.base_network.left_context


class TdsStreamingWrapper:
    """TDS流式推理的完整包装器
    
    结合StreamingTdsNetwork和增量重采样器，
    提供完整的流式编码器功能。
    """
    
    def __init__(self, base_network: TdsNetwork, config: dict):
        self.base_network = base_network
        self.config = config
        
        # 创建流式网络
        step_size = int(config['constants']['emg_sample_rate'] * 
                       config['runtime']['streaming']['step_ms'] / 1000)  # 20ms@2kHz = 40
        self.streaming_network = StreamingTdsNetwork(base_network, step_size)
        
        # 特征缓冲区（用于收集编码器输出）
        self.feature_buffer = []
        
        # 状态管理
        self.is_initialized = False
    
    def initialize(self, warmup_data: torch.Tensor):
        """初始化流式网络
        
        Args:
            warmup_data: 预热数据 (B, C, left_context)
        """
        self.streaming_network.warmup(warmup_data)
        self.is_initialized = True
    
    def process_step(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """处理一步输入数据
        
        Args:
            x: EMG数据 (B, C, step_size)
            
        Returns:
            编码器特征 (B, feature_dim, T) 或 None
        """
        if not self.is_initialized:
            raise RuntimeError("包装器未初始化，请先调用initialize()")
        
        # 通过流式网络处理
        features = self.streaming_network.push_step(x)
        
        if features is not None:
            # 收集特征用于重采样
            self.feature_buffer.append(features)
            return features
        
        return None
    
    def reset(self):
        """重置所有状态"""
        self.streaming_network.reset()
        self.feature_buffer.clear()
        self.is_initialized = False
    
    def get_accumulated_features(self) -> Optional[torch.Tensor]:
        """获取累积的特征用于重采样
        
        Returns:
            拼接的特征张量 (B, feature_dim, total_T) 或 None
        """
        if not self.feature_buffer:
            return None
        
        return torch.cat(self.feature_buffer, dim=2)
    
    def clear_feature_buffer(self):
        """清空特征缓冲区"""
        self.feature_buffer.clear()


