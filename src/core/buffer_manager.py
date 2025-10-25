# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
缓冲区管理系统

实现循环缓冲区和双缓冲区管理，用于实时数据流处理。
"""

import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class CircularBuffer:
    """循环缓冲区基类
    
    实现高效的循环缓冲区，支持多维数据存储和并发访问。
    """
    
    def __init__(self, capacity: int, data_shape: Tuple[int, ...], dtype=np.float32):
        """
        初始化循环缓冲区
        
        Args:
            capacity: 缓冲区容量（样本数）
            data_shape: 每个样本的形状，例如(16,)表示16通道EMG
            dtype: 数据类型
        """
        self.capacity = capacity
        self.data_shape = data_shape
        self.dtype = dtype
        
        # 创建缓冲区存储
        self.buffer = np.zeros((capacity,) + data_shape, dtype=dtype)
        
        # 缓冲区状态
        self.head = 0  # 写入位置
        self.size = 0  # 当前存储的样本数
        self.full = False  # 是否已满
        
        # 线程锁，确保并发安全
        self._lock = threading.RLock()
    
    def put(self, data: np.ndarray) -> None:
        """
        添加数据到缓冲区
        
        Args:
            data: 要添加的数据，形状为(batch_size,) + data_shape
        """
        with self._lock:
            batch_size = data.shape[0]
            
            for i in range(batch_size):
                self.buffer[self.head] = data[i]
                self.head = (self.head + 1) % self.capacity
                
                if self.full:
                    # 缓冲区已满，覆盖旧数据
                    pass
                else:
                    self.size += 1
                    if self.size == self.capacity:
                        self.full = True
    
    def get_recent(self, n_samples: int) -> Optional[np.ndarray]:
        """
        获取最近的n个样本
        
        Args:
            n_samples: 要获取的样本数
            
        Returns:
            形状为(n_samples,) + data_shape的数组，如果数据不足则返回None
        """
        with self._lock:
            if self.size < n_samples:
                return None
            
            # 计算起始位置
            if self.full:
                start_idx = (self.head - n_samples) % self.capacity
            else:
                start_idx = max(0, self.size - n_samples)
            
            # 提取数据
            if self.full and start_idx + n_samples > self.capacity:
                # 数据跨越缓冲区边界
                end_part_size = self.capacity - start_idx
                start_part_size = n_samples - end_part_size
                
                result = np.zeros((n_samples,) + self.data_shape, dtype=self.dtype)
                result[:end_part_size] = self.buffer[start_idx:]
                result[end_part_size:] = self.buffer[:start_part_size]
                return result
            else:
                # 连续数据
                end_idx = start_idx + n_samples
                return self.buffer[start_idx:end_idx].copy()
    
    def get_all(self) -> np.ndarray:
        """获取缓冲区中的所有数据"""
        with self._lock:
            if self.size == 0:
                return np.zeros((0,) + self.data_shape, dtype=self.dtype)
            
            if self.full:
                # 缓冲区已满，按正确顺序返回数据
                result = np.zeros((self.capacity,) + self.data_shape, dtype=self.dtype)
                result[:self.capacity - self.head] = self.buffer[self.head:]
                result[self.capacity - self.head:] = self.buffer[:self.head]
                return result
            else:
                # 缓冲区未满
                return self.buffer[:self.size].copy()
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self.head = 0
            self.size = 0
            self.full = False
            self.buffer.fill(0)
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        with self._lock:
            return self.size == 0
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        with self._lock:
            return self.full
    
    def get_size(self) -> int:
        """获取当前存储的样本数"""
        with self._lock:
            return self.size
    
    def get_utilization(self) -> float:
        """获取缓冲区利用率"""
        with self._lock:
            return self.size / self.capacity


class RawDataBuffer(CircularBuffer):
    """原始sEMG数据缓冲区
    
    专门用于存储原始的sEMG信号数据。
    """
    
    def __init__(self, capacity: int, num_channels: int = 16):
        """
        初始化原始数据缓冲区
        
        Args:
            capacity: 缓冲区容量（样本数）
            num_channels: EMG通道数，默认16
        """
        super().__init__(capacity, (num_channels,), dtype=np.float32)
        self.num_channels = num_channels
        logger.info(f"初始化原始数据缓冲区: 容量={capacity}, 通道数={num_channels}")
    
    def add_emg_packet(self, emg_data: np.ndarray) -> None:
        """
        添加EMG数据包
        
        Args:
            emg_data: EMG数据，形状为(batch_size, num_channels)
        """
        if emg_data.shape[1] != self.num_channels:
            raise ValueError(f"EMG数据通道数不匹配: 期望{self.num_channels}, 实际{emg_data.shape[1]}")
        
        self.put(emg_data)
    
    def get_window_data(self, window_size: int) -> Optional[torch.Tensor]:
        """
        获取窗口数据，转换为PyTorch张量
        
        Args:
            window_size: 窗口大小（样本数）
            
        Returns:
            形状为(1, num_channels, window_size)的张量，如果数据不足则返回None
        """
        data = self.get_recent(window_size)
        if data is None:
            return None
        
        # 转换为PyTorch张量并调整维度: (window_size, num_channels) -> (1, num_channels, window_size)
        tensor = torch.from_numpy(data.T[None, :, :])
        return tensor


class FeatureBuffer(CircularBuffer):
    """编码器特征缓冲区
    
    专门用于存储编码器输出的特征向量。
    """
    
    def __init__(self, capacity: int, feature_dim: int):
        """
        初始化特征缓冲区
        
        Args:
            capacity: 缓冲区容量（特征向量数）
            feature_dim: 特征维度
        """
        super().__init__(capacity, (feature_dim,), dtype=np.float32)
        self.feature_dim = feature_dim
        logger.info(f"初始化特征缓冲区: 容量={capacity}, 特征维度={feature_dim}")
    
    def add_features(self, features: torch.Tensor) -> None:
        """
        添加特征向量
        
        Args:
            features: 特征张量，形状为(batch_size, feature_dim, time_steps)或(batch_size, feature_dim)
        """
        if features.dim() == 3:
            # (batch_size, feature_dim, time_steps) -> (time_steps, feature_dim)
            features = features.squeeze(0).T
        elif features.dim() == 2:
            # (batch_size, feature_dim) -> (1, feature_dim)
            features = features
        else:
            raise ValueError(f"不支持的特征张量维度: {features.dim()}")
        
        # 转换为numpy并添加到缓冲区
        features_np = features.detach().cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np[None, :]
        
        self.put(features_np)
    
    def get_recent_features(self, n_features: int) -> Optional[torch.Tensor]:
        """
        获取最近的特征，转换为PyTorch张量
        
        Args:
            n_features: 要获取的特征数量
            
        Returns:
            形状为(1, feature_dim, n_features)的张量，如果数据不足则返回None
        """
        data = self.get_recent(n_features)
        if data is None:
            return None
        
        # 转换为PyTorch张量并调整维度: (n_features, feature_dim) -> (1, feature_dim, n_features)
        tensor = torch.from_numpy(data.T[None, :, :])
        return tensor


class BufferManager:
    """双缓冲区管理器
    
    统一管理原始数据缓冲区和特征缓冲区，提供高级接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化缓冲区管理器
        
        Args:
            config: 缓冲区配置
        """
        self.config = config
        
        # 创建原始数据缓冲区
        raw_config = config["raw_data"]
        self.raw_buffer = RawDataBuffer(
            capacity=raw_config["size"],
            num_channels=16  # EMG通道数
        )
        
        # 创建特征缓冲区（特征维度需要根据模型确定）
        feature_config = config["feature"]
        self.feature_buffer = FeatureBuffer(
            capacity=feature_config["size"],
            feature_dim=64  # 根据模型配置调整
        )
        
        logger.info("缓冲区管理器初始化完成")
    
    def add_raw_data(self, emg_data: np.ndarray) -> None:
        """添加原始EMG数据"""
        self.raw_buffer.add_emg_packet(emg_data)
    
    def add_features(self, features: torch.Tensor) -> None:
        """添加特征数据"""
        self.feature_buffer.add_features(features)
    
    def get_fast_window(self, window_size: int) -> Optional[torch.Tensor]:
        """获取快速预测窗口数据"""
        return self.raw_buffer.get_window_data(window_size)
    
    def get_correction_window(self, window_size: int) -> Optional[torch.Tensor]:
        """获取校正预测窗口数据"""
        return self.raw_buffer.get_window_data(window_size)
    
    def get_recent_features(self, n_features: int) -> Optional[torch.Tensor]:
        """获取最近的特征"""
        return self.feature_buffer.get_recent_features(n_features)
    
    def get_training_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取训练批次数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (input_data, target_data)元组，如果数据不足则返回None
        """
        # 获取输入数据
        input_data = self.raw_buffer.get_window_data(batch_size)
        if input_data is None:
            return None
        
        # 这里应该从真实标签中获取目标数据，暂时返回零张量
        target_data = torch.zeros(1, 20, batch_size)  # 20个关节角度
        
        return input_data, target_data
    
    def clear_all(self) -> None:
        """清空所有缓冲区"""
        self.raw_buffer.clear()
        self.feature_buffer.clear()
        logger.info("所有缓冲区已清空")
    
    def get_status(self) -> Dict[str, Any]:
        """获取缓冲区状态信息"""
        return {
            "raw_buffer": {
                "size": self.raw_buffer.get_size(),
                "capacity": self.raw_buffer.capacity,
                "utilization": self.raw_buffer.get_utilization(),
                "is_full": self.raw_buffer.is_full()
            },
            "feature_buffer": {
                "size": self.feature_buffer.get_size(),
                "capacity": self.feature_buffer.capacity,
                "utilization": self.feature_buffer.get_utilization(),
                "is_full": self.feature_buffer.is_full()
            }
        }
