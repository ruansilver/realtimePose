# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
数据变换模块

数据预处理和变换功能，重构自原有组件。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar, Union, List
import numpy as np
import torch

# 类型变量
T = TypeVar('T')
U = TypeVar('U')


class Transform(ABC, Generic[T, U]):
    """抽象变换基类"""
    
    @abstractmethod
    def __call__(self, data: T) -> U:
        pass


class ExtractToTensor(Transform[np.ndarray, torch.Tensor]):
    """
    从结构化数组中提取EMG数据并转换为PyTorch张量
    
    这是默认的变换，提取EMG信号并将其转换为浮点张量。
    """
    
    def __init__(self, emg_key: str = "emg", dtype: torch.dtype = torch.float32):
        """
        初始化提取器
        
        Args:
            emg_key: EMG数据在结构化数组中的键名
            dtype: 输出张量的数据类型
        """
        self.emg_key = emg_key
        self.dtype = dtype
    
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """
        提取EMG数据并转换为张量
        
        Args:
            data: 结构化numpy数组
            
        Returns:
            EMG张量，形状为(time_steps, channels)
        """
        # 提取EMG数据
        emg_data = data[self.emg_key]
        
        # 确保是numpy数组
        if not isinstance(emg_data, np.ndarray):
            emg_data = np.array(emg_data)
        
        # 转换为PyTorch张量
        tensor = torch.from_numpy(emg_data.astype(np.float32))
        
        return tensor.to(self.dtype)


class Normalize(Transform[torch.Tensor, torch.Tensor]):
    """标准化变换"""
    
    def __init__(self, mean: Union[float, torch.Tensor] = 0.0, std: Union[float, torch.Tensor] = 1.0):
        """
        初始化标准化变换
        
        Args:
            mean: 均值，可以是标量或张量
            std: 标准差，可以是标量或张量
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用标准化"""
        return (data - self.mean) / self.std


class MinMaxScale(Transform[torch.Tensor, torch.Tensor]):
    """最小-最大缩放变换"""
    
    def __init__(self, min_val: Union[float, torch.Tensor] = 0.0, max_val: Union[float, torch.Tensor] = 1.0):
        """
        初始化最小-最大缩放
        
        Args:
            min_val: 最小值
            max_val: 最大值
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用最小-最大缩放"""
        data_min = data.min()
        data_max = data.max()
        
        # 避免除零
        data_range = data_max - data_min
        if data_range == 0:
            return torch.full_like(data, (self.min_val + self.max_val) / 2)
        
        # 缩放到[0, 1]然后映射到[min_val, max_val]
        normalized = (data - data_min) / data_range
        scaled = normalized * (self.max_val - self.min_val) + self.min_val
        
        return scaled


class Clip(Transform[torch.Tensor, torch.Tensor]):
    """裁剪变换"""
    
    def __init__(self, min_val: float = -float('inf'), max_val: float = float('inf')):
        """
        初始化裁剪变换
        
        Args:
            min_val: 最小值
            max_val: 最大值
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用裁剪"""
        return torch.clamp(data, self.min_val, self.max_val)


class AddNoise(Transform[torch.Tensor, torch.Tensor]):
    """添加噪声变换"""
    
    def __init__(self, noise_std: float = 0.01, noise_type: str = "gaussian"):
        """
        初始化噪声变换
        
        Args:
            noise_std: 噪声标准差
            noise_type: 噪声类型 ("gaussian" 或 "uniform")
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """添加噪声"""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(data) * self.noise_std
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(data) - 0.5) * 2 * self.noise_std
        else:
            raise ValueError(f"不支持的噪声类型: {self.noise_type}")
        
        return data + noise


class FilterFrequency(Transform[torch.Tensor, torch.Tensor]):
    """频率滤波变换（简单的高通/低通滤波）"""
    
    def __init__(self, filter_type: str = "none", cutoff: float = 0.1):
        """
        初始化频率滤波
        
        Args:
            filter_type: 滤波类型 ("highpass", "lowpass", "none")
            cutoff: 截止频率（归一化频率，0-1之间）
        """
        self.filter_type = filter_type
        self.cutoff = cutoff
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用频率滤波"""
        if self.filter_type == "none":
            return data
        
        # 简单的一阶差分近似高通滤波
        if self.filter_type == "highpass":
            if data.shape[0] > 1:
                diff = torch.diff(data, dim=0)
                # 保持原始长度
                filtered = torch.cat([data[:1], diff], dim=0)
                return filtered * (1 - self.cutoff) + data * self.cutoff
        
        # 简单的移动平均近似低通滤波
        elif self.filter_type == "lowpass":
            kernel_size = max(3, int(1 / self.cutoff))
            if data.shape[0] >= kernel_size:
                # 一维卷积实现移动平均
                kernel = torch.ones(kernel_size) / kernel_size
                # 为每个通道应用滤波
                filtered = torch.zeros_like(data)
                for ch in range(data.shape[1]):
                    channel_data = data[:, ch]
                    # 填充以保持长度
                    padded = torch.nn.functional.pad(channel_data, (kernel_size//2, kernel_size//2), mode='reflect')
                    convolved = torch.nn.functional.conv1d(padded.unsqueeze(0).unsqueeze(0), 
                                                         kernel.unsqueeze(0).unsqueeze(0), 
                                                         padding=0)
                    filtered[:, ch] = convolved.squeeze()[:data.shape[0]]
                return filtered
        
        return data


class Resample(Transform[torch.Tensor, torch.Tensor]):
    """重采样变换"""
    
    def __init__(self, target_length: int, mode: str = "linear"):
        """
        初始化重采样变换
        
        Args:
            target_length: 目标长度
            mode: 插值模式 ("linear", "nearest")
        """
        self.target_length = target_length
        self.mode = mode
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """应用重采样"""
        if data.shape[0] == self.target_length:
            return data
        
        # 转换为(1, channels, time)格式用于插值
        data_for_interp = data.T.unsqueeze(0)  # (1, channels, time)
        
        # 使用torch的插值函数
        resampled = torch.nn.functional.interpolate(
            data_for_interp, 
            size=self.target_length, 
            mode=self.mode,
            align_corners=False if self.mode == "linear" else None
        )
        
        # 转换回(time, channels)格式
        return resampled.squeeze(0).T


class Compose(Transform[T, U]):
    """组合多个变换"""
    
    def __init__(self, transforms: List[Transform]):
        """
        初始化组合变换
        
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, data: T) -> U:
        """依次应用所有变换"""
        result = data
        for transform in self.transforms:
            result = transform(result)
        return result
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class Lambda(Transform[T, U]):
    """使用lambda函数的自定义变换"""
    
    def __init__(self, func: Callable[[T], U]):
        """
        初始化Lambda变换
        
        Args:
            func: 要应用的函数
        """
        self.func = func
    
    def __call__(self, data: T) -> U:
        """应用lambda函数"""
        return self.func(data)


# 常用变换组合
def get_standard_transform() -> Compose:
    """获取标准的EMG数据变换"""
    return Compose([
        ExtractToTensor(),
        Clip(min_val=-1000, max_val=1000),  # 裁剪异常值
        # 可以根据需要添加其他变换
    ])


def get_augmented_transform(noise_std: float = 0.01) -> Compose:
    """获取带数据增强的变换"""
    return Compose([
        ExtractToTensor(),
        Clip(min_val=-1000, max_val=1000),
        AddNoise(noise_std=noise_std),
        # 可以添加其他增强
    ])


def get_filtered_transform(filter_type: str = "highpass", cutoff: float = 0.1) -> Compose:
    """获取带滤波的变换"""
    return Compose([
        ExtractToTensor(),
        FilterFrequency(filter_type=filter_type, cutoff=cutoff),
        Clip(min_val=-1000, max_val=1000),
    ])
