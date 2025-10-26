"""数据加载模块：HDF5读取、窗口化切片、IK失败过滤"""

import platform
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """计算IK失败掩码（全零帧表示IK失败）
    
    Args:
        joint_angles: 关节角度数组 (..., num_joints)
        
    Returns:
        布尔掩码，True表示无IK失败
    """
    zeros = np.zeros_like(joint_angles)
    is_zero = np.isclose(joint_angles, zeros)
    return ~np.all(is_zero, axis=-1)


def get_contiguous_ones(binary_vector: np.ndarray) -> list[tuple[int, int]]:
    """获取连续True值的(start_idx, end_idx)列表"""
    if (binary_vector == 0).all():
        return []
    
    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [
        (ones[i], ones[j])
        for i, j in zip(
            np.insert(boundaries + 1, 0, 0),
            np.append(boundaries, len(ones) - 1)
        )
    ]


@dataclass
class SessionData:
    """HDF5会话数据读取接口"""
    
    HDF5_GROUP: ClassVar[str] = "emg2pose"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    
    hdf5_path: Path
    
    def __post_init__(self):
        self._ensure_file_open()
    
    def _ensure_file_open(self):
        """确保HDF5文件已打开（支持多进程重新打开）"""
        if not hasattr(self, '_file') or not self._file.id.valid:
            self._file = h5py.File(self.hdf5_path, "r")
            emg2pose_group = self._file[self.HDF5_GROUP]
            
            # timeseries 是HDF5复合数据集
            self.timeseries: h5py.Dataset = emg2pose_group[self.TIMESERIES]
            assert self.timeseries.dtype.fields is not None
            assert self.EMG in self.timeseries.dtype.fields
            assert self.JOINT_ANGLES in self.timeseries.dtype.fields
            assert self.TIMESTAMPS in self.timeseries.dtype.fields
            
            # 加载元数据
            self.metadata: dict[str, Any] = {}
            for key, val in emg2pose_group.attrs.items():
                self.metadata[key] = val
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
    
    def __len__(self) -> int:
        return len(self.timeseries)
    
    def __getitem__(self, key: slice) -> np.ndarray:
        self._ensure_file_open()
        return self.timeseries[key]
    
    @property
    def no_ik_failure(self):
        """缓存的IK失败掩码"""
        if not hasattr(self, "_no_ik_failure"):
            self._ensure_file_open()
            joint_angles = self.timeseries[self.JOINT_ANGLES]
            self._no_ik_failure = get_ik_failures_mask(joint_angles)
        return self._no_ik_failure
    
    @property
    def timestamps(self) -> np.ndarray:
        """EMG时间戳"""
        self._ensure_file_open()
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "时间戳非单调递增"
        return emg_timestamps


@dataclass
class WindowedDataset(Dataset):
    """窗口化EMG数据集
    
    Args:
        hdf5_path: HDF5文件路径
        window_length: 窗口长度（样本数）
        stride: 滑动步长，None则等于window_length
        padding: 左右填充 (left_padding, right_padding)
        jitter: 是否随机抖动窗口偏移（训练时使用）
        skip_ik_failures: 是否跳过包含IK失败的窗口
    """
    
    hdf5_path: Path
    window_length: InitVar[int | None] = 10000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    skip_ik_failures: bool = False
    
    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ):
        self.session_length = len(self.session)
        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0
        
        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0
        
        if window_length is None and self.skip_ik_failures:
            raise ValueError(
                "skip_ik_failures=True 需要指定 window_length"
            )
        
        # 预计算窗口起止位置
        self.windows: list[tuple[int, int]] = self.precompute_windows()
    
    def __len__(self) -> int:
        return len(self.windows)
    
    @property
    def session(self):
        """延迟加载会话数据（每次访问时重新打开以支持多进程）"""
        if not hasattr(self, "_session"):
            self._session = SessionData(self.hdf5_path)
        else:
            # 确保文件仍然有效，如果无效则重新打开
            self._session._ensure_file_open()
        return self._session
    
    @property
    def blocks(self) -> list[tuple[int, int]]:
        """有效数据块的(start, end)列表"""
        if not hasattr(self, "_blocks"):
            # 包含所有时间
            if not self.skip_ik_failures:
                self._blocks = [(0, len(self.session))]
            # 仅包含无IK失败的时间
            else:
                blocks = get_contiguous_ones(self.session.no_ik_failure)
                blocks = [
                    (t0, t1 - 1)
                    for (t0, t1) in blocks
                    if (t1 - t0) >= self.window_length
                ]
                self._blocks = blocks
        
        return self._blocks
    
    def _get_block_len(self, block: tuple[int, int]) -> int:
        """获取块中的样本数"""
        return (block[1] - block[0] - self.window_length) // self.stride + 1
    
    def precompute_windows(self) -> list[tuple[int, int]]:
        """为每个数据集索引预计算窗口的起止位置"""
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self.blocks])
        
        for idx in range(sum(self._get_block_len(b) for b in self.blocks)):
            block_idx = np.searchsorted(cumsum, idx, "right") - 1
            start_idx, end_idx = self.blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))
        
        return windows
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取窗口数据
        
        Returns:
            字典包含:
                - emg: (C, T) EMG信号
                - joint_angles: (C, T) 关节角度
                - no_ik_failure: (T,) IK失败掩码
                - window_start_idx: 窗口起始索引
                - window_end_idx: 窗口结束索引
        """
        # 随机抖动窗口偏移
        offset, end_idx = self.windows[idx]
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"索引 {idx} 越界，leftover {leftover}")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))
        
        # 扩展窗口以包含上下文填充
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]
        
        # 提取EMG张量
        emg = torch.as_tensor(window[SessionData.EMG], dtype=torch.float32)
        
        # 提取关节角度标签
        joint_angles = torch.as_tensor(
            window[SessionData.JOINT_ANGLES], dtype=torch.float32
        )
        
        # IK失败掩码
        no_ik_failure = torch.as_tensor(
            self.session.no_ik_failure[window_start:window_end]
        )
        
        return {
            "emg": emg.T,  # (T, C) -> (C, T)
            "joint_angles": joint_angles.T,  # (T, C) -> (C, T)
            "no_ik_failure": no_ik_failure,  # (T,)
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }


def create_dataloader(
    hdf5_files: list[Path],
    batch_size: int,
    num_workers: int,
    window_length: int,
    padding: tuple[int, int],
    jitter: bool = False,
    skip_ik_failures: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """创建DataLoader
    
    Args:
        hdf5_files: HDF5文件路径列表
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        window_length: 窗口长度
        padding: 填充 (left, right)
        jitter: 是否抖动
        skip_ik_failures: 是否跳过IK失败
        shuffle: 是否打乱
        
    Returns:
        DataLoader实例
    """
    datasets = [
        WindowedDataset(
            hdf5_path=hdf5_file,
            window_length=window_length,
            padding=padding,
            jitter=jitter,
            skip_ik_failures=skip_ik_failures,
        )
        for hdf5_file in hdf5_files
    ]
    
    if len(datasets) == 0:
        raise ValueError("未找到有效的HDF5文件")
    
    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    # Windows平台h5py对象无法pickle，需要使用单进程
    # Linux/macOS可以正常使用多进程
    actual_num_workers = 0 if platform.system() == "Windows" else num_workers
    
    if actual_num_workers != num_workers:
        print(f"检测到Windows平台，将num_workers从{num_workers}调整为0以避免h5py pickle问题")
    
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )

