# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
数据集模块

重构的数据加载组件，移除原有标识，适配实时预测系统。
"""

from __future__ import annotations

from collections.abc import KeysView
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch

from .transforms import Transform
from .transforms import ExtractToTensor


def get_contiguous_ones(mask: np.ndarray) -> list[tuple[int, int]]:
    """获取连续为True的区间"""
    if len(mask) == 0:
        return []
    
    blocks = []
    start = None
    
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            blocks.append((start, i))
            start = None
    
    # 处理以True结尾的情况
    if start is not None:
        blocks.append((start, len(mask)))
    
    return blocks


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """获取IK失败掩码"""
    # 如果关节角度全为0，认为是IK失败
    return ~np.all(joint_angles == 0, axis=1)


@dataclass
class RealtimeSessionData:
    """
    实时会话数据接口
    
    重构的单个会话文件读取接口，移除原有标识。
    """

    HDF5_GROUP: ClassVar[str] = "emg2pose"
    # timeseries keys
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    # metadata keys
    SESSION_NAME: ClassVar[str] = "session"
    SIDE: ClassVar[str] = "side"
    STAGE: ClassVar[str] = "stage"
    START_TIME: ClassVar[str] = "start"
    END_TIME: ClassVar[str] = "end"
    NUM_CHANNELS: ClassVar[str] = "num_channels"
    DATASET_NAME: ClassVar[str] = "dataset"
    USER: ClassVar[str] = "user"
    SAMPLE_RATE: ClassVar[str] = "sample_rate"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        realtime_group: h5py.Group = self._file[self.HDF5_GROUP]

        # timeseries是HDF5复合数据集
        self.timeseries: h5py.Dataset = realtime_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # 将元数据完全加载到内存中
        self.metadata: dict[str, Any] = {}
        for key, val in realtime_group.attrs.items():
            self.metadata[key] = val

    def __enter__(self) -> RealtimeSessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """通过时间戳窗口加载时间序列片段"""
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    @property
    def fields(self) -> KeysView[str]:
        """timeseries中的字段名"""
        fields: KeysView[str] = self.timeseries.dtype.fields.keys()
        return fields

    @property
    def timestamps(self) -> np.ndarray:
        """EMG时间戳"""
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "时间戳不单调"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        """会话唯一名称"""
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        """用户唯一ID"""
        return self.metadata[self.USER]

    @property
    def no_ik_failure(self):
        if not hasattr(self, "_no_ik_failure"):
            joint_angles = self.timeseries[self.JOINT_ANGLES]
            self._no_ik_failure = get_ik_failures_mask(joint_angles)
        return self._no_ik_failure

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"


@dataclass
class WindowedDataset(torch.utils.data.Dataset):
    """
    窗口化数据集
    
    重构的窗口化数据集，适配实时预测系统。
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=ExtractToTensor
    )
    skip_ik_failures: bool = False

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
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
                "skip_ik_failures=True需要指定window_length"
            )

        # 基于skip_ik_failures设置预计算(start, end)窗口
        self.windows: list[tuple[int, int]] = self.precompute_windows()

    def __len__(self) -> int:
        return sum(self._get_block_len(b) for b in self.blocks)

    def _get_block_len(self, block: tuple[float, float]) -> tuple[int]:
        """获取(start, end)时间块中的样本数"""
        return (block[1] - block[0] - self.window_length) // self.stride + 1

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = RealtimeSessionData(self.hdf5_path)
        return self._session

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """要包含在数据集中的(start, end)时间列表"""

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

    def precompute_windows(self) -> list[tuple[int, int]]:
        """为每个数据集索引预计算EMG开始和结束时间"""
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self.blocks])

        for idx in range(len(self)):
            block_idx = np.searchsorted(cumsum, idx, "right") - 1
            start_idx, end_idx = self.blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))

        return windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # 随机抖动窗口偏移
        offset, end_idx = self.windows[idx]
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"索引{idx}越界，剩余{leftover}")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # 扩展窗口以包含上下文填充并获取
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # 提取对应窗口的EMG张量
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # 提取关节角度标签
        joint_angles = window[RealtimeSessionData.JOINT_ANGLES]
        joint_angles = torch.as_tensor(joint_angles)

        # 无IK失败的样本掩码
        no_ik_failure = torch.as_tensor(
            self.session.no_ik_failure[window_start:window_end]
        )
        return {
            "emg": emg.T,  # CT
            "joint_angles": joint_angles.T,  # CT
            "no_ik_failure": no_ik_failure,  # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }
