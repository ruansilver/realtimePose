# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
数据流模拟器

模拟真实sEMG设备的数据流，支持固定时间间隔和随机数据包大小。
"""

import time
import random
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple, Optional, List
import numpy as np
import h5py
import logging

logger = logging.getLogger(__name__)


class DataPacket:
    """数据包类"""
    def __init__(self, emg_data: np.ndarray, joint_angles: np.ndarray, timestamp: float):
        self.emg_data = emg_data  # 形状: (packet_size, 16)
        self.joint_angles = joint_angles  # 形状: (packet_size, 20)
        self.timestamp = timestamp
        self.packet_size = emg_data.shape[0]


class DataSimulator:
    """
    sEMG数据流模拟器
    
    模拟真实EMG设备的数据流特性，按固定时间间隔发送随机大小的数据包。
    """
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]):
        """
        初始化数据模拟器
        
        Args:
            dataset_path: 数据集路径
            config: 模拟配置
        """
        self.dataset_path = Path(dataset_path)
        self.config = config
        
        # 配置参数
        self.packet_interval = config["packet_interval"]  # 固定时间间隔（秒）
        self.min_packet_size = config["packet_size"]["min"]  # 最小包大小
        self.max_packet_size = config["packet_size"]["max"]  # 最大包大小
        self.random_seed = config.get("random_seed", 42)
        
        # 设置随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 数据加载
        self.dataset_files = []
        self.current_file_idx = 0
        self.current_data_idx = 0
        self.current_session_data = None
        
        # 统计信息
        self.total_packets_sent = 0
        self.total_samples_sent = 0
        self.simulation_start_time = None
        
        # 加载数据集
        self._load_dataset()
        
        logger.info(f"数据模拟器初始化完成")
        logger.info(f"数据包间隔: {self.packet_interval}s")
        logger.info(f"数据包大小范围: {self.min_packet_size}-{self.max_packet_size}样本")
        logger.info(f"找到{len(self.dataset_files)}个数据文件")
    
    def _load_dataset(self) -> None:
        """加载数据集文件列表"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        
        # 查找所有HDF5文件
        hdf5_files = list(self.dataset_path.glob("*.hdf5"))
        hdf5_files.extend(list(self.dataset_path.glob("*.h5")))
        
        if not hdf5_files:
            raise FileNotFoundError(f"在{self.dataset_path}中未找到HDF5文件")
        
        self.dataset_files = sorted(hdf5_files)
        logger.info(f"找到数据文件: {[f.name for f in self.dataset_files]}")
    
    def _load_session_data(self, file_path: Path) -> Optional[np.ndarray]:
        """
        加载单个会话数据
        
        Args:
            file_path: HDF5文件路径
            
        Returns:
            时间序列数据或None（如果加载失败）
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # 获取时间序列数据
                timeseries = f['emg2pose/timeseries'][:]
                
                # 过滤掉joint_angles全为0的数据帧
                joint_angles = timeseries['joint_angles']
                valid_mask = ~np.all(joint_angles == 0, axis=1)
                
                if np.sum(valid_mask) == 0:
                    logger.warning(f"文件{file_path.name}中没有有效数据")
                    return None
                
                filtered_data = timeseries[valid_mask]
                logger.info(f"加载{file_path.name}: {len(filtered_data)}个有效样本")
                
                return filtered_data
                
        except Exception as e:
            logger.error(f"加载文件{file_path}失败: {e}")
            return None
    
    def _get_next_session(self) -> bool:
        """
        加载下一个会话数据
        
        Returns:
            是否成功加载
        """
        while self.current_file_idx < len(self.dataset_files):
            file_path = self.dataset_files[self.current_file_idx]
            session_data = self._load_session_data(file_path)
            
            if session_data is not None:
                self.current_session_data = session_data
                self.current_data_idx = 0
                logger.info(f"切换到文件: {file_path.name}")
                return True
            
            self.current_file_idx += 1
        
        # 所有文件都处理完毕，重新开始
        self.current_file_idx = 0
        return self._get_next_session() if self.dataset_files else False
    
    def generate_packet(self) -> Optional[DataPacket]:
        """
        生成随机大小的数据包
        
        Returns:
            数据包或None（如果没有更多数据）
        """
        # 检查当前会话数据
        if self.current_session_data is None:
            if not self._get_next_session():
                return None
        
        # 生成随机包大小
        packet_size = random.randint(self.min_packet_size, self.max_packet_size)
        
        # 检查是否有足够的数据
        remaining_samples = len(self.current_session_data) - self.current_data_idx
        if remaining_samples < packet_size:
            # 当前会话数据不足，尝试加载下一个会话
            self.current_file_idx += 1
            if not self._get_next_session():
                return None
            remaining_samples = len(self.current_session_data) - self.current_data_idx
            packet_size = min(packet_size, remaining_samples)
        
        # 提取数据包
        start_idx = self.current_data_idx
        end_idx = start_idx + packet_size
        
        packet_data = self.current_session_data[start_idx:end_idx]
        
        # 提取EMG和关节角度数据
        emg_data = np.array([sample['emg'] for sample in packet_data])
        joint_angles = np.array([sample['joint_angles'] for sample in packet_data])
        
        # 创建数据包
        packet = DataPacket(
            emg_data=emg_data,
            joint_angles=joint_angles, 
            timestamp=time.time()
        )
        
        # 更新索引
        self.current_data_idx = end_idx
        
        # 更新统计
        self.total_packets_sent += 1
        self.total_samples_sent += packet_size
        
        logger.debug(f"生成数据包: 大小={packet_size}, 总包数={self.total_packets_sent}")
        
        return packet
    
    def simulate_realtime_stream(self) -> Iterator[DataPacket]:
        """
        模拟实时数据流
        
        按固定时间间隔发送随机大小的数据包。
        
        Yields:
            DataPacket: 数据包
        """
        logger.info("开始模拟实时数据流")
        self.simulation_start_time = time.time()
        
        next_send_time = time.time()
        
        try:
            while True:
                # 等待到下一个发送时间
                current_time = time.time()
                if current_time < next_send_time:
                    sleep_time = next_send_time - current_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # 生成并发送数据包
                packet = self.generate_packet()
                if packet is None:
                    logger.warning("没有更多数据，重新开始循环")
                    # 重置到第一个文件
                    self.current_file_idx = 0
                    self.current_session_data = None
                    self.current_data_idx = 0
                    continue
                
                yield packet
                
                # 计算下一个发送时间
                next_send_time += self.packet_interval
                
                # 每1000个包记录一次统计
                if self.total_packets_sent % 1000 == 0:
                    self._log_statistics()
                    
        except KeyboardInterrupt:
            logger.info("数据流模拟被用户中断")
        except Exception as e:
            logger.error(f"数据流模拟出错: {e}")
            raise
    
    def simulate_batch_stream(self, batch_size: int = 32) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        模拟批次数据流，用于训练
        
        Args:
            batch_size: 批次大小
            
        Yields:
            (emg_batch, joint_angles_batch): EMG数据和关节角度批次
        """
        logger.info(f"开始模拟批次数据流，批次大小: {batch_size}")
        
        emg_batch = []
        joint_angles_batch = []
        
        try:
            for packet in self.simulate_realtime_stream():
                emg_batch.extend(packet.emg_data)
                joint_angles_batch.extend(packet.joint_angles)
                
                # 当累积足够样本时返回批次
                if len(emg_batch) >= batch_size:
                    # 截取精确的批次大小
                    batch_emg = np.array(emg_batch[:batch_size])
                    batch_joints = np.array(joint_angles_batch[:batch_size])
                    
                    # 保留剩余样本
                    emg_batch = emg_batch[batch_size:]
                    joint_angles_batch = joint_angles_batch[batch_size:]
                    
                    yield batch_emg, batch_joints
                    
        except Exception as e:
            logger.error(f"批次数据流模拟出错: {e}")
            raise
    
    def _log_statistics(self) -> None:
        """记录统计信息"""
        if self.simulation_start_time is None:
            return
        
        elapsed_time = time.time() - self.simulation_start_time
        packets_per_sec = self.total_packets_sent / elapsed_time if elapsed_time > 0 else 0
        samples_per_sec = self.total_samples_sent / elapsed_time if elapsed_time > 0 else 0
        avg_packet_size = self.total_samples_sent / self.total_packets_sent if self.total_packets_sent > 0 else 0
        
        logger.info(
            f"数据流统计 - "
            f"总包数: {self.total_packets_sent}, "
            f"总样本: {self.total_samples_sent}, "
            f"包/秒: {packets_per_sec:.2f}, "
            f"样本/秒: {samples_per_sec:.2f}, "
            f"平均包大小: {avg_packet_size:.1f}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取模拟器统计信息"""
        elapsed_time = 0
        if self.simulation_start_time is not None:
            elapsed_time = time.time() - self.simulation_start_time
        
        return {
            "total_packets_sent": self.total_packets_sent,
            "total_samples_sent": self.total_samples_sent,
            "elapsed_time": elapsed_time,
            "packets_per_second": self.total_packets_sent / elapsed_time if elapsed_time > 0 else 0,
            "samples_per_second": self.total_samples_sent / elapsed_time if elapsed_time > 0 else 0,
            "average_packet_size": self.total_samples_sent / self.total_packets_sent if self.total_packets_sent > 0 else 0,
            "current_file": self.dataset_files[self.current_file_idx].name if self.current_file_idx < len(self.dataset_files) else "N/A",
            "current_file_progress": f"{self.current_data_idx}/{len(self.current_session_data) if self.current_session_data is not None else 0}",
            "config": {
                "packet_interval": self.packet_interval,
                "min_packet_size": self.min_packet_size,
                "max_packet_size": self.max_packet_size,
                "random_seed": self.random_seed
            }
        }
    
    def reset(self) -> None:
        """重置模拟器状态"""
        self.current_file_idx = 0
        self.current_data_idx = 0
        self.current_session_data = None
        self.total_packets_sent = 0
        self.total_samples_sent = 0
        self.simulation_start_time = None
        
        # 重新设置随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info("数据模拟器状态已重置")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """动态更新配置"""
        old_config = self.config.copy()
        
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # 更新相关属性
        if "packet_interval" in new_config:
            self.packet_interval = new_config["packet_interval"]
        
        if "packet_size" in new_config:
            self.min_packet_size = new_config["packet_size"]["min"]
            self.max_packet_size = new_config["packet_size"]["max"]
        
        if "random_seed" in new_config:
            self.random_seed = new_config["random_seed"]
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        logger.info(f"数据模拟器配置已更新: {new_config}")
