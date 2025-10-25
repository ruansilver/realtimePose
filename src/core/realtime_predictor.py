# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
实时预测器

实现Fast Path和Correction Path的混合窗口策略，管理模型状态和预测流程。
"""

import time
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
import logging

from ..models.modules import RealtimePoseModel
from ..models.networks import SequentialLSTM
from .buffer_manager import BufferManager

logger = logging.getLogger(__name__)


class RealtimePredictor:
    """
    核心实时预测器
    
    管理模型状态和预测流程，实现Fast Path和Correction Path的混合策略。
    """
    
    def __init__(self, model: RealtimePoseModel, config: Dict[str, Any]):
        """
        初始化实时预测器
        
        Args:
            model: 实时姿态预测模型
            config: 实时预测配置
        """
        self.model = model
        self.config = config
        
        # 配置参数
        self.fast_window_size = config["fast_window_size"]  # 秒
        self.correction_window_size = config["correction_window_size"]  # 秒
        self.max_latency = config["max_latency"] / 1000.0  # 转换为秒
        self.correction_frequency = config["correction_frequency"]
        self.fusion_weight = config.get("fusion_weight", 0.7)  # Fast Path权重
        self.sample_rate = config["sample_rate"]
        self.rollout_freq = config["rollout_freq"]
        
        # 将时间窗口转换为样本数
        self.fast_window_samples = int(self.fast_window_size * self.sample_rate)
        self.correction_window_samples = int(self.correction_window_size * self.sample_rate)
        
        # 预测状态
        self.prediction_count = 0
        self.last_prediction = None
        self.last_fast_prediction = None
        self.last_correction_prediction = None
        
        # LSTM隐藏状态缓存
        self.lstm_hidden_state = None
        self.fast_hidden_state = None
        self.correction_hidden_state = None
        
        # 特征缓存
        self.encoder_feature_cache = None
        self.fast_feature_cache = None
        
        # 性能统计
        self.prediction_times = []
        self.fast_prediction_times = []
        self.correction_prediction_times = []
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info(f"实时预测器初始化完成")
        logger.info(f"Fast窗口: {self.fast_window_size}s ({self.fast_window_samples}样本)")
        logger.info(f"Correction窗口: {self.correction_window_size}s ({self.correction_window_samples}样本)")
    
    def predict_fast(self, emg_chunk: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Fast Path: 使用小窗口进行快速预测
        
        Args:
            emg_chunk: EMG数据块，形状为(1, channels, samples)
            
        Returns:
            prediction: 预测结果，形状为(1, num_joints)  
            latency: 预测延迟（秒）
        """
        start_time = time.time()
        
        try:
            # 确保数据在正确设备上
            emg_chunk = emg_chunk.to(self.device)
            
            # 使用增量预测
            prediction, updated_features, new_hidden_state = self.model.forward_incremental(
                emg_chunk=emg_chunk,
                cached_features=self.fast_feature_cache,
                hidden_state=self.fast_hidden_state,
                step_count=self.prediction_count
            )
            
            # 更新缓存状态
            self.fast_feature_cache = updated_features
            self.fast_hidden_state = new_hidden_state
            
            # 记录预测
            self.last_fast_prediction = prediction
            
            latency = time.time() - start_time
            self.fast_prediction_times.append(latency)
            
            # 保持性能统计列表长度
            if len(self.fast_prediction_times) > 1000:
                self.fast_prediction_times = self.fast_prediction_times[-500:]
            
            logger.debug(f"Fast预测完成，延迟: {latency*1000:.2f}ms")
            
            return prediction, latency
            
        except Exception as e:
            logger.error(f"Fast预测失败: {e}")
            # 返回上次预测或零预测
            if self.last_fast_prediction is not None:
                return self.last_fast_prediction, time.time() - start_time
            else:
                zero_pred = torch.zeros(1, 20, device=self.device)
                return zero_pred, time.time() - start_time
    
    def predict_correction(self, emg_window: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Correction Path: 使用大窗口进行校正预测
        
        Args:
            emg_window: EMG窗口数据，形状为(1, channels, window_samples)
            
        Returns:
            prediction: 校正预测结果，形状为(1, num_joints, time_steps)
            latency: 预测延迟（秒）
        """
        start_time = time.time()
        
        try:
            # 确保数据在正确设备上
            emg_window = emg_window.to(self.device)
            
            # 使用完整窗口预测
            predictions = self.model.forward_correction(emg_window)
            
            # 取最后一个时间步的预测
            if predictions.dim() == 3:
                current_prediction = predictions[:, :, -1]  # (1, num_joints)
            else:
                current_prediction = predictions  # (1, num_joints)
            
            # 记录预测
            self.last_correction_prediction = current_prediction
            
            latency = time.time() - start_time
            self.correction_prediction_times.append(latency)
            
            # 保持性能统计列表长度
            if len(self.correction_prediction_times) > 100:
                self.correction_prediction_times = self.correction_prediction_times[-50:]
            
            logger.debug(f"Correction预测完成，延迟: {latency*1000:.2f}ms")
            
            return current_prediction, latency
            
        except Exception as e:
            logger.error(f"Correction预测失败: {e}")
            # 返回上次预测或零预测
            if self.last_correction_prediction is not None:
                return self.last_correction_prediction, time.time() - start_time
            else:
                zero_pred = torch.zeros(1, 20, device=self.device)
                return zero_pred, time.time() - start_time
    
    def fuse_predictions(
        self, 
        fast_pred: torch.Tensor, 
        correction_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        融合Fast Path和Correction Path的预测结果
        
        Args:
            fast_pred: Fast Path预测，形状为(1, num_joints)
            correction_pred: Correction Path预测，形状为(1, num_joints)
            
        Returns:
            融合后的预测，形状为(1, num_joints)
        """
        # 使用加权平均融合
        fused_pred = (self.fusion_weight * fast_pred + 
                     (1 - self.fusion_weight) * correction_pred)
        
        logger.debug(f"预测融合完成，Fast权重: {self.fusion_weight}")
        
        return fused_pred
    
    def should_run_correction(self) -> bool:
        """判断是否应该运行校正预测"""
        return self.prediction_count % self.correction_frequency == 0
    
    def predict(
        self, 
        buffer_manager: BufferManager
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行完整的预测流程
        
        Args:
            buffer_manager: 缓冲区管理器
            
        Returns:
            prediction: 最终预测结果，形状为(1, num_joints)
            metrics: 预测指标字典
        """
        start_time = time.time()
        
        # 初始化指标
        metrics = {
            "prediction_count": self.prediction_count,
            "fast_latency": 0.0,
            "correction_latency": 0.0,
            "total_latency": 0.0,
            "used_correction": False,
            "buffer_status": buffer_manager.get_status()
        }
        
        try:
            # 获取Fast Path数据
            fast_data = buffer_manager.get_fast_window(self.fast_window_samples)
            if fast_data is None:
                logger.warning("Fast窗口数据不足，使用零预测")
                zero_pred = torch.zeros(1, 20, device=self.device)
                metrics["total_latency"] = time.time() - start_time
                return zero_pred, metrics
            
            # 执行Fast预测
            fast_pred, fast_latency = self.predict_fast(fast_data)
            metrics["fast_latency"] = fast_latency
            
            final_prediction = fast_pred
            
            # 判断是否需要校正
            if self.should_run_correction():
                correction_data = buffer_manager.get_correction_window(self.correction_window_samples)
                
                if correction_data is not None:
                    # 执行校正预测
                    correction_pred, correction_latency = self.predict_correction(correction_data)
                    metrics["correction_latency"] = correction_latency
                    metrics["used_correction"] = True
                    
                    # 融合预测结果
                    final_prediction = self.fuse_predictions(fast_pred, correction_pred)
                    
                    logger.debug(f"使用校正预测，预测计数: {self.prediction_count}")
                else:
                    logger.warning("Correction窗口数据不足，仅使用Fast预测")
            
            # 更新状态
            self.last_prediction = final_prediction
            self.prediction_count += 1
            
            # 检查延迟
            total_latency = time.time() - start_time
            metrics["total_latency"] = total_latency
            
            if total_latency > self.max_latency:
                logger.warning(f"预测延迟超标: {total_latency*1000:.2f}ms > {self.max_latency*1000:.2f}ms")
            
            self.prediction_times.append(total_latency)
            if len(self.prediction_times) > 1000:
                self.prediction_times = self.prediction_times[-500:]
            
            return final_prediction, metrics
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            # 返回上次预测或零预测
            if self.last_prediction is not None:
                prediction = self.last_prediction
            else:
                prediction = torch.zeros(1, 20, device=self.device)
            
            metrics["total_latency"] = time.time() - start_time
            return prediction, metrics
    
    def reset_state(self) -> None:
        """重置预测器状态"""
        self.prediction_count = 0
        self.last_prediction = None
        self.last_fast_prediction = None
        self.last_correction_prediction = None
        
        # 重置LSTM状态
        if isinstance(self.model.decoder, SequentialLSTM):
            self.model.decoder.reset_state()
        
        self.lstm_hidden_state = None
        self.fast_hidden_state = None
        self.correction_hidden_state = None
        
        # 清空特征缓存
        self.encoder_feature_cache = None
        self.fast_feature_cache = None
        
        logger.info("预测器状态已重置")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {
            "prediction_count": self.prediction_count,
            "total_predictions": len(self.prediction_times),
            "fast_predictions": len(self.fast_prediction_times),
            "correction_predictions": len(self.correction_prediction_times)
        }
        
        if self.prediction_times:
            stats.update({
                "avg_total_latency": np.mean(self.prediction_times) * 1000,  # ms
                "max_total_latency": np.max(self.prediction_times) * 1000,   # ms
                "min_total_latency": np.min(self.prediction_times) * 1000,   # ms
                "p95_total_latency": np.percentile(self.prediction_times, 95) * 1000  # ms
            })
        
        if self.fast_prediction_times:
            stats.update({
                "avg_fast_latency": np.mean(self.fast_prediction_times) * 1000,  # ms
                "max_fast_latency": np.max(self.fast_prediction_times) * 1000,   # ms
                "p95_fast_latency": np.percentile(self.fast_prediction_times, 95) * 1000  # ms
            })
        
        if self.correction_prediction_times:
            stats.update({
                "avg_correction_latency": np.mean(self.correction_prediction_times) * 1000,  # ms  
                "max_correction_latency": np.max(self.correction_prediction_times) * 1000,   # ms
                "correction_frequency_actual": len(self.correction_prediction_times) / max(self.prediction_count, 1)
            })
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """动态更新配置参数"""
        old_config = self.config.copy()
        
        # 更新配置
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # 重新计算窗口大小
        if "fast_window_size" in new_config:
            self.fast_window_size = new_config["fast_window_size"]
            self.fast_window_samples = int(self.fast_window_size * self.sample_rate)
        
        if "correction_window_size" in new_config:
            self.correction_window_size = new_config["correction_window_size"]
            self.correction_window_samples = int(self.correction_window_size * self.sample_rate)
        
        if "fusion_weight" in new_config:
            self.fusion_weight = new_config["fusion_weight"]
        
        if "correction_frequency" in new_config:
            self.correction_frequency = new_config["correction_frequency"]
        
        logger.info(f"预测器配置已更新: {new_config}")
        
        # 如果窗口大小改变，重置状态
        if ("fast_window_size" in new_config or "correction_window_size" in new_config):
            self.reset_state()
            logger.info("由于窗口大小改变，预测器状态已重置")
