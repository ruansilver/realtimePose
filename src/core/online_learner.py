# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
在线学习器

实现复合触发策略和分层学习率的在线学习功能，支持模型的动态适应。
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import logging

from ..models.modules import RealtimePoseModel
from .data_simulator import DataPacket

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    在线学习器
    
    负责模型的动态适应，使用复合触发策略和分层学习率进行在线微调。
    """
    
    def __init__(self, model: RealtimePoseModel, config: Dict[str, Any]):
        """
        初始化在线学习器
        
        Args:
            model: 实时姿态预测模型
            config: 在线学习配置
        """
        self.model = model
        self.config = config
        
        # 配置参数
        self.enabled = config["enabled"]
        self.encoder_lr = config["learning_rates"]["encoder"]
        self.decoder_lr = config["learning_rates"]["decoder"]
        self.batch_size = config["batch_size"]
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # 触发条件配置
        trigger_config = config["trigger_conditions"]
        self.error_threshold = trigger_config["error_threshold"]
        self.data_count_threshold = trigger_config["data_count"]
        self.time_interval_threshold = trigger_config["time_interval"]
        self.trigger_logic = trigger_config.get("trigger_logic", "any")  # "any" 或 "all"
        
        # 学习状态
        self.last_update_time = time.time()
        self.new_data_count = 0
        self.error_history = deque(maxlen=100)  # 保留最近100个误差值
        self.recent_errors = deque(maxlen=20)   # 最近20个误差，用于触发判断
        
        # 优化器
        self.optimizers = None
        self.schedulers = None
        
        # 训练数据缓存
        self.training_buffer = deque(maxlen=1000)  # 训练样本缓存
        
        # 统计信息
        self.total_updates = 0
        self.trigger_counts = {
            "error": 0,
            "data_count": 0, 
            "time_interval": 0,
            "total": 0
        }
        self.update_times = []
        self.losses = []
        
        # 设备
        self.device = next(model.parameters()).device
        
        # 初始化优化器
        if self.enabled:
            self.setup_optimizers()
        
        logger.info(f"在线学习器初始化完成，状态: {'启用' if self.enabled else '禁用'}")
        if self.enabled:
            logger.info(f"编码器学习率: {self.encoder_lr}, 解码器学习率: {self.decoder_lr}")
            logger.info(f"触发策略: {self.trigger_logic}")
    
    def setup_optimizers(self) -> None:
        """设置分层学习率优化器"""
        if not self.enabled:
            return
        
        # 分别为编码器和解码器设置不同的学习率
        encoder_params = list(self.model.network.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        # 创建参数组
        param_groups = [
            {"params": encoder_params, "lr": self.encoder_lr, "name": "encoder"},
            {"params": decoder_params, "lr": self.decoder_lr, "name": "decoder"}
        ]
        
        # 使用Adam优化器
        self.optimizers = {
            "main": optim.Adam(param_groups, weight_decay=1e-5)
        }
        
        # 设置学习率调度器
        self.schedulers = {
            "main": optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers["main"], 
                mode='min', 
                factor=0.8, 
                patience=10
                # 移除verbose参数以兼容新版PyTorch
            )
        }
        
        logger.info("优化器和调度器设置完成")
    
    def add_training_sample(self, emg_data: torch.Tensor, target_angles: torch.Tensor) -> None:
        """
        添加训练样本到缓存
        
        Args:
            emg_data: EMG数据，形状为(batch_size, channels, time_steps)
            target_angles: 目标关节角度，形状为(batch_size, num_joints, time_steps)
        """
        if not self.enabled:
            return
        
        # 确保数据在CPU上以节省GPU内存
        emg_cpu = emg_data.detach().cpu()
        target_cpu = target_angles.detach().cpu()
        
        # 添加到训练缓存
        sample = (emg_cpu, target_cpu)
        self.training_buffer.append(sample)
        
        # 更新新数据计数
        self.new_data_count += emg_data.shape[0]  # batch_size
        
        logger.debug(f"添加训练样本，缓存大小: {len(self.training_buffer)}")
    
    def add_prediction_error(self, error: float) -> None:
        """
        添加预测误差
        
        Args:
            error: 预测误差值（如MSE）
        """
        if not self.enabled:
            return
        
        self.error_history.append(error)
        self.recent_errors.append(error)
        
        logger.debug(f"添加预测误差: {error:.6f}")
    
    def should_trigger_learning(self) -> Tuple[bool, List[str]]:
        """
        判断是否应该触发在线学习
        
        Returns:
            (should_trigger, reasons): 是否触发和触发原因列表
        """
        if not self.enabled:
            return False, []
        
        current_time = time.time()
        reasons = []
        
        # 检查误差条件
        error_trigger = False
        if len(self.recent_errors) >= 5:  # 至少需要5个误差值
            recent_avg_error = np.mean(list(self.recent_errors)[-5:])
            if recent_avg_error > self.error_threshold:
                error_trigger = True
                reasons.append(f"误差超阈值 ({recent_avg_error:.4f} > {self.error_threshold})")
        
        # 检查数据量条件
        data_count_trigger = self.new_data_count >= self.data_count_threshold
        if data_count_trigger:
            reasons.append(f"新数据量达到阈值 ({self.new_data_count} >= {self.data_count_threshold})")
        
        # 检查时间间隔条件
        time_elapsed = current_time - self.last_update_time
        time_trigger = time_elapsed >= self.time_interval_threshold
        if time_trigger:
            reasons.append(f"时间间隔达到阈值 ({time_elapsed:.1f}s >= {self.time_interval_threshold}s)")
        
        # 根据触发逻辑判断
        if self.trigger_logic == "any":
            should_trigger = error_trigger or data_count_trigger or time_trigger
        else:  # "all"
            should_trigger = error_trigger and data_count_trigger and time_trigger
        
        # 更新触发统计
        if should_trigger:
            if error_trigger:
                self.trigger_counts["error"] += 1
            if data_count_trigger:
                self.trigger_counts["data_count"] += 1
            if time_trigger:
                self.trigger_counts["time_interval"] += 1
            
            logger.info(f"触发在线学习: {', '.join(reasons)}")
        
        return should_trigger, reasons
    
    def create_training_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        从训练缓存中创建训练批次
        
        Returns:
            (batch_emg, batch_targets): 训练批次或None（如果数据不足）
        """
        if len(self.training_buffer) < self.batch_size:
            logger.warning(f"训练数据不足: {len(self.training_buffer)} < {self.batch_size}")
            return None
        
        # 随机采样训练样本
        indices = np.random.choice(len(self.training_buffer), self.batch_size, replace=False)
        
        batch_emg = []
        batch_targets = []
        
        for idx in indices:
            emg, target = self.training_buffer[idx]
            batch_emg.append(emg)
            batch_targets.append(target)
        
        # 转换为张量并移动到设备
        batch_emg = torch.stack(batch_emg).to(self.device)
        batch_targets = torch.stack(batch_targets).to(self.device)
        
        return batch_emg, batch_targets
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            predictions: 模型预测，形状为(batch_size, num_joints, time_steps)
            targets: 目标值，形状为(batch_size, num_joints, time_steps)
            mask: 可选的掩码，形状为(batch_size, time_steps)
            
        Returns:
            损失值
        """
        # 使用MSE损失
        if mask is not None:
            # 应用掩码
            mask = mask.unsqueeze(1)  # (batch_size, 1, time_steps)
            valid_predictions = predictions * mask
            valid_targets = targets * mask
            loss = nn.functional.mse_loss(valid_predictions, valid_targets, reduction='sum')
            loss = loss / mask.sum()  # 归一化
        else:
            loss = nn.functional.mse_loss(predictions, targets)
        
        return loss
    
    def mini_finetune(self) -> Dict[str, Any]:
        """
        执行小批量微调
        
        Returns:
            训练统计信息
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        start_time = time.time()
        
        # 创建训练批次
        batch_data = self.create_training_batch()
        if batch_data is None:
            return {"status": "insufficient_data"}
        
        batch_emg, batch_targets = batch_data
        
        # 设置模型为训练模式
        self.model.train()
        
        try:
            # 前向传播
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                # 创建模拟批次字典
                batch_dict = {
                    "emg": batch_emg,
                    "joint_angles": batch_targets,
                    "no_ik_failure": torch.ones(batch_emg.shape[0], batch_emg.shape[2], dtype=torch.bool, device=self.device)
                }
                
                predictions, targets, mask = self.model(batch_dict, provide_initial_pos=False)
            
            # 计算损失
            loss = self.compute_loss(predictions, targets, mask)
            
            # 反向传播
            self.optimizers["main"].zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizers["main"].step()
            
            # 更新学习率调度器
            self.schedulers["main"].step(loss.item())
            
            # 记录统计信息
            training_time = time.time() - start_time
            self.update_times.append(training_time)
            self.losses.append(loss.item())
            self.total_updates += 1
            self.trigger_counts["total"] += 1
            
            # 重置学习触发状态
            self.last_update_time = time.time()
            self.new_data_count = 0
            self.recent_errors.clear()
            
            # 保持统计列表长度  
            if len(self.update_times) > 100:
                self.update_times = self.update_times[-50:]
            if len(self.losses) > 100:
                self.losses = self.losses[-50:]
            
            logger.info(f"在线学习完成 - 损失: {loss.item():.6f}, 梯度范数: {grad_norm:.4f}, 用时: {training_time:.3f}s")
            
            return {
                "status": "success",
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "training_time": training_time,
                "batch_size": batch_emg.shape[0],
                "total_updates": self.total_updates
            }
        
        except Exception as e:
            logger.error(f"在线学习失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "training_time": time.time() - start_time
            }
        
        finally:
            # 恢复模型为评估模式
            self.model.eval()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = {
            "enabled": self.enabled,
            "total_updates": self.total_updates,
            "new_data_count": self.new_data_count,
            "training_buffer_size": len(self.training_buffer),
            "error_history_size": len(self.error_history),
            "trigger_counts": self.trigger_counts.copy()
        }
        
        if self.error_history:
            stats.update({
                "avg_error": np.mean(self.error_history),
                "recent_avg_error": np.mean(list(self.recent_errors)) if self.recent_errors else 0,
                "error_threshold": self.error_threshold
            })
        
        if self.update_times:
            stats.update({
                "avg_update_time": np.mean(self.update_times),
                "total_update_time": np.sum(self.update_times)
            })
        
        if self.losses:
            stats.update({
                "avg_loss": np.mean(self.losses),
                "recent_loss": self.losses[-1],
                "loss_trend": "decreasing" if len(self.losses) > 1 and self.losses[-1] < self.losses[-2] else "increasing"
            })
        
        # 时间相关统计
        current_time = time.time()
        stats.update({
            "time_since_last_update": current_time - self.last_update_time,
            "time_threshold": self.time_interval_threshold
        })
        
        # 学习率信息
        if self.optimizers:
            current_lrs = []
            for param_group in self.optimizers["main"].param_groups:
                current_lrs.append({
                    "name": param_group.get("name", "unknown"),
                    "lr": param_group["lr"]
                })
            stats["current_learning_rates"] = current_lrs
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """动态更新配置"""
        old_enabled = self.enabled
        
        # 更新配置
        for key, value in new_config.items():
            if key in self.config:
                if key == "learning_rates":
                    self.config[key].update(value)
                elif key == "trigger_conditions":
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # 更新相关属性
        self.enabled = self.config["enabled"]
        
        if "learning_rates" in new_config:
            self.encoder_lr = self.config["learning_rates"].get("encoder", self.encoder_lr)
            self.decoder_lr = self.config["learning_rates"].get("decoder", self.decoder_lr)
        
        if "trigger_conditions" in new_config:
            trigger_config = self.config["trigger_conditions"]
            self.error_threshold = trigger_config.get("error_threshold", self.error_threshold)
            self.data_count_threshold = trigger_config.get("data_count", self.data_count_threshold)
            self.time_interval_threshold = trigger_config.get("time_interval", self.time_interval_threshold)
            self.trigger_logic = trigger_config.get("trigger_logic", self.trigger_logic)
        
        if "batch_size" in new_config:
            self.batch_size = new_config["batch_size"]
        
        if "max_grad_norm" in new_config:
            self.max_grad_norm = new_config["max_grad_norm"]
        
        # 如果启用状态改变，重新设置优化器
        if old_enabled != self.enabled:
            if self.enabled:
                self.setup_optimizers()
            else:
                self.optimizers = None
                self.schedulers = None
        
        # 如果学习率改变，更新优化器
        elif self.enabled and "learning_rates" in new_config:
            self.setup_optimizers()
        
        logger.info(f"在线学习器配置已更新: {new_config}")
    
    def reset(self) -> None:
        """重置学习器状态"""
        self.last_update_time = time.time()
        self.new_data_count = 0
        self.error_history.clear()
        self.recent_errors.clear()
        self.training_buffer.clear()
        
        # 重置统计
        self.total_updates = 0
        self.trigger_counts = {
            "error": 0,
            "data_count": 0,
            "time_interval": 0, 
            "total": 0
        }
        self.update_times.clear()
        self.losses.clear()
        
        logger.info("在线学习器状态已重置")
