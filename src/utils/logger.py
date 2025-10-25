# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
日志管理器

提供实时系统的日志记录和TensorBoard集成功能。
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class RealtimeLogger:
    """
    实时日志管理器
    
    集成标准日志和TensorBoard，用于实时系统的监控和调试。
    """
    
    def __init__(
        self, 
        log_level: str = "INFO",
        tensorboard_dir: str = "./logs",
        log_file: Optional[str] = None,
        log_interval: int = 10
    ):
        """
        初始化日志管理器
        
        Args:
            log_level: 日志级别
            tensorboard_dir: TensorBoard日志目录
            log_file: 日志文件路径（可选）
            log_interval: 日志记录间隔
        """
        self.log_level = log_level
        self.tensorboard_dir = Path(tensorboard_dir)
        self.log_interval = log_interval
        
        # 创建日志目录
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 首先设置logger，这样其他方法可以使用它
        self.logger = logging.getLogger(__name__)
        
        # 初始化标准日志
        self._setup_standard_logging(log_file)
        
        # 初始化TensorBoard
        self._setup_tensorboard()
        
        # 日志计数器
        self.log_count = 0
        self.start_time = time.time()
        
        # 性能统计
        self.metrics_buffer = {}
        
        self.logger.info("实时日志管理器初始化完成")
    
    def _setup_standard_logging(self, log_file: Optional[str]) -> None:
        """设置标准日志"""
        # 创建根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（如果指定）
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def _setup_tensorboard(self) -> None:
        """设置TensorBoard"""
        try:
            # 创建带时间戳的运行目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.tensorboard_dir / f"run_{timestamp}"
            
            self.tb_writer = SummaryWriter(log_dir=str(run_dir))
            
            self.logger.info(f"TensorBoard日志目录: {run_dir}")
            
        except Exception as e:
            self.logger.error(f"TensorBoard初始化失败: {e}")
            self.tb_writer = None
    
    def log_prediction(
        self, 
        prediction: torch.Tensor, 
        timestamp: float,
        target: Optional[torch.Tensor] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录预测结果
        
        Args:
            prediction: 预测结果
            timestamp: 时间戳
            target: 真实目标（可选）
            metrics: 预测指标（可选）
        """
        self.log_count += 1
        
        # 只在指定间隔记录
        if self.log_count % self.log_interval != 0:
            return
        
        try:
            step = self.log_count // self.log_interval
            elapsed_time = timestamp - self.start_time
            
            # 记录预测统计
            if self.tb_writer:
                pred_np = prediction.detach().cpu().numpy()
                
                # 记录预测值分布
                self.tb_writer.add_histogram('prediction/values', pred_np, step)
                self.tb_writer.add_scalar('prediction/mean', np.mean(pred_np), step)
                self.tb_writer.add_scalar('prediction/std', np.std(pred_np), step)
                self.tb_writer.add_scalar('prediction/max', np.max(pred_np), step)
                self.tb_writer.add_scalar('prediction/min', np.min(pred_np), step)
                
                # 记录时间信息
                self.tb_writer.add_scalar('timing/elapsed_time', elapsed_time, step)
                self.tb_writer.add_scalar('timing/predictions_per_second', self.log_count / elapsed_time, step)
            
            # 记录误差（如果有目标）
            if target is not None:
                error = torch.nn.functional.mse_loss(prediction, target).item()
                if self.tb_writer:
                    self.tb_writer.add_scalar('error/mse', error, step)
                
                self.logger.debug(f"预测误差: {error:.6f}")
            
            # 记录额外指标
            if metrics:
                self._log_metrics(metrics, step)
            
        except Exception as e:
            self.logger.error(f"记录预测失败: {e}")
    
    def log_learning_update(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        update_count: int,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录学习更新
        
        Args:
            loss: 损失值
            learning_rate: 学习率
            grad_norm: 梯度范数
            update_count: 更新次数
            additional_metrics: 额外指标
        """
        try:
            if self.tb_writer:
                self.tb_writer.add_scalar('learning/loss', loss, update_count)
                self.tb_writer.add_scalar('learning/learning_rate', learning_rate, update_count)
                self.tb_writer.add_scalar('learning/grad_norm', grad_norm, update_count)
            
            # 记录额外指标
            if additional_metrics:
                self._log_metrics(additional_metrics, update_count, prefix='learning/')
            
            self.logger.info(f"学习更新 #{update_count} - 损失: {loss:.6f}, 学习率: {learning_rate:.2e}, 梯度范数: {grad_norm:.4f}")
            
        except Exception as e:
            self.logger.error(f"记录学习更新失败: {e}")
    
    def log_buffer_status(self, buffer_status: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录缓冲区状态
        
        Args:
            buffer_status: 缓冲区状态信息
            step: 步数（可选）
        """
        try:
            if step is None:
                step = self.log_count // self.log_interval
            
            if self.tb_writer:
                for buffer_name, status in buffer_status.items():
                    if isinstance(status, dict):
                        for key, value in status.items():
                            if isinstance(value, (int, float)):
                                self.tb_writer.add_scalar(f'buffer/{buffer_name}_{key}', value, step)
            
            self.logger.debug(f"缓冲区状态: {buffer_status}")
            
        except Exception as e:
            self.logger.error(f"记录缓冲区状态失败: {e}")
    
    def log_performance_stats(self, stats: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录性能统计
        
        Args:
            stats: 性能统计信息
            step: 步数（可选）
        """
        try:
            if step is None:
                step = self.log_count // self.log_interval
            
            if self.tb_writer:
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f'performance/{key}', value, step)
            
            # 记录关键性能指标
            key_metrics = ['avg_total_latency', 'p95_total_latency', 'prediction_count']
            log_parts = []
            for metric in key_metrics:
                if metric in stats:
                    log_parts.append(f"{metric}: {stats[metric]}")
            
            if log_parts:
                self.logger.info(f"性能统计 - {', '.join(log_parts)}")
            
        except Exception as e:
            self.logger.error(f"记录性能统计失败: {e}")
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float, gpu_usage: Optional[float] = None) -> None:
        """
        记录系统指标
        
        Args:
            cpu_usage: CPU使用率（百分比）
            memory_usage: 内存使用率（百分比）
            gpu_usage: GPU使用率（百分比，可选）
        """
        try:
            step = self.log_count // self.log_interval
            
            if self.tb_writer:
                self.tb_writer.add_scalar('system/cpu_usage', cpu_usage, step)
                self.tb_writer.add_scalar('system/memory_usage', memory_usage, step)
                
                if gpu_usage is not None:
                    self.tb_writer.add_scalar('system/gpu_usage', gpu_usage, step)
            
            log_msg = f"系统指标 - CPU: {cpu_usage:.1f}%, 内存: {memory_usage:.1f}%"
            if gpu_usage is not None:
                log_msg += f", GPU: {gpu_usage:.1f}%"
            
            self.logger.debug(log_msg)
            
        except Exception as e:
            self.logger.error(f"记录系统指标失败: {e}")
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """
        记录模型信息
        
        Args:
            model: 模型
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if self.tb_writer:
                self.tb_writer.add_scalar('model/total_parameters', total_params, 0)
                self.tb_writer.add_scalar('model/trainable_parameters', trainable_params, 0)
            
            self.logger.info(f"模型信息 - 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"记录模型信息失败: {e}")
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = '') -> None:
        """记录指标到TensorBoard"""
        if not self.tb_writer:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(f'{prefix}{key}', value, step)
            elif isinstance(value, dict):
                self._log_metrics(value, step, f'{prefix}{key}/')
    
    def log_event(self, message: str, level: str = "INFO") -> None:
        """
        记录事件
        
        Args:
            message: 事件消息
            level: 日志级别
        """
        try:
            log_func = getattr(self.logger, level.lower())
            log_func(message)
            
            # 也记录到TensorBoard作为文本
            if self.tb_writer:
                step = self.log_count // self.log_interval
                self.tb_writer.add_text('events', message, step)
                
        except Exception as e:
            self.logger.error(f"记录事件失败: {e}")
    
    def add_figure(self, tag: str, figure, step: Optional[int] = None) -> None:
        """
        添加图表到TensorBoard
        
        Args:
            tag: 标签
            figure: matplotlib图表
            step: 步数
        """
        try:
            if self.tb_writer and figure is not None:
                if step is None:
                    step = self.log_count // self.log_interval
                self.tb_writer.add_figure(tag, figure, step)
                
        except Exception as e:
            self.logger.error(f"添加图表失败: {e}")
    
    def flush(self) -> None:
        """刷新日志缓冲区"""
        try:
            if self.tb_writer:
                self.tb_writer.flush()
                
        except Exception as e:
            self.logger.error(f"刷新日志失败: {e}")
    
    def close(self) -> None:
        """关闭日志管理器"""
        try:
            if self.tb_writer:
                self.tb_writer.close()
                self.tb_writer = None
            
            self.logger.info("日志管理器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭日志管理器失败: {e}")
    
    def get_tensorboard_url(self) -> str:
        """获取TensorBoard访问URL"""
        return f"http://localhost:6006"
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        elapsed_time = time.time() - self.start_time
        
        return {
            "log_count": self.log_count,
            "elapsed_time": elapsed_time,
            "logs_per_second": self.log_count / elapsed_time if elapsed_time > 0 else 0,
            "tensorboard_dir": str(self.tensorboard_dir),
            "log_level": self.log_level,
            "log_interval": self.log_interval
        }


def setup_logging(config: Dict[str, Any]) -> RealtimeLogger:
    """
    根据配置设置日志系统
    
    Args:
        config: 日志配置
        
    Returns:
        配置好的日志管理器
    """
    return RealtimeLogger(
        log_level=config.get("level", "INFO"),
        tensorboard_dir=config.get("tensorboard_dir", "./logs"),
        log_file=config.get("log_file"),
        log_interval=config.get("log_interval", 10)
    )
