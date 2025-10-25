# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
实时sEMG姿态预测系统 - 主入口脚本

具备在线学习能力的实时姿态预测系统，支持低延迟预测、动态模型适应和全面的监控功能。
"""

import sys
import time
import signal
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import logging

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_manager import ConfigManager
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.logger import RealtimeLogger, setup_logging
from src.models.realtime_pose_model import create_realtime_pose_model, load_pretrained_weights
from src.core.realtime_predictor import RealtimePredictor
from src.core.buffer_manager import BufferManager
from src.core.online_learner import OnlineLearner
from src.core.data_simulator import DataSimulator

logger = logging.getLogger(__name__)


class RealtimePoseSystem:
    """
    实时姿态预测系统主类
    
    整合所有组件，提供完整的实时预测和在线学习功能。
    """
    
    def __init__(self, config_path: str, args: Optional[list] = None):
        """
        初始化实时姿态预测系统
        
        Args:
            config_path: 配置文件路径
            args: 命令行参数
        """
        # 加载配置
        self.config = ConfigManager.load_config(config_path, args)
        
        # 初始化日志系统
        self.logger_manager = setup_logging(self.config["logging"])
        
        # 系统组件
        self.model = None
        self.predictor = None
        self.buffer_manager = None
        self.online_learner = None
        self.data_simulator = None
        self.checkpoint_manager = None
        
        # 系统状态
        self.running = False
        self.start_time = None
        self.total_predictions = 0
        self.total_learning_updates = 0
        
        # 性能监控
        self.last_stats_time = time.time()
        self.stats_interval = 30.0  # 每30秒记录一次统计
        
        logger.info("实时姿态预测系统初始化开始")
        
        # 初始化各组件
        self._initialize_components()
        
        logger.info("实时姿态预测系统初始化完成")
    
    def _initialize_components(self) -> None:
        """初始化系统各组件"""
        try:
            # 1. 创建模型
            logger.info("创建实时姿态预测模型...")
            self.model = create_realtime_pose_model(self.config["model"])
            
            # 记录模型信息
            self.logger_manager.log_model_info(self.model)
            
            # 2. 加载预训练权重
            if "checkpoint_path" in self.config and self.config["checkpoint_path"]:
                logger.info("加载预训练权重...")
                load_pretrained_weights(self.model, self.config["checkpoint_path"], strict=False)
            
            # 3. 创建缓冲区管理器
            logger.info("初始化缓冲区管理器...")
            self.buffer_manager = BufferManager(self.config["buffers"])
            
            # 4. 创建实时预测器
            logger.info("初始化实时预测器...")
            self.predictor = RealtimePredictor(self.model, self.config["realtime"])
            
            # 5. 创建在线学习器
            logger.info("初始化在线学习器...")
            self.online_learner = OnlineLearner(self.model, self.config["online_learning"])
            
            # 6. 创建数据模拟器
            logger.info("初始化数据模拟器...")
            self.data_simulator = DataSimulator(
                self.config["data"]["dataset_path"], 
                self.config["simulation"]
            )
            
            # 7. 创建检查点管理器
            logger.info("初始化检查点管理器...")
            self.checkpoint_manager = CheckpointManager(
                self.config["checkpoint"]["save_dir"],
                self.config["checkpoint"]["keep_last_n"]
            )
            
            # 设置模型为评估模式
            self.model.eval()
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，开始优雅停机...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self) -> None:
        """运行实时预测系统"""
        logger.info("启动实时姿态预测系统")
        
        try:
            self._setup_signal_handlers()
            self.running = True
            self.start_time = time.time()
            
            # 记录启动事件
            self.logger_manager.log_event("实时预测系统启动", "INFO")
            
            # 启动主循环
            self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("用户中断，停止系统")
        except Exception as e:
            logger.error(f"系统运行出错: {e}")
            raise
        finally:
            self._cleanup()
    
    def _main_loop(self) -> None:
        """主预测循环"""
        logger.info("开始实时预测循环")
        
        # 启动数据流
        data_stream = self.data_simulator.simulate_realtime_stream()
        
        try:
            for data_packet in data_stream:
                if not self.running:
                    break
                
                # 处理数据包
                self._process_data_packet(data_packet)
                
                # 执行预测
                prediction, metrics = self._execute_prediction()
                
                if prediction is not None:
                    # 处理预测结果
                    self._handle_prediction(prediction, metrics, data_packet)
                    
                    # 检查是否需要在线学习
                    self._check_online_learning(data_packet)
                    
                    # 定期保存检查点
                    self._check_checkpoint_save()
                    
                    # 记录统计信息
                    self._log_periodic_stats()
                
        except Exception as e:
            logger.error(f"主循环出错: {e}")
            raise
    
    def _process_data_packet(self, data_packet) -> None:
        """处理数据包"""
        try:
            # 添加EMG数据到缓冲区
            self.buffer_manager.add_raw_data(data_packet.emg_data)
            
            # 可以在这里添加特征提取和缓存逻辑
            
        except Exception as e:
            logger.error(f"处理数据包失败: {e}")
    
    def _execute_prediction(self) -> tuple:
        """执行预测"""
        try:
            prediction, metrics = self.predictor.predict(self.buffer_manager)
            self.total_predictions += 1
            return prediction, metrics
            
        except Exception as e:
            logger.error(f"预测执行失败: {e}")
            return None, {}
    
    def _handle_prediction(self, prediction, metrics, data_packet) -> None:
        """处理预测结果"""
        try:
            # 跳过零预测的误差计算
            if prediction is None or torch.allclose(prediction, torch.zeros_like(prediction)):
                logger.debug("跳过零预测的误差计算")
                return
            
            # 计算预测误差（如果有目标数据）
            # 确保目标数据形状与预测一致
            if hasattr(data_packet, 'joint_angles') and data_packet.joint_angles is not None:
                target_joints = data_packet.joint_angles[-1]  # 取最后一个时间步的关节角度 (20,)
                target = torch.from_numpy(target_joints).unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
                
                # 确保预测和目标在同一设备上
                target = target.to(prediction.device)
                
                # 确保形状匹配
                if prediction.shape != target.shape:
                    # 调整预测或目标的形状
                    if prediction.dim() == 2 and target.dim() == 3:
                        prediction = prediction.unsqueeze(-1)  # (1, 20) -> (1, 20, 1)
                    elif prediction.dim() == 3 and target.dim() == 2:
                        target = target.unsqueeze(-1)  # (1, 20) -> (1, 20, 1)
                
                # 计算MSE误差
                error = torch.nn.functional.mse_loss(prediction, target).item()
                
                # 记录到在线学习器
                self.online_learner.add_prediction_error(error)
                
                # 添加训练样本
                emg_data = torch.from_numpy(data_packet.emg_data.T).unsqueeze(0)  # (1, channels, time)
                joint_data = torch.from_numpy(data_packet.joint_angles.T).unsqueeze(0)  # (1, joints, time)
                self.online_learner.add_training_sample(emg_data, joint_data)
                
                # 记录预测日志
                self.logger_manager.log_prediction(
                    prediction, 
                    data_packet.timestamp, 
                    target,
                    metrics
                )
            else:
                logger.debug("数据包中没有关节角度数据，跳过误差计算")
            
            # 记录缓冲区状态
            if "buffer_status" in metrics:
                self.logger_manager.log_buffer_status(metrics["buffer_status"])
            
        except Exception as e:
            logger.error(f"处理预测结果失败: {e}")
            logger.debug(f"预测形状: {prediction.shape if prediction is not None else 'None'}")
            if hasattr(data_packet, 'joint_angles') and data_packet.joint_angles is not None:
                logger.debug(f"目标数据形状: {data_packet.joint_angles.shape}")
    
    def _check_online_learning(self, data_packet) -> None:
        """检查并执行在线学习"""
        try:
            should_trigger, reasons = self.online_learner.should_trigger_learning()
            
            if should_trigger:
                logger.info(f"触发在线学习: {', '.join(reasons)}")
                
                # 执行在线学习
                learning_result = self.online_learner.mini_finetune()
                
                if learning_result["status"] == "success":
                    self.total_learning_updates += 1
                    
                    # 记录学习更新
                    self.logger_manager.log_learning_update(
                        learning_result["loss"],
                        learning_result.get("learning_rate", 0),
                        learning_result["grad_norm"],
                        self.total_learning_updates,
                        learning_result
                    )
                    
                    logger.info(f"在线学习完成，总更新次数: {self.total_learning_updates}")
                else:
                    logger.warning(f"在线学习失败: {learning_result.get('error', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"在线学习检查失败: {e}")
    
    def _check_checkpoint_save(self) -> None:
        """检查是否需要保存检查点"""
        try:
            if self.start_time is None:
                return
            
            elapsed_time = time.time() - self.start_time
            save_interval = self.config["checkpoint"]["save_interval"]
            
            # 检查是否到了保存时间
            if elapsed_time > 0 and int(elapsed_time) % save_interval == 0:
                # 避免重复保存
                if not hasattr(self, '_last_checkpoint_time') or \
                   elapsed_time - getattr(self, '_last_checkpoint_time', 0) >= save_interval:
                    
                    self._save_checkpoint()
                    self._last_checkpoint_time = elapsed_time
            
        except Exception as e:
            logger.error(f"检查点保存检查失败: {e}")
    
    def _save_checkpoint(self) -> None:
        """保存检查点"""
        try:
            metadata = {
                "total_predictions": self.total_predictions,
                "total_learning_updates": self.total_learning_updates,
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
                "predictor_stats": self.predictor.get_performance_stats(),
                "learning_stats": self.online_learner.get_learning_stats(),
                "system_config": self.config
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model,
                self.online_learner.optimizers["main"] if self.online_learner.optimizers else None,
                metadata=metadata
            )
            
            logger.info(f"检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _log_periodic_stats(self) -> None:
        """定期记录统计信息"""
        try:
            current_time = time.time()
            
            if current_time - self.last_stats_time >= self.stats_interval:
                # 记录预测器性能统计
                predictor_stats = self.predictor.get_performance_stats()
                self.logger_manager.log_performance_stats(predictor_stats)
                
                # 记录在线学习统计
                learning_stats = self.online_learner.get_learning_stats()
                if learning_stats:
                    for key, value in learning_stats.items():
                        if isinstance(value, (int, float)):
                            self.logger_manager.tb_writer.add_scalar(f'learning_stats/{key}', value, 
                                                                   self.total_predictions)
                
                # 记录系统指标
                process = psutil.Process()
                cpu_usage = process.cpu_percent()
                memory_usage = process.memory_percent()
                
                # GPU使用率（如果可用）
                gpu_usage = None
                if torch.cuda.is_available():
                    try:
                        gpu_usage = torch.cuda.utilization()
                    except:
                        pass
                
                self.logger_manager.log_system_metrics(cpu_usage, memory_usage, gpu_usage)
                
                # 记录数据模拟器统计
                sim_stats = self.data_simulator.get_statistics()
                self.logger_manager.log_event(
                    f"数据流统计 - 包/秒: {sim_stats.get('packets_per_second', 0):.1f}, "
                    f"样本/秒: {sim_stats.get('samples_per_second', 0):.1f}",
                    "INFO"
                )
                
                self.last_stats_time = current_time
                
                # 刷新日志
                self.logger_manager.flush()
                
        except Exception as e:
            logger.error(f"记录统计信息失败: {e}")
    
    def stop(self) -> None:
        """停止系统"""
        logger.info("停止实时预测系统")
        self.running = False
    
    def _cleanup(self) -> None:
        """清理资源"""
        try:
            logger.info("开始清理系统资源")
            
            # 保存最终检查点
            if self.checkpoint_manager and self.model:
                logger.info("保存最终检查点...")
                self._save_checkpoint()
            
            # 记录最终统计
            if self.start_time:
                total_time = time.time() - self.start_time
                avg_predictions_per_sec = self.total_predictions / total_time if total_time > 0 else 0
                
                self.logger_manager.log_event(
                    f"系统运行完成 - 总预测数: {self.total_predictions}, "
                    f"总学习更新: {self.total_learning_updates}, "
                    f"运行时间: {total_time:.1f}s, "
                    f"平均预测率: {avg_predictions_per_sec:.1f}/s",
                    "INFO"
                )
            
            # 关闭日志管理器
            if self.logger_manager:
                self.logger_manager.close()
            
            logger.info("系统资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


def main():
    """主函数"""
    try:
        # 默认配置文件路径
        config_path = "configs/realtime_config.yaml"
        
        # 创建并运行系统
        system = RealtimePoseSystem(config_path, sys.argv[1:])
        system.run()
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
