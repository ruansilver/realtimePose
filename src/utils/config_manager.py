# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
配置管理器

结合YAML配置文件和命令行参数的配置系统。
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器，支持YAML配置文件和命令行参数的组合使用"""
    
    @staticmethod
    def load_config(
        yaml_path: Union[str, Path], 
        args: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        加载并合并YAML配置和命令行参数
        
        Args:
            yaml_path: YAML配置文件路径
            args: 命令行参数列表，默认为None则使用sys.argv[1:]
            
        Returns:
            合并后的配置字典
        """
        # 加载YAML配置
        config = ConfigManager._load_yaml_config(yaml_path)
        
        # 解析命令行参数
        if args is not None:
            cli_args = ConfigManager._parse_cli_args(args)
            config = ConfigManager._merge_configs(config, cli_args)
        
        # 验证配置
        ConfigManager.validate_config(config)
        
        logger.info(f"配置已加载自: {yaml_path}")
        return config
    
    @staticmethod
    def _load_yaml_config(yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """加载YAML配置文件"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"配置文件为空: {yaml_path}")
        
        return config
    
    @staticmethod
    def _parse_cli_args(args: list) -> Dict[str, Any]:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="实时sEMG姿态预测系统",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # 数据相关参数
        parser.add_argument(
            "--dataset-path", 
            type=str,
            help="数据集路径"
        )
        parser.add_argument(
            "--batch-size", 
            type=int,
            help="批次大小"
        )
        
        # 实时预测参数
        parser.add_argument(
            "--fast-window-size", 
            type=float,
            help="快速预测窗口大小（秒）"
        )
        parser.add_argument(
            "--correction-window-size", 
            type=float,
            help="校正预测窗口大小（秒）"
        )
        parser.add_argument(
            "--max-latency", 
            type=int,
            help="最大延迟（毫秒）"
        )
        
        # 在线学习参数
        parser.add_argument(
            "--enable-online-learning", 
            action="store_true",
            help="启用在线学习"
        )
        parser.add_argument(
            "--encoder-lr", 
            type=float,
            help="编码器学习率"
        )
        parser.add_argument(
            "--decoder-lr", 
            type=float,
            help="解码器学习率"
        )
        
        # 缓冲区参数
        parser.add_argument(
            "--raw-buffer-size", 
            type=int,
            help="原始数据缓冲区大小"
        )
        parser.add_argument(
            "--feature-buffer-size", 
            type=int,
            help="特征缓冲区大小"
        )
        
        # 日志参数
        parser.add_argument(
            "--log-level", 
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="日志级别"
        )
        parser.add_argument(
            "--tensorboard-dir", 
            type=str,
            help="TensorBoard日志目录"
        )
        
        # 检查点参数
        parser.add_argument(
            "--checkpoint-path", 
            type=str,
            help="预训练模型检查点路径"
        )
        parser.add_argument(
            "--save-dir", 
            type=str,
            help="检查点保存目录"
        )
        
        parsed_args = parser.parse_args(args)
        return {k: v for k, v in vars(parsed_args).items() if v is not None}
    
    @staticmethod
    def _merge_configs(base_config: Dict[str, Any], cli_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并基础配置和命令行配置"""
        merged = base_config.copy()
        
        # 映射命令行参数到配置结构
        cli_mapping = {
            "dataset_path": ("data", "dataset_path"),
            "batch_size": ("data", "batch_size"),
            "fast_window_size": ("realtime", "fast_window_size"),
            "correction_window_size": ("realtime", "correction_window_size"),
            "max_latency": ("realtime", "max_latency"),
            "enable_online_learning": ("online_learning", "enabled"),
            "encoder_lr": ("online_learning", "learning_rates", "encoder"),
            "decoder_lr": ("online_learning", "learning_rates", "decoder"),
            "raw_buffer_size": ("buffers", "raw_data", "size"),
            "feature_buffer_size": ("buffers", "feature", "size"),
            "log_level": ("logging", "level"),
            "tensorboard_dir": ("logging", "tensorboard_dir"),
            "checkpoint_path": ("checkpoint_path",),
            "save_dir": ("checkpoint", "save_dir"),
        }
        
        for cli_key, config_path in cli_mapping.items():
            if cli_key in cli_config:
                ConfigManager._set_nested_value(merged, config_path, cli_config[cli_key])
        
        return merged
    
    @staticmethod
    def _set_nested_value(config: Dict[str, Any], path: tuple, value: Any):
        """设置嵌套字典的值"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """验证配置文件的完整性和正确性"""
        required_sections = [
            "data", "model", "realtime", "buffers", 
            "online_learning", "simulation", "checkpoint", "logging"
        ]
        
        # 检查必需的顶级配置节
        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证数据配置
        data_config = config["data"]
        required_data_keys = ["dataset_path", "batch_size"]
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"数据配置中缺少必需的键: {key}")
        
        # 验证数据集路径存在
        dataset_path = Path(data_config["dataset_path"])
        if not dataset_path.exists():
            logger.warning(f"数据集路径不存在: {dataset_path}")
        
        # 验证模型配置
        model_config = config["model"]
        required_model_keys = ["type", "encoder", "decoder"]
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"模型配置中缺少必需的键: {key}")
        
        # 验证实时配置
        realtime_config = config["realtime"]
        if realtime_config["fast_window_size"] >= realtime_config["correction_window_size"]:
            raise ValueError("快速窗口大小必须小于校正窗口大小")
        
        if realtime_config["max_latency"] <= 0:
            raise ValueError("最大延迟必须大于0")
        
        # 验证缓冲区配置
        buffers_config = config["buffers"]
        for buffer_name in ["raw_data", "feature"]:
            if buffer_name not in buffers_config:
                raise ValueError(f"缓冲区配置中缺少: {buffer_name}")
            if buffers_config[buffer_name]["size"] <= 0:
                raise ValueError(f"{buffer_name}缓冲区大小必须大于0")
        
        # 验证在线学习配置
        online_config = config["online_learning"]
        if online_config["enabled"]:
            if "learning_rates" not in online_config:
                raise ValueError("启用在线学习时必须提供学习率配置")
            
            lr_config = online_config["learning_rates"]
            for lr_name in ["encoder", "decoder"]:
                if lr_name not in lr_config:
                    raise ValueError(f"缺少{lr_name}学习率配置")
                try:
                    lr_value = float(lr_config[lr_name])
                    if lr_value <= 0:
                        raise ValueError(f"无效的{lr_name}学习率: {lr_value}")
                except (ValueError, TypeError):
                    raise ValueError(f"无效的{lr_name}学习率格式: {lr_config[lr_name]}")
        
        logger.info("配置验证通过")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """保存配置到YAML文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已保存到: {save_path}")
    
    @staticmethod
    def get_nested_value(config: Dict[str, Any], path: str, default=None) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
