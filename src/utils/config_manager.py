"""配置管理模块：加载YAML配置并支持命令行覆写"""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(yaml_path: str) -> Dict[str, Any]:
    """加载YAML配置文件
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        配置字典
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """解析命令行参数
    
    支持 --set key.subkey=value 格式的覆写
    
    Returns:
        命令行参数
    """
    parser = argparse.ArgumentParser(
        description='实时EMG姿态预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py
  python main.py --config configs/custom.yaml
  python main.py --set train.enabled=true
  python main.py --set data.batch_size=32 --set runtime.device=cpu
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/realtime_config.yaml',
        help='配置文件路径 (默认: configs/realtime_config.yaml)'
    )
    
    parser.add_argument(
        '--set',
        action='append',
        dest='overrides',
        metavar='KEY=VALUE',
        help='覆写配置项，格式: key.subkey=value (可多次使用)'
    )
    
    return parser.parse_args()


def override_config(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """合并命令行覆写到配置字典
    
    Args:
        config: 原始配置字典
        overrides: 覆写列表，格式为 ["key.subkey=value", ...]
        
    Returns:
        合并后的配置字典
    """
    if not overrides:
        return config
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"覆写格式错误: {override}，应为 key=value")
        
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # 类型推断：尝试转换为 bool, int, float, 否则保持字符串
        value = _parse_value(value_str)
        
        # 递归设置嵌套字典的值
        _set_nested_value(config, keys, value)
    
    return config


def _parse_value(value_str: str) -> Any:
    """解析字符串值为适当的Python类型"""
    # 布尔值
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # 整数
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # 字符串（移除引号）
    return value_str.strip('"\'')


def _set_nested_value(config: Dict[str, Any], keys: list[str], value: Any):
    """在嵌套字典中设置值"""
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置的基本有效性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 配置无效时抛出
    """
    # 检查必需的顶层键
    required_keys = ['data', 'model', 'runtime', 'logging']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必需的键: {key}")
    
    # 检查数据路径
    dataset_root = config['data'].get('dataset_root')
    if not dataset_root:
        raise ValueError("data.dataset_root 未配置")
    
    # 检查窗口长度
    window_length = config['data'].get('window_length')
    if not window_length or window_length <= 0:
        raise ValueError("data.window_length 必须为正整数")
    
    # 检查设备
    device = config['runtime'].get('device', 'cuda')
    if device not in ('cuda', 'cpu'):
        raise ValueError(f"runtime.device 必须为 'cuda' 或 'cpu'，当前为: {device}")
    
    # 检查模型配置
    if 'type' not in config['model']:
        raise ValueError("model.type 未配置")
    
    if 'network' not in config['model']:
        raise ValueError("model.network 未配置")
    
    if 'decoder' not in config['model']:
        raise ValueError("model.decoder 未配置")


def get_config(config_path: str = None, cli_overrides: list[str] = None) -> Dict[str, Any]:
    """获取完整配置（便捷函数）
    
    Args:
        config_path: 配置文件路径，None则使用默认值
        cli_overrides: 命令行覆写列表
        
    Returns:
        完整配置字典
    """
    if config_path is None:
        config_path = 'configs/realtime_config.yaml'
    
    config = load_config(config_path)
    
    if cli_overrides:
        config = override_config(config, cli_overrides)
    
    validate_config(config)
    
    return config

