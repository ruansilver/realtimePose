"""Checkpoint加载器：处理Lightning checkpoint和权重映射"""

from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn


def load_reference_checkpoint(ckpt_path: str | Path) -> Dict[str, Any]:
    """加载Lightning checkpoint并提取state_dict
    
    Args:
        ckpt_path: checkpoint文件路径
        
    Returns:
        包含模型权重的字典
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {ckpt_path}")
    
    # 加载checkpoint (设置 weights_only=False 以支持Lightning checkpoint格式)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Lightning checkpoint包含 'state_dict' 键
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    return state_dict


def load_model_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False
) -> tuple[list[str], list[str]]:
    """将权重加载到模型（处理key映射）
    
    Lightning模型的state_dict包含 'model.' 前缀，需要移除以匹配我们的模型结构。
    
    Args:
        model: 目标模型
        state_dict: 源state_dict
        strict: 是否严格匹配所有键
        
    Returns:
        (missing_keys, unexpected_keys) 元组
    """
    # 显示原始checkpoint中的键
    print(f"  原始checkpoint键样例（前10个）:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"    {key}")
        if i >= 9:
            break
    
    # 创建新的state_dict，移除 'model.' 前缀
    new_state_dict = {}
    key_mappings = []
    
    for key, value in state_dict.items():
        # 移除 Lightning 的 'model.' 前缀
        if key.startswith('model.'):
            new_key = key[6:]  # 移除 'model.'
        else:
            new_key = key
        
        new_state_dict[new_key] = value
        key_mappings.append((key, new_key))
    
    print(f"  映射后键样例（前10个）:")
    for i, (old_key, new_key) in enumerate(key_mappings[:10]):
        print(f"    {old_key} -> {new_key}")
        if i >= 9:
            break
    
    # 显示模型期望的键
    model_keys = list(model.state_dict().keys())
    print(f"  模型期望键样例（前10个）:")
    for i, key in enumerate(model_keys[:10]):
        print(f"    {key}")
        if i >= 9:
            break
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=strict
    )
    
    return missing_keys, unexpected_keys


def create_reference_model_from_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """从checkpoint创建参考模型
    
    Args:
        ckpt_path: checkpoint文件路径
        model: 模型实例
        device: 计算设备
        
    Returns:
        加载权重后的模型
    """
    # 加载checkpoint
    state_dict = load_reference_checkpoint(ckpt_path)
    
    # 加载权重
    missing_keys, unexpected_keys = load_model_weights(model, state_dict, strict=False)
    
    print(f"\n权重加载详情:")
    print(f"  checkpoint中的键数量: {len(state_dict)}")
    print(f"  模型参数数量: {len(list(model.named_parameters()))}")
    
    if missing_keys:
        print(f"  缺失的键 ({len(missing_keys)}): {missing_keys[:10]}...")
    if unexpected_keys:
        print(f"  意外的键 ({len(unexpected_keys)}): {unexpected_keys[:10]}...")
    
    # 检查是否成功加载了关键组件
    loaded_params = set(model.state_dict().keys()) - set(missing_keys)
    print(f"  成功加载的参数: {len(loaded_params)}/{len(model.state_dict())}")
    
    # 移动到设备并设置为评估模式
    model.to(device)
    model.eval()
    
    return model

