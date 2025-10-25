# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
实时姿态预测模型

包含模型构建工厂函数和配置解析逻辑。
"""

from typing import Dict, Any
import torch
from torch import nn

from .networks import TdsNetwork, SequentialLSTM, Conv1dBlock, TdsStage
from .modules import RealtimePoseModel


def create_conv_blocks(conv_configs: list[Dict[str, Any]]) -> list[Conv1dBlock]:
    """创建卷积块列表"""
    conv_blocks = []
    for config in conv_configs:
        conv_blocks.append(Conv1dBlock(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            norm_type=config.get("norm_type", "layer"),
            dropout=config.get("dropout", 0.0)
        ))
    return conv_blocks


def create_tds_stages(tds_configs: list[Dict[str, Any]]) -> list[TdsStage]:
    """创建TDS阶段列表"""
    tds_stages = []
    for config in tds_configs:
        tds_stages.append(TdsStage(
            in_channels=config["in_channels"],
            in_conv_kernel_width=config["in_conv_kernel_width"],
            in_conv_stride=config["in_conv_stride"],
            num_blocks=config["num_blocks"],
            channels=config["channels"],
            feature_width=config["feature_width"],
            kernel_width=config["kernel_width"],
            out_channels=config.get("out_channels")
        ))
    return tds_stages


def create_encoder(encoder_config: Dict[str, Any]) -> nn.Module:
    """根据配置创建编码器网络"""
    encoder_type = encoder_config["type"]
    
    if encoder_type == "TdsNetwork":
        # 创建卷积块
        conv_blocks = create_conv_blocks(encoder_config["conv_blocks"])
        
        # 创建TDS阶段
        tds_stages = create_tds_stages(encoder_config["tds_stages"])
        
        # 创建TDS网络
        return TdsNetwork(conv_blocks, tds_stages)
    
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")


def create_decoder(decoder_config: Dict[str, Any]) -> nn.Module:
    """根据配置创建解码器网络"""
    decoder_type = decoder_config["type"]
    
    if decoder_type == "SequentialLSTM":
        return SequentialLSTM(
            in_channels=decoder_config["in_channels"],
            out_channels=decoder_config["out_channels"],
            hidden_size=decoder_config["hidden_size"],
            num_layers=decoder_config["num_layers"],
            scale=decoder_config.get("scale", 1.0)
        )
    
    else:
        raise ValueError(f"不支持的解码器类型: {decoder_type}")


def create_realtime_pose_model(model_config: Dict[str, Any]) -> RealtimePoseModel:
    """根据配置创建实时姿态预测模型"""
    model_type = model_config["type"]
    
    if model_type != "RealtimePoseModel":
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建编码器
    encoder = create_encoder(model_config["encoder"])
    
    # 创建解码器  
    decoder = create_decoder(model_config["decoder"])
    
    # 创建实时姿态模型
    model = RealtimePoseModel(
        network=encoder,
        decoder=decoder,
        num_position_steps=model_config["num_position_steps"],
        state_condition=model_config["state_condition"],
        rollout_freq=model_config.get("rollout_freq", 50),
        out_channels=model_config["decoder"]["out_channels"]  # 从解码器配置获取输出通道数
    )
    
    return model


def load_pretrained_weights(
    model: RealtimePoseModel, 
    checkpoint_path: str, 
    strict: bool = False
) -> None:
    """
    加载预训练权重到模型中
    
    Args:
        model: 实时姿态预测模型
        checkpoint_path: 检查点文件路径
        strict: 是否严格匹配权重键名
    """
    try:
        # 设置weights_only=False以兼容PyTorch 2.6，因为我们信任这个检查点文件
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 提取状态字典
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # 处理键名映射（从emg2pose到实时模型）
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除模型前缀
            if key.startswith("model."):
                new_key = key[6:]
            elif key.startswith("pose_module."):
                new_key = key[12:]
            else:
                new_key = key
            
            # 映射网络组件名称
            new_key = new_key.replace("pose_module", "")
            if new_key.startswith("."):
                new_key = new_key[1:]
            
            new_state_dict[new_key] = value
        
        # 加载权重，允许部分匹配
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        
        if missing_keys:
            print(f"警告: 以下键在检查点中缺失: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 以下键在模型中未找到: {unexpected_keys}")
        
        print(f"成功加载预训练权重从: {checkpoint_path}")
        
    except Exception as e:
        print(f"加载预训练权重失败: {e}")
        raise


def get_model_info(model: RealtimePoseModel) -> Dict[str, Any]:
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "RealtimePoseModel",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "encoder_left_context": model.left_context,
        "encoder_right_context": model.right_context,
        "decoder_type": type(model.decoder).__name__,
        "num_position_steps": model.num_position_steps,
        "state_condition": model.state_condition,
        "rollout_freq": model.rollout_freq
    }
