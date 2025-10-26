"""姿态预测头：自回归rollout + 插值对齐"""

import torch
from torch import nn
from torch.nn.functional import interpolate

from src.models.decoder import SequentialLSTM


# EMG采样率常量
EMG_SAMPLE_RATE = 2000


class BasePoseModule(nn.Module):
    """基础姿态预测模块
    
    包含网络（encoder）和上下文窗口信息，预测覆盖 inputs[left_context : -right_context]，
    并上采样到与输入相同的采样率。
    """
    
    def __init__(
        self,
        network: nn.Module,
        out_channels: int = 20,
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels
        
        self.left_context = network.left_context
        self.right_context = network.right_context
    
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            batch: 包含 emg, joint_angles, no_ik_failure 的字典
            provide_initial_pos: 是否提供初始位置
            
        Returns:
            (pred, joint_angles, no_ik_failure) 元组
        """
        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        no_ik_failure = batch["no_ik_failure"]
        
        # 获取初始位置
        initial_pos = joint_angles[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)
        
        # 生成预测
        pred = self._predict_pose(emg, initial_pos)
        
        # 裁剪joint_angles以匹配预测的范围
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]
        no_ik_failure = no_ik_failure[..., slice(start, stop)]
        
        # 匹配预测的采样率到joint_angles
        n_time = joint_angles.shape[-1]
        pred = self.align_predictions(pred, n_time)
        no_ik_failure = self.align_mask(no_ik_failure, n_time)
        
        return pred, joint_angles, no_ik_failure
    
    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        """预测姿态（子类实现）"""
        raise NotImplementedError
    
    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """时序重采样预测以匹配目标长度（线性插值）"""
        return interpolate(pred, size=n_time, mode="linear")
    
    def align_mask(self, mask: torch.Tensor, n_time: int):
        """时序重采样掩码以匹配目标长度（最近邻插值）"""
        # 2D输入不支持interpolate()，添加dummy通道维度
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)


class VEMG2PoseWithInitialState(BasePoseModule):
    """位置+速度混合预测模块
    
    前num_position_steps步预测位置，之后积分速度。
    
    Args:
        network: 编码器网络（Encoder）
        decoder: 解码器（SequentialLSTM）
        num_position_steps: 预测位置的步数（在2kHz采样率下）
        state_condition: 是否条件于前一步状态
        rollout_freq: rollout频率（Hz）
    """
    
    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        num_position_steps: int,
        state_condition: bool = True,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.num_position_steps = num_position_steps
        self.state_condition = state_condition
        self.rollout_freq = rollout_freq
    
    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        """预测姿态
        
        Args:
            emg: EMG信号 (B, C, T)
            initial_pos: 初始位置 (B, num_joints)
            
        Returns:
            预测姿态 (B, num_joints, T_rollout)
        """
        features = self.network(emg)  # BCT
        
        # 重采样特征到rollout频率
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)
        
        # 重置LSTM隐藏状态
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()
        
        # 计算在新采样率下的num_position_steps
        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        preds = [initial_pos]
        
        for t in range(features.shape[-1]):
            # 准备解码器输入
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)
            
            # 预测位置和速度
            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)
            
            # 前num_position_steps步预测位置，之后积分速度
            pred = pos if t < num_position_steps else preds[-1] + vel
            preds.append(pred)
        
        # 移除第一个pred（initial_pos，非网络预测）
        return torch.stack(preds[1:], dim=-1)


def build_pose_head_from_config(
    config: dict,
    network: nn.Module,
    decoder: nn.Module
) -> VEMG2PoseWithInitialState:
    """根据配置构建姿态预测头
    
    Args:
        config: 模型配置字典
        network: 已构建的编码器网络
        decoder: 已构建的解码器
        
    Returns:
        VEMG2PoseWithInitialState实例
    """
    return VEMG2PoseWithInitialState(
        network=network,
        decoder=decoder,
        num_position_steps=config['num_position_steps'],
        state_condition=config['state_condition'],
        rollout_freq=config['rollout_freq'],
    )

