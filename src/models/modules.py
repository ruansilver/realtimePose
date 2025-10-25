# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
姿态预测模块

包含不同类型的姿态预测模块，支持实时和增量预测。
"""

import torch
from torch import nn
from torch.nn.functional import interpolate

from .constants import EMG_SAMPLE_RATE
from .networks import SequentialLSTM


class BasePoseModule(nn.Module):
    """
    基础姿态模块，由具有左右上下文的网络组成。
    预测覆盖输入[left_context : -right_context]，并上采样以匹配输入的采样率。
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
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        no_ik_failure = batch["no_ik_failure"]

        # 获取初始位置
        initial_pos = joint_angles[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # 生成预测
        pred = self._predict_pose(emg, initial_pos)

        # 切片关节角度以匹配预测的范围
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]
        no_ik_failure = no_ik_failure[..., slice(start, stop)]

        # 将预测的采样率匹配到关节角度的采样率
        n_time = joint_angles.shape[-1]
        pred = self.align_predictions(pred, n_time)
        no_ik_failure = self.align_mask(no_ik_failure, n_time)

        return pred, joint_angles, no_ik_failure

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        raise NotImplementedError

    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """时间重新采样预测以匹配目标长度"""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int):
        """时间重新采样掩码以匹配目标长度"""
        # 2D输入对interpolate()不起作用，所以我们添加一个虚拟通道维度
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)


class PoseModule(BasePoseModule):
    """
    通过预测位置或速度来跟踪姿态，可选地给定初始状态。
    """

    def __init__(self, network: nn.Module, predict_vel: bool = False):
        super().__init__(network)
        self.predict_vel = predict_vel

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        pred = self.network(emg)  # BCT
        if self.predict_vel:
            pred = initial_pos[..., None] + torch.cumsum(pred, -1)
        return pred


class StatePoseModule(BasePoseModule):
    """
    通过预测位置或速度来跟踪姿态，可选地给定初始状态并在每个时间点以先前状态为条件。
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):

        features = self.network(emg)  # BCT
        preds = [initial_pos]

        # 将特征重新采样到展开频率
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # 重置LSTM隐藏状态
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        for t in range(features.shape[-1]):

            # 准备解码器输入
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # 预测姿态
            pred = self.decoder(inputs)
            if self.predict_vel:
                pred = pred + preds[-1]
            preds.append(pred)

        # 移除第一个预测，因为它是initial_pos（不是网络预测）
        return torch.stack(preds[1:], dim=-1)


class RealtimePoseModel(BasePoseModule):
    """
    实时姿态预测模型
    
    预测num_position_steps步的姿态，然后积分速度。
    基于原有的VEMG2PoseWithInitialState架构进行重构。
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        num_position_steps: int,
        state_condition: bool = True,
        rollout_freq: int = 50,
        out_channels: int = 20,
    ):
        super().__init__(network, out_channels)
        self.decoder = decoder
        self.num_position_steps = num_position_steps
        self.state_condition = state_condition
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        """完整的姿态预测流程"""
        features = self.network(emg)  # BCT

        # 将特征重新采样到展开频率
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # 重置LSTM隐藏状态
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        # 计算新采样率下的num_position_steps
        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        preds = [initial_pos]

        for t in range(features.shape[-1]):

            # 准备解码器输入
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # 预测姿态和速度
            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)

            # 前num_position_steps预测姿态，之后积分速度
            pred = pos if t < num_position_steps else preds[-1] + vel
            preds.append(pred)

        # 移除第一个预测，因为它是initial_pos（不是网络预测）
        return torch.stack(preds[1:], dim=-1)

    def forward_incremental(
        self, 
        emg_chunk: torch.Tensor, 
        cached_features: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        step_count: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        增量前向传播，用于实时预测
        
        Args:
            emg_chunk: 新的EMG数据块，形状为(B, C, T_chunk)
            cached_features: 缓存的特征，形状为(B, C_feat, T_cached)
            hidden_state: LSTM隐藏状态
            step_count: 当前步数计数
            
        Returns:
            prediction: 当前步的预测，形状为(B, out_channels)
            new_features: 更新后的特征缓存
            new_hidden_state: 更新后的LSTM隐藏状态
        """
        # 提取新特征
        new_features = self.network(emg_chunk)  # BCT
        
        # 更新特征缓存
        if cached_features is not None:
            # 拼接新特征到缓存中
            updated_features = torch.cat([cached_features, new_features], dim=-1)
        else:
            updated_features = new_features
        
        # 获取最新的特征进行预测
        current_feature = updated_features[:, :, -1]  # B, C_feat
        
        # 准备解码器输入（需要上一步的预测作为状态条件）
        if self.state_condition:
            # 如果使用状态条件，始终需要拼接上一步的预测
            # 第一步时使用零填充，后续步骤应该从外部传入真实的上一步预测
            last_pred = torch.zeros(emg_chunk.size(0), self.out_channels // 2, device=emg_chunk.device)
            inputs = torch.cat([current_feature, last_pred], dim=-1)
        else:
            inputs = current_feature
        
        # 设置LSTM隐藏状态
        if isinstance(self.decoder, SequentialLSTM) and hidden_state is not None:
            self.decoder.hidden = hidden_state
        
        # 预测
        output = self.decoder(inputs)
        pos, vel = torch.split(output, output.shape[1] // 2, dim=1)
        
        # 判断使用位置预测还是速度积分
        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        
        if step_count < num_position_steps:
            prediction = pos
        else:
            # 需要上一步的预测来积分速度，这里暂时使用位置预测
            prediction = pos  # 实际应用中需要传入上一步预测
        
        # 获取新的隐藏状态
        new_hidden_state = self.decoder.hidden if isinstance(self.decoder, SequentialLSTM) else None
        
        return prediction, updated_features, new_hidden_state

    def forward_correction(self, emg_window: torch.Tensor) -> torch.Tensor:
        """
        完整窗口前向传播，用于校正预测
        
        Args:
            emg_window: 完整的EMG窗口，形状为(B, C, T_window)
            
        Returns:
            predictions: 完整窗口的预测，形状为(B, out_channels, T_out)
        """
        # 使用完整的前向传播流程
        # 只使用位置部分（前out_channels//2维）与forward_incremental保持一致
        initial_pos = torch.zeros(emg_window.size(0), self.out_channels // 2, device=emg_window.device)
        return self._predict_pose(emg_window, initial_pos)
