"""实时预测器：流式窗口推理"""

import torch
from torch import nn

from src.core.buffer_manager import InputBuffer


class RealtimePredictor:
    """实时预测器：执行单窗口推理
    
    Args:
        model: 姿态预测模型
        config: 配置字典
        device: 计算设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.provide_initial_pos = config['model'].get('provide_initial_pos', False)
    
    @torch.no_grad()
    def predict_window(
        self,
        emg_window: torch.Tensor,
        joint_angles: torch.Tensor = None,
        no_ik_failure: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """执行单窗口推理
        
        Args:
            emg_window: EMG窗口 (C, window_length) 或 (B, C, window_length)
            joint_angles: 可选的关节角度标签 (C, window_length)
            no_ik_failure: 可选的IK失败掩码 (window_length,)
            
        Returns:
            包含预测结果的字典:
                - pred: 预测姿态 (B, C, T_valid)
                - joint_angles: 有效区间标签 (B, C, T_valid) [如果提供]
                - no_ik_failure: 有效区间掩码 (B, T_valid) [如果提供]
        """
        # 添加batch维度（如果需要）
        if emg_window.ndim == 2:
            emg_window = emg_window.unsqueeze(0)
        
        # 移动到设备
        emg_window = emg_window.to(self.device)
        
        # 准备批次数据
        batch = {"emg": emg_window}
        
        if joint_angles is not None:
            if joint_angles.ndim == 2:
                joint_angles = joint_angles.unsqueeze(0)
            batch["joint_angles"] = joint_angles.to(self.device)
        else:
            # 创建零标签（用于前向传播）
            batch["joint_angles"] = torch.zeros_like(emg_window[:, :20, :])
        
        if no_ik_failure is not None:
            if no_ik_failure.ndim == 1:
                no_ik_failure = no_ik_failure.unsqueeze(0)
            batch["no_ik_failure"] = no_ik_failure.to(self.device)
        else:
            # 创建全True掩码
            batch["no_ik_failure"] = torch.ones(
                emg_window.shape[0], emg_window.shape[2],
                dtype=torch.bool, device=self.device
            )
        
        # 前向传播
        pred, joint_angles_valid, no_ik_failure_valid = self.model(
            batch, self.provide_initial_pos
        )
        
        result = {
            "pred": pred,
        }
        
        if joint_angles is not None:
            result["joint_angles"] = joint_angles_valid
        
        if no_ik_failure is not None:
            result["no_ik_failure"] = no_ik_failure_valid
        
        return result
    
    def process_stream(self, buffer: InputBuffer) -> torch.Tensor | None:
        """从缓冲区处理流式数据
        
        Args:
            buffer: 输入缓冲区
            
        Returns:
            预测结果 (C, T_valid) 或 None（如果缓冲区未满）
        """
        if not buffer.has_full_window():
            return None
        
        # 获取窗口
        window = buffer.get_window()
        
        # 推理
        result = self.predict_window(window)
        
        # 返回有效区间预测（移除batch维度）
        return result["pred"].squeeze(0)

