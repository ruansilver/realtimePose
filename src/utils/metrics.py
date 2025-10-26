"""评估指标：角度误差计算"""

import torch


def compute_angle_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """计算角度平均绝对误差（MAE）
    
    Args:
        pred: 预测角度 (B, C, T)
        target: 目标角度 (B, C, T)
        mask: 有效掩码 (B, T)，True表示有效样本
        
    Returns:
        标量MAE损失
    """
    if mask is not None:
        # 扩展掩码到所有关节
        mask_expanded = mask.unsqueeze(1).expand_as(pred)  # (B, C, T)
        return torch.nn.L1Loss()(pred[mask_expanded], target[mask_expanded])
    else:
        return torch.nn.L1Loss()(pred, target)


def compute_angle_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """计算角度均方误差（MSE）
    
    Args:
        pred: 预测角度 (B, C, T)
        target: 目标角度 (B, C, T)
        mask: 有效掩码 (B, T)，True表示有效样本
        
    Returns:
        标量MSE损失
    """
    if mask is not None:
        # 扩展掩码到所有关节
        mask_expanded = mask.unsqueeze(1).expand_as(pred)  # (B, C, T)
        return torch.nn.MSELoss()(pred[mask_expanded], target[mask_expanded])
    else:
        return torch.nn.MSELoss()(pred, target)


def compute_max_absolute_diff(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """计算最大绝对差异
    
    Args:
        pred: 预测角度 (B, C, T)
        target: 目标角度 (B, C, T)
        mask: 有效掩码 (B, T)，True表示有效样本
        
    Returns:
        标量最大差异
    """
    diff = torch.abs(pred - target)
    
    if mask is not None:
        mask_expanded = mask.unsqueeze(1).expand_as(pred)
        diff = diff[mask_expanded]
    
    return torch.max(diff)


def evaluate_predictions(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None
) -> dict[str, float]:
    """评估预测结果
    
    Args:
        pred: 预测角度 (B, C, T)
        target: 目标角度 (B, C, T)
        mask: 有效掩码 (B, T)
        
    Returns:
        指标字典
    """
    metrics = {
        'mae': compute_angle_mae(pred, target, mask).item(),
        'mse': compute_angle_mse(pred, target, mask).item(),
        'max_diff': compute_max_absolute_diff(pred, target, mask).item(),
    }
    
    return metrics

