"""主程序：端到端EMG姿态预测系统"""

import sys
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_manager import parse_args, get_config
from src.utils.logger import TensorBoardLogger
from src.utils.metrics import evaluate_predictions
from src.utils.checkpoint import create_reference_model_from_checkpoint
from src.data.dataset import create_dataloader
from src.models.encoder import build_tds_network_from_config
from src.models.decoder import build_decoder_from_config
from src.models.pose_head import build_pose_head_from_config
from src.core.buffer_manager import InputBuffer
from src.core.realtime_predictor import RealtimePredictor


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: dict) -> nn.Module:
    """根据配置构建完整模型
    
    Args:
        config: 配置字典
        
    Returns:
        姿态预测模型
    """
    model_config = config['model']
    
    # 构建模型组件
    encoder = build_tds_network_from_config(model_config['network'])
    decoder = build_decoder_from_config(model_config['decoder'])
    model = build_pose_head_from_config(model_config, encoder, decoder)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    return model


def prepare_data(config: dict) -> tuple:
    """准备数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        (train_loader, val_loader, test_loader) 元组（可能为None）
    """
    data_config = config['data']
    dataset_root = Path(data_config['dataset_root'])
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    
    # 获取所有HDF5文件
    hdf5_files = list(dataset_root.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"在 {dataset_root} 中未找到HDF5文件")
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 简单划分：80% train, 10% val, 10% test
    n_files = len(hdf5_files)
    n_train = int(0.8 * n_files)
    n_val = int(0.1 * n_files)
    
    train_files = hdf5_files[:n_train] if n_train > 0 else []
    val_files = hdf5_files[n_train:n_train+n_val] if n_val > 0 else hdf5_files[:1]
    test_files = hdf5_files[n_train+n_val:] if n_train+n_val < n_files else val_files
    
    # 创建DataLoader
    train_loader = None
    if config.get('train', {}).get('enabled', False) and train_files:
        train_loader = create_dataloader(
            hdf5_files=train_files,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            window_length=data_config['window_length'],
            padding=tuple(data_config['padding']),
            jitter=True,
            skip_ik_failures=data_config.get('skip_ik_failures', False),
            shuffle=True,
        )
    
    val_loader = None
    if val_files:
        val_loader = create_dataloader(
            hdf5_files=val_files,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            window_length=data_config.get('val_test_window_length', data_config['window_length']),
            padding=tuple(data_config['padding']),
            jitter=False,
            skip_ik_failures=data_config.get('skip_ik_failures', False),
            shuffle=False,
        )
    
    test_loader = None
    if test_files:
        test_loader = create_dataloader(
            hdf5_files=test_files,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            window_length=data_config.get('val_test_window_length', data_config['window_length']),
            padding=tuple(data_config['padding']),
            jitter=False,
            skip_ik_failures=data_config.get('skip_ik_failures', False),
            shuffle=False,
        )
    
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    logger: TensorBoardLogger,
    device: torch.device
):
    """训练模型（可选）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    train_config = config['train']
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate']
    )
    
    # 损失权重
    loss_weights = train_config.get('loss_weights', {'mae': 1.0})
    
    max_epochs = train_config.get('max_epochs', 100)
    best_val_loss = float('inf')
    patience = train_config.get('early_stopping_patience', 10)
    patience_counter = 0
    
    print(f"开始训练 ({max_epochs} epochs)...")
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            pred, target, mask = model(batch, config['model'].get('provide_initial_pos', False))
            
            # 计算损失
            from src.utils.metrics import compute_angle_mae
            loss = compute_angle_mae(pred, target, mask) * loss_weights.get('mae', 1.0)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        # 记录训练损失
        avg_train_loss = np.mean(train_losses)
        logger.log_scalar('train/loss', avg_train_loss, epoch)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    pred, target, mask = model(batch, config['model'].get('provide_initial_pos', False))
                    from src.utils.metrics import compute_angle_mae
                    loss = compute_angle_mae(pred, target, mask)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            logger.log_scalar('val/loss', avg_val_loss, epoch)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), logger.log_dir / 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    print("训练完成!")


def update_ik_failure_mask(
    no_ik_failure: torch.Tensor, 
    model: nn.Module, 
    provide_initial_pos: bool
) -> torch.Tensor:
    """更新IK失败掩码（参考lightning.py第194-208行）"""
    mask = no_ik_failure.clone()
    
    if provide_initial_pos:
        mask[~mask[:, model.left_context]] = False
    
    return mask


def alignment_check(
    model: nn.Module,
    reference_model: nn.Module,
    val_loader,
    device: torch.device,
    config: dict
) -> dict:
    """数值对齐验证
    
    Args:
        model: 当前模型
        reference_model: 参考模型
        val_loader: 验证数据加载器
        device: 设备
        config: 配置字典
        
    Returns:
        对齐结果字典
    """
    print("开始数值对齐验证...")
    
    # 确保模型处于评估模式
    model.eval()
    reference_model.eval()
    
    # 确保没有随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 取第一个batch进行对齐检查
    batch = next(iter(val_loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v 
            for k, v in batch.items()}
    
    provide_initial_pos = config['model'].get('provide_initial_pos', False)
    
    with torch.no_grad():
        original_mask = batch["no_ik_failure"].clone()
        
        # 当前模型预测
        batch["no_ik_failure"] = update_ik_failure_mask(
            original_mask.clone(), model, provide_initial_pos
        )
        pred, target, mask = model(batch, provide_initial_pos)
        
        # 参考模型预测
        batch["no_ik_failure"] = update_ik_failure_mask(
            original_mask.clone(), reference_model, provide_initial_pos
        )
        ref_pred, ref_target, ref_mask = reference_model(batch, provide_initial_pos)
    
    # 计算差异
    mask_expanded = mask.unsqueeze(1).expand_as(pred)
    ref_mask_expanded = ref_mask.unsqueeze(1).expand_as(ref_pred)
    
    mse = torch.nn.MSELoss()(pred[mask_expanded], ref_pred[ref_mask_expanded]).item()
    max_diff = torch.max(torch.abs(pred[mask_expanded] - ref_pred[ref_mask_expanded])).item()
    
    # 验收阈值
    mse_threshold = 1e-6
    max_diff_threshold = 1e-5
    passed = mse < mse_threshold and max_diff < max_diff_threshold
    
    print(f"对齐验证结果:")
    print(f"  MSE: {mse:.2e} (阈值: {mse_threshold:.2e}) {'✓' if mse < mse_threshold else '✗'}")
    print(f"  Max Diff: {max_diff:.2e} (阈值: {max_diff_threshold:.2e}) {'✓' if max_diff < max_diff_threshold else '✗'}")
    print(f"  总体结果: {'通过 ✓' if passed else '失败 ✗'}")
    
    return {
        'mse': mse,
        'max_diff': max_diff,
        'passed': passed,
        'pred_shape': pred.shape,
        'valid_samples': mask.sum().item()
    }


def realtime_simulation(
    model: nn.Module,
    test_loader,
    config: dict,
    logger: TensorBoardLogger,
    device: torch.device
):
    """实时模拟推理
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        config: 配置字典
        logger: 日志记录器
        device: 设备
    """
    print("开始实时模拟推理...")
    
    # 创建实时预测器
    predictor = RealtimePredictor(model, config, device)
    buffer = InputBuffer(max_samples=config['runtime']['buffer']['max_samples'])
    
    all_metrics = []
    window_count = 0
    
    # 从测试集模拟流式输入
    for batch_idx, batch in enumerate(test_loader):
        emg = batch['emg']
        joint_angles = batch['joint_angles']
        no_ik_failure = batch['no_ik_failure']
        
        for sample_idx in range(emg.shape[0]):
            buffer.append(emg[sample_idx])
            
            if buffer.has_full_window():
                window_count += 1
                
                result = predictor.predict_window(
                    buffer.get_window(),
                    joint_angles[sample_idx],
                    no_ik_failure[sample_idx]
                )
                
                metrics = evaluate_predictions(
                    result['pred'],
                    result['joint_angles'],
                    result['no_ik_failure']
                )
                all_metrics.append(metrics)
                
                # 记录到TensorBoard
                logger.log_scalar('realtime/mae', metrics['mae'], window_count)
                logger.log_scalar('realtime/mse', metrics['mse'], window_count)
                logger.log_scalar('realtime/max_diff', metrics['max_diff'], window_count)
        
        # 限制处理的batch数
        if batch_idx >= 10:
            break
    
    # 汇总统计
    if all_metrics:
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        print(f"实时推理完成 (共 {window_count} 窗口)")
        print(f"平均MAE: {avg_metrics['mae']:.6f}")
        print(f"平均MSE: {avg_metrics['mse']:.6f}")


def main():
    """主函数"""
    print("="*60)
    print("实时EMG姿态预测系统")
    print("="*60)
    
    # 加载配置
    args = parse_args()
    config = get_config(args.config, args.overrides)
    print(f"配置文件: {args.config}")
    
    # 设置随机种子和设备
    seed = config['runtime'].get('seed', 42)
    set_seed(seed)
    
    device_name = config['runtime'].get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 初始化日志
    if config['logging'].get('tensorboard', {}).get('enabled', True):
        logger = TensorBoardLogger(config['logging']['log_dir'])
    else:
        logger = None
    
    # 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # 构建模型
    print("构建模型...")
    model = build_model(config).to(device)
    
    # 训练（可选）
    if config.get('train', {}).get('enabled', False):
        if train_loader is None:
            print("警告: train.enabled=true 但未找到训练数据")
        else:
            train_model(model, train_loader, val_loader, config, logger, device)
    else:
        print("跳过训练 (train.enabled=false)")
    
    # 数值对齐验证
    if config.get('eval', {}).get('alignment_check', True):
        ref_ckpt_path = config.get('reference_checkpoint')
        if ref_ckpt_path and Path(ref_ckpt_path).exists() and val_loader is not None:
            print("加载参考checkpoint进行对齐验证...")
            
            # 当前模型加载checkpoint权重
            model = create_reference_model_from_checkpoint(
                ref_ckpt_path, model, device
            )
            
            # 创建参考模型（从相同checkpoint加载）
            reference_model = build_model(config)
            reference_model = create_reference_model_from_checkpoint(
                ref_ckpt_path, reference_model, device
            )
            
            alignment_result = alignment_check(
                model, reference_model, val_loader, device, config
            )
            
            if logger:
                logger.log_text('alignment/result', str(alignment_result))
        else:
            if not ref_ckpt_path:
                print("跳过对齐验证: 未配置 reference_checkpoint")
            elif not Path(ref_ckpt_path).exists():
                print(f"跳过对齐验证: checkpoint不存在 {ref_ckpt_path}")
            else:
                print("跳过对齐验证: 无验证数据")
    
    # 实时模拟推理
    if test_loader is not None and logger:
        realtime_simulation(model, test_loader, config, logger, device)
    else:
        print("跳过实时模拟: 无测试数据或日志未启用")
    
    # 清理
    if logger:
        logger.close()
        print(f"日志已保存到: {logger.log_dir}")
    
    print("="*60)
    print("运行完成!")
    print("="*60)


if __name__ == "__main__":
    main()

