# Copyright (c) Realtime Pose Prediction System
# All rights reserved.

"""
检查点管理器

负责模型检查点的保存、加载和管理。
"""

import os
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    检查点管理器
    
    提供模型检查点的保存、加载、清理等功能。
    """
    
    def __init__(self, save_dir: str, keep_last_n: int = 5):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 检查点保存目录
            keep_last_n: 保留最近N个检查点
        """
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"检查点管理器初始化完成")
        logger.info(f"保存目录: {self.save_dir}")
        logger.info(f"保留检查点数量: {self.keep_last_n}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        保存模型检查点
        
        Args:
            model: 要保存的模型
            optimizer: 优化器状态（可选）
            scheduler: 学习率调度器状态（可选）
            metadata: 额外的元数据（可选）
            checkpoint_name: 检查点名称（可选，默认使用时间戳）
            
        Returns:
            保存的检查点文件路径
        """
        try:
            # 生成检查点名称
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{timestamp}.ckpt"
            
            checkpoint_path = self.save_dir / checkpoint_name
            
            # 准备保存的数据
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__
            }
            
            # 添加优化器状态
            if optimizer is not None:
                checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint_data["optimizer_type"] = type(optimizer).__name__
            
            # 添加调度器状态
            if scheduler is not None:
                checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
                checkpoint_data["scheduler_type"] = type(scheduler).__name__
            
            # 添加元数据
            if metadata is not None:
                checkpoint_data["metadata"] = metadata
            
            # 添加模型结构信息
            checkpoint_data["model_info"] = self._get_model_info(model)
            
            # 保存检查点
            torch.save(checkpoint_data, checkpoint_path)
            
            # 保存元数据文件（便于快速查看）
            metadata_path = checkpoint_path.with_suffix('.json')
            self._save_metadata(metadata_path, checkpoint_data)
            
            logger.info(f"检查点已保存: {checkpoint_path}")
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            raise
    
    def load_checkpoint(
        self, 
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            model: 要加载状态的模型（可选）
            optimizer: 要加载状态的优化器（可选）
            scheduler: 要加载状态的调度器（可选）
            strict: 是否严格匹配状态字典键名
            
        Returns:
            检查点数据字典
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
            
            # 加载检查点数据，设置weights_only=False以兼容PyTorch 2.6
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # 加载模型状态
            if model is not None and "model_state_dict" in checkpoint_data:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data["model_state_dict"], strict=strict
                )
                
                if missing_keys:
                    logger.warning(f"模型中缺失的键: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"检查点中多余的键: {unexpected_keys}")
            
            # 加载优化器状态
            if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            # 加载调度器状态
            if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            
            logger.info(f"检查点加载成功: {checkpoint_path}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
    
    def load_latest_checkpoint(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pattern: str = "checkpoint_*.ckpt"
    ) -> Optional[Dict[str, Any]]:
        """
        加载最新的检查点
        
        Args:
            model: 要加载状态的模型（可选）
            optimizer: 要加载状态的优化器（可选）
            scheduler: 要加载状态的调度器（可选）
            pattern: 检查点文件匹配模式
            
        Returns:
            检查点数据字典或None（如果没有找到）
        """
        try:
            # 查找所有检查点文件
            checkpoint_files = list(self.save_dir.glob(pattern))
            
            if not checkpoint_files:
                logger.warning(f"未找到匹配的检查点文件: {pattern}")
                return None
            
            # 按修改时间排序，获取最新的
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            logger.info(f"找到最新检查点: {latest_checkpoint}")
            
            return self.load_checkpoint(
                str(latest_checkpoint), model, optimizer, scheduler
            )
            
        except Exception as e:
            logger.error(f"加载最新检查点失败: {e}")
            return None
    
    def list_checkpoints(self, pattern: str = "checkpoint_*.ckpt") -> List[Dict[str, Any]]:
        """
        列出所有检查点
        
        Args:
            pattern: 检查点文件匹配模式
            
        Returns:
            检查点信息列表
        """
        checkpoints = []
        
        try:
            checkpoint_files = list(self.save_dir.glob(pattern))
            
            for checkpoint_file in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
                # 尝试读取元数据文件
                metadata_file = checkpoint_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"读取元数据文件失败 {metadata_file}: {e}")
                
                # 获取文件信息
                stat = checkpoint_file.stat()
                
                checkpoint_info = {
                    "name": checkpoint_file.name,
                    "path": str(checkpoint_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "metadata": metadata
                }
                
                checkpoints.append(checkpoint_info)
                
        except Exception as e:
            logger.error(f"列出检查点失败: {e}")
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        删除指定的检查点
        
        Args:
            checkpoint_name: 检查点文件名
            
        Returns:
            是否删除成功
        """
        try:
            checkpoint_path = self.save_dir / checkpoint_name
            metadata_path = checkpoint_path.with_suffix('.json')
            
            # 删除检查点文件
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"已删除检查点: {checkpoint_path}")
            
            # 删除元数据文件
            if metadata_path.exists():
                metadata_path.unlink()
                logger.debug(f"已删除元数据文件: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
            return False
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的检查点文件"""
        try:
            checkpoint_files = list(self.save_dir.glob("checkpoint_*.ckpt"))
            
            if len(checkpoint_files) <= self.keep_last_n:
                return
            
            # 按修改时间排序
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除多余的检查点
            files_to_delete = checkpoint_files[self.keep_last_n:]
            
            for file_path in files_to_delete:
                # 删除检查点文件
                file_path.unlink()
                
                # 删除对应的元数据文件
                metadata_path = file_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"已删除旧检查点: {file_path.name}")
                
        except Exception as e:
            logger.error(f"清理旧检查点失败: {e}")
    
    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(total_params * 4 / (1024 * 1024), 2)  # 假设float32
        }
    
    def _save_metadata(self, metadata_path: Path, checkpoint_data: Dict[str, Any]) -> None:
        """保存元数据到JSON文件"""
        try:
            # 提取可序列化的元数据
            serializable_metadata = {
                "timestamp": checkpoint_data.get("timestamp"),
                "model_type": checkpoint_data.get("model_type"),
                "optimizer_type": checkpoint_data.get("optimizer_type"),
                "scheduler_type": checkpoint_data.get("scheduler_type"),
                "model_info": checkpoint_data.get("model_info", {}),
                "metadata": checkpoint_data.get("metadata", {})
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"保存元数据文件失败: {e}")
    
    def create_backup(self, backup_dir: str) -> bool:
        """
        创建检查点目录的备份
        
        Args:
            backup_dir: 备份目录路径
            
        Returns:
            是否备份成功
        """
        try:
            backup_path = Path(backup_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"checkpoint_backup_{timestamp}"
            full_backup_path = backup_path / backup_name
            
            # 创建备份目录
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 复制整个检查点目录
            shutil.copytree(self.save_dir, full_backup_path)
            
            logger.info(f"检查点备份创建成功: {full_backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建检查点备份失败: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        try:
            checkpoint_files = list(self.save_dir.glob("checkpoint_*.ckpt"))
            total_size = sum(f.stat().st_size for f in checkpoint_files)
            
            return {
                "checkpoint_count": len(checkpoint_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "save_dir": str(self.save_dir),
                "keep_last_n": self.keep_last_n,
                "available_space_gb": round(shutil.disk_usage(self.save_dir)[2] / (1024**3), 2)
            }
            
        except Exception as e:
            logger.error(f"获取存储信息失败: {e}")
            return {}
