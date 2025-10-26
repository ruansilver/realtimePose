"""TensorBoard日志系统"""

from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard日志记录器
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称，None则使用时间戳
    """
    
    def __init__(self, log_dir: str = './logs', experiment_name: str | None = None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step = 0
        
        print(f"TensorBoard日志目录: {self.log_dir}")
        print(f"启动TensorBoard: tensorboard --logdir {log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int | None = None):
        """记录标量
        
        Args:
            tag: 标签名
            value: 标量值
            step: 步数，None则使用内部计数器
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int | None = None):
        """记录多个标量
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-值字典
            step: 步数
        """
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int | None = None):
        """记录直方图
        
        Args:
            tag: 标签名
            values: 张量值
            step: 步数
        """
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_graph(self, model: nn.Module, input_sample: torch.Tensor):
        """记录模型结构图
        
        Args:
            model: 模型
            input_sample: 输入样本（用于追踪）
        """
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"警告: 无法记录模型图，错误: {e}")
    
    def log_text(self, tag: str, text: str, step: int | None = None):
        """记录文本
        
        Args:
            tag: 标签名
            text: 文本内容
            step: 步数
        """
        if step is None:
            step = self.step
        self.writer.add_text(tag, text, step)
    
    def increment_step(self):
        """增加步数计数器"""
        self.step += 1
    
    def close(self):
        """关闭writer"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

