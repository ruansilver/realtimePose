"""流式预测器：端到端流式推理"""

import torch
from torch import nn
import time
from typing import Dict, List, Optional, Tuple

from src.core.ring_buffer import RingBuffer
from src.core.incremental_resampler import IncrementalResampler
from src.models.tds_streaming import TdsStreamingWrapper
from src.models.decoder import SequentialLSTM


class StreamingPredictor:
    """端到端流式预测器
    
    整合环形缓冲器、流式编码器、增量重采样器和LSTM解码器，
    实现每20ms连续输出的流式推理。
    
    Args:
        model: 完整的姿态预测模型（用于获取组件）
        config: 配置字典
        device: 计算设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device
    ):
        self.config = config
        self.device = device
        
        # 从完整模型中提取组件
        self.encoder = model.network  # TdsNetwork (在VEMG2PoseWithInitialState中存储为network)
        self.decoder = model.decoder  # SequentialLSTM
        self.head = model  # 完整的head（包含rollout逻辑）
        
        # 配置参数
        self.sample_rate = config['constants']['emg_sample_rate']  # 2000Hz
        self.step_ms = config['runtime']['streaming']['step_ms']  # 20ms
        self.step_samples = int(self.sample_rate * self.step_ms / 1000)  # 40 samples
        self.warmup_samples = config['runtime']['streaming']['warmup_samples']  # 1790
        self.rollout_freq = config['model']['rollout_freq']  # 50Hz
        self.num_position_steps = config['model']['num_position_steps']  # 500
        self.state_condition = config['model']['state_condition']  # True
        
        # 滑动窗口方案配置
        self.window_samples_base = config['runtime']['streaming']['window_samples_base']  # 1830
        self.window_thicken_factor = config['runtime']['streaming']['window_thicken_factor']  # 增厚因子
        # 增厚后的实际编码器窗口大小
        self.window_samples_thickened = self.window_samples_base + self.window_thicken_factor * self.step_samples
        
        # 创建流式组件
        self.ring_buffer = RingBuffer(
            max_samples=config['runtime']['streaming']['max_buffer_seconds'] * self.sample_rate,
            num_channels=config['constants']['num_emg_channels'],
            sample_rate=self.sample_rate,
            step_ms=self.step_ms
        )
        
        # 状态管理
        self.is_warmed_up = False
        self.step_count = 0
        self.last_position = None  # 上一步的关节位置
        
        # 计算rollout切换点（只计算一次，与exact模式一致）
        self.rollout_position_steps = round(self.num_position_steps * (self.rollout_freq / 2000))
        print(f"DEBUG[S3]: rollout_position_steps={self.rollout_position_steps} (num_position_steps={self.num_position_steps}, rollout_freq={self.rollout_freq})")
        print(f"DEBUG[S3]: window_thickening - base={self.window_samples_base}, factor={self.window_thicken_factor}, thickened={self.window_samples_thickened}")
        
        # 简化增厚窗口策略：不需要额外的采样状态变量
        
        # 性能监控
        self.latency_stats = []
        
    def reset(self):
        """重置所有流式状态"""
        self.ring_buffer.reset()
        self.decoder.reset_state()
        
        self.is_warmed_up = False
        self.step_count = 0
        self.last_position = None
        self.latency_stats.clear()
    
    def warmup(self, warmup_data: torch.Tensor):
        """预热流式预测器
        
        Args:
            warmup_data: 预热EMG数据 (B, C, warmup_samples)
        """
        print(f"开始预热流式预测器 (样本数: {warmup_data.shape[2]})...")
        
        # 简化实现：将预热数据添加到环形缓冲器
        for t in range(warmup_data.shape[2]):
            self.ring_buffer.append(warmup_data[0, :, t])  # 移除batch维度
        
        self.is_warmed_up = True
        print("预热完成")
    
    def push_samples(self, samples: torch.Tensor) -> List[torch.Tensor]:
        """推入新的EMG样本并尝试产生预测
        
        Args:
            samples: EMG样本 (C, T) 或 (T, C)
            
        Returns:
            预测结果列表，每个元素为 (num_joints,)
        """
        if not self.is_warmed_up:
            raise RuntimeError("预测器未预热，请先调用warmup()")
        
        # 确保格式为 (T, C)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[0] == self.config['constants']['num_emg_channels']:
            samples = samples.T  # (C, T) -> (T, C)
        
        predictions = []
        
        # 逐样本添加到缓冲器
        for t in range(samples.shape[0]):
            self.ring_buffer.append(samples[t])
            
            # 检查是否可以进行推理步骤
            if self.ring_buffer.can_step():
                prediction = self._inference_step()
                if prediction is not None:
                    predictions.append(prediction)
        
        return predictions
    
    def _inference_step(self) -> Optional[torch.Tensor]:
        """执行单步推理
        
        Returns:
            关节角度预测 (num_joints,) 或 None
        """
        # GPU同步计时（按plan2.md指令）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        try:
            # 滑动窗口方案：按照 plan3.md 的窗口增厚策略
            # 1. 检查是否有足够数据进行增厚窗口推理
            if self.ring_buffer.get_current_samples() < self.window_samples_thickened:
                return None
                
            # 2. 从尾部提取增厚长度窗口 (1830+n*40 样本)
            window_data = self.ring_buffer.slice_for_context(self.window_samples_thickened)
            window_data = window_data.T.unsqueeze(0).to(self.device)  # (1, C, L)
            
            # 3. 完整前向传播 TdsNetwork
            with torch.no_grad():
                features = self.encoder(window_data)  # (1, C, T_encoded)
            
            # 3.1 增厚窗口策略：不做特征裁剪，直接用增厚窗口编码结果
            
            # 4. 单点线性插值到50Hz（n_time=1，align_corners=true）
            if features.shape[2] == 0:
                return None
            
            # 4. 单点线性插值：左缘取样，align_corners=true
            # 按照plan3.md指令：seconds基于base窗口（1830对齐），不随增厚变化
            seconds = (self.window_samples_base - self.encoder.left_context - self.encoder.right_context) / 2000
            
            # 执行单点线性插值（左缘取样）
            from torch.nn.functional import interpolate
            
            # 单点插值：n_time=1，align_corners=True（左缘取样）
            interpolated = interpolate(
                features, 
                size=1, 
                mode='linear', 
                align_corners=True
            )  # (1, C, 1)
            if self.step_count == 0:
                print("DEBUG[S2]: using_feature_interpolation=True, mode=linear, size=1, align_corners=True")
            
            # 提取单点特征
            resampled_feature = interpolated[0, :, 0]  # (feature_dim,)
            
            # 5. 准备LSTM单步输入
            lstm_input = resampled_feature  # (64,)
            
            if self.state_condition and self.last_position is not None:
                # 拼接上一步的关节状态
                lstm_input = torch.cat([lstm_input, self.last_position], dim=0)  # (84,)
            else:
                # 填充零状态
                zero_state = torch.zeros(20, device=self.device)
                lstm_input = torch.cat([lstm_input, zero_state], dim=0)  # (84,)
            
            lstm_input = lstm_input.unsqueeze(0)  # (1, 84)
            
            # 6. LSTM单步解码（保持跨步状态）
            lstm_output = self.decoder(lstm_input)  # (1, 40)
            lstm_output = lstm_output.squeeze(0)  # (40,)
            
            # 7. 分离位置和速度
            position_output = lstm_output[:20]  # (20,)
            velocity_output = lstm_output[20:]  # (20,)
            
            # 8. 根据步数决定使用位置还是速度积分
            # 使用预先计算的rollout_position_steps，确保与exact模式一致
            if self.step_count < self.rollout_position_steps:
                # 位置模式：前几步直接使用位置输出
                current_position = position_output
            else:
                # 速度积分模式：后续步骤使用速度积分
                if self.last_position is not None:
                    current_position = self.last_position + velocity_output
                else:
                    current_position = velocity_output
            
            # 9. 更新状态
            self.last_position = current_position.clone()
            self.step_count += 1
            self.ring_buffer.mark_step_completed()
            
            # 记录延迟（GPU同步计时）
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            # T_compute = 编码器前向 + 单点插值 + LSTM单步
            latency_ms = (end_time - start_time) * 1000
            self.latency_stats.append(latency_ms)
            
            return current_position
            
        except Exception as e:
            print(f"推理步骤失败: {e}")
            return None
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计信息
        
        Returns:
            延迟统计字典 (P50, P95, P99, mean, std)
        """
        if not self.latency_stats:
            return {}
        
        import numpy as np
        latencies = np.array(self.latency_stats)
        
        return {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'count': len(latencies)
        }
    
    def clear_latency_stats(self):
        """清空延迟统计"""
        self.latency_stats.clear()
    
    def get_status(self) -> Dict:
        """获取预测器状态信息
        
        Returns:
            状态信息字典
        """
        return {
            'is_warmed_up': self.is_warmed_up,
            'step_count': self.step_count,
            'window_samples_base': self.window_samples_base,
            'window_thicken_factor': self.window_thicken_factor,
            'window_samples_thickened': self.window_samples_thickened,
            'buffer_samples': self.ring_buffer.get_current_samples(),
            'samples_since_last_step': self.ring_buffer.get_samples_since_last_step(),
            'latency_stats_count': len(self.latency_stats)
        }
