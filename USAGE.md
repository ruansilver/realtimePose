# 使用说明

## 快速开始

### 1. 环境准备

确保已激活正确的 conda 环境：

```bash
conda activate D:\env\condaEnv\emg2Pose
```

或安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置系统

打开 `configs/realtime_config.yaml` 查看和修改配置：

```yaml
# 关键配置项
data:
  dataset_root: D:/Dataset/emg2pose_dataset_mini  # 数据集路径
  window_length: 10000  # 窗口长度（5秒@2kHz）
  
runtime:
  device: cuda  # 使用GPU或CPU
  
eval:
  alignment_check: true  # 是否进行数值对齐验证
  
reference_checkpoint: refer/referCheckpoint/regression_vemg2pose.ckpt
```

### 3. 运行系统

#### 基础推理（默认）

```bash
python main.py
```

这将：
- 加载配置和数据
- 构建模型
- 进行数值对齐验证（如果启用）
- 执行实时模拟推理
- 记录日志到 TensorBoard

#### 命令行覆写配置

```bash
# 更改设备
python main.py --set runtime.device=cpu

# 更改batch size
python main.py --set data.batch_size=32

# 启用训练
python main.py --set train.enabled=true

# 组合多个覆写
python main.py --set runtime.device=cuda --set data.batch_size=16 --set train.enabled=true
```

#### 使用自定义配置文件

```bash
python main.py --config configs/my_config.yaml
```

### 4. 查看结果

启动 TensorBoard 查看实时曲线：

```bash
tensorboard --logdir ./logs
```

然后在浏览器打开 http://localhost:6006

## 系统架构

```
cursorRealTimePose/
├── configs/
│   └── realtime_config.yaml        # 主配置文件
├── src/
│   ├── data/
│   │   └── dataset.py              # HDF5数据加载
│   ├── models/
│   │   ├── encoder.py              # TDS编码器
│   │   ├── decoder.py              # LSTM解码器
│   │   └── pose_head.py            # 姿态预测头
│   ├── core/
│   │   ├── buffer_manager.py       # 流式缓冲管理
│   │   └── realtime_predictor.py   # 实时推理器
│   └── utils/
│       ├── config_manager.py       # 配置管理
│       ├── logger.py               # TensorBoard日志
│       ├── metrics.py              # 评估指标
│       └── checkpoint.py           # checkpoint加载
├── main.py                          # 主程序入口
├── logs/                            # 日志输出目录
└── refer/                           # 参考代码（不修改）
```

## 工作流程

### 推理模式（默认）

1. **初始化**: 加载配置、设置随机种子、初始化设备
2. **数据准备**: 从HDF5文件加载数据，创建DataLoader
3. **模型构建**: 
   - 构建TDS Encoder（left_context=1790）
   - 构建Sequential LSTM Decoder
   - 构建VEMG2PoseWithInitialState Head
4. **数值对齐验证**: 
   - 加载参考checkpoint
   - 对比同一窗口的输出
   - 验证 MSE < 1e-6, max|diff| < 1e-5
5. **实时模拟推理**:
   - 使用 InputBuffer 累积样本
   - 每累积10000点触发一次推理
   - 输出有效区间预测（~8210点，4.105秒）
   - 计算并记录指标
6. **清理**: 关闭日志，输出总结

### 训练模式（可选）

设置 `train.enabled=true` 启用训练：

```bash
python main.py --set train.enabled=true
```

训练流程：
- 使用 Adam 优化器
- 监督信号：joint_angles
- 损失函数：angle_mae (可配置权重)
- Early stopping（基于验证集）
- 保存最佳模型到 `logs/<exp_name>/best_model.pth`

## 核心技术细节

### 有效区间计算

- **输入窗口**: 10000点（5秒@2kHz）
- **Left context**: 1790点（由TDS网络架构决定）
- **Right context**: 0点
- **有效输出**: [1790, 10000) → 8210点 → 约4.105秒

### Rollout流程

1. **特征提取**: (B, 16, 10000) → Encoder → (B, 64, T_feat)
2. **有效区间裁剪**: 去除left_context部分
3. **重采样到50Hz**: 8210/2000*50 ≈ 205步
4. **LSTM逐步预测**:
   - 输入: features[t] (64) + prev_state (20) = 84维
   - 输出: pos (20) + vel (20) = 40维
   - 前~12步使用pos，后续使用vel积分
5. **线性插值**: 205步 → 8210点（对齐回原始采样率）

### 数值对齐策略

- 固定随机种子确保可复现
- 严格使用相同的模型架构和超参数
- 插值方法: linear, align_corners=True
- 仅比较有效区间输出（不含padding）

## 配置参考

### 模型配置

```yaml
model:
  network:
    type: TdsNetwork
    conv_blocks: [...]  # 初始卷积块
    tds_stages: [...]   # TDS阶段
  
  decoder:
    type: SequentialLSTM
    in_channels: 84      # 64(特征) + 20(状态)
    out_channels: 40     # 20(位置) + 20(速度)
    hidden_size: 512
    num_layers: 2
    scale: 0.01
  
  type: VEMG2PoseWithInitialState
  num_position_steps: 500     # 前500步预测位置
  state_condition: true       # 条件于前一步状态
  rollout_freq: 50           # 50Hz rollout
  provide_initial_pos: false  # 不提供初始位置
```

### 评估指标

- **angle_mae**: 角度平均绝对误差
- **angle_mse**: 角度均方误差
- **max_diff**: 最大绝对差异

所有指标仅在有效掩码（no_ik_failure=True）上计算。

## 故障排查

### 问题: 数据集路径错误

```
FileNotFoundError: 数据集目录不存在
```

**解决**: 在 `configs/realtime_config.yaml` 中修改 `data.dataset_root` 为正确路径。

### 问题: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决**: 
- 减小 batch_size: `--set data.batch_size=8`
- 使用CPU: `--set runtime.device=cpu`

### 问题: 对齐验证失败

```
总体结果: 失败 ✗
```

**可能原因**:
- 模型架构与参考不一致
- 随机种子不同
- 插值参数不匹配

**解决**: 检查配置是否严格对齐 `refer/referConfig/regression_model.yaml`

### 问题: 找不到参考checkpoint

```
跳过对齐验证: checkpoint不存在
```

**解决**: 确保 `refer/referCheckpoint/regression_vemg2pose.ckpt` 存在，或在配置中更新路径。

## 扩展开发

### 添加新指标

在 `src/utils/metrics.py` 中添加函数：

```python
def compute_my_metric(pred, target, mask):
    # 实现自定义指标
    return result
```

### 添加新模型组件

1. 在 `src/models/` 下创建新文件
2. 实现模型类
3. 在配置中添加相应字段
4. 在 `main.py` 的 `build_model()` 中集成

### 修改训练流程

编辑 `main.py` 中的 `train_model()` 函数，添加：
- 新的损失函数
- 学习率调度器
- 数据增强
- 其他训练技巧

## 性能优化（第二阶段）

当前实现专注于数值对齐，第二阶段可优化：

- **小窗快路径**: 使用更小的窗口（如1000点）降低延迟
- **缓存/增量卷积**: 复用前一窗口的计算结果
- **状态跨窗复用**: 保持LSTM状态而非每窗重置
- **模型量化**: 使用INT8量化提升推理速度
- **批处理优化**: 优化DataLoader的预取策略

## 参考

- 参考实现: `refer/emg2pose/`
- 参考配置: `refer/referConfig/regression_model.yaml`
- 计划文档: `refer/plan.md`

