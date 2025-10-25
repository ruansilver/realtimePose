# 实时sEMG姿态预测系统使用说明

## 系统概述

本系统是一个具备在线增量学习能力的实时sEMG姿态预测系统，基于优秀的离线VEMG2Pose模型构建。系统支持：

- 低延迟实时预测（<500ms）
- Fast Path + Correction Path混合窗口策略  
- 在线增量学习和模型适应
- 双缓冲区管理
- 全面的性能监控和日志记录

## 快速开始

### 1. 环境要求

系统运行需要以下Python包（假设已安装）：
- torch
- numpy  
- h5py
- yaml
- psutil
- tensorboard

### 2. 配置系统

系统使用YAML配置文件进行管理。主要配置文件：

- `configs/realtime_config.yaml` - 生产环境配置
- `test_config.yaml` - 测试环境配置（简化版）

### 3. 运行系统

**使用默认配置运行：**
```bash
python main.py
```

**使用命令行参数：**
```bash
python main.py --dataset-path "D:/Dataset/emg2pose_dataset_mini" --fast-window-size 1.5 --enable-online-learning
```

## 系统架构

```
cursorRealTimePose/
├── main.py                    # 主入口脚本
├── configs/
│   └── realtime_config.yaml   # 主配置文件
├── src/
│   ├── models/                 # 模型组件
│   │   ├── realtime_pose_model.py
│   │   ├── networks.py
│   │   ├── modules.py
│   │   └── constants.py
│   ├── core/                   # 核心功能
│   │   ├── realtime_predictor.py
│   │   ├── buffer_manager.py
│   │   ├── online_learner.py
│   │   └── data_simulator.py
│   ├── data/                   # 数据处理
│   │   ├── dataset.py
│   │   └── transforms.py
│   └── utils/                  # 工具组件
│       ├── config_manager.py
│       ├── checkpoint_manager.py
│       └── logger.py
├── logs/                       # TensorBoard日志
└── checkpoints/               # 模型检查点
```

## 配置说明

### 核心配置项

**数据配置 (data):**
- `dataset_path`: 数据集路径
- `batch_size`: 批次大小
- `window_length`: 数据窗口长度

**实时预测配置 (realtime):**
- `fast_window_size`: Fast Path窗口大小（秒）
- `correction_window_size`: Correction Path窗口大小（秒）
- `max_latency`: 最大允许延迟（毫秒）
- `fusion_weight`: Fast Path与Correction Path融合权重

**在线学习配置 (online_learning):**
- `enabled`: 是否启用在线学习
- `learning_rates`: 分层学习率设置
- `trigger_conditions`: 触发条件配置

**缓冲区配置 (buffers):**
- `raw_data.size`: 原始数据缓冲区大小
- `feature.size`: 特征缓冲区大小

### 命令行参数

系统支持通过命令行参数覆盖配置文件设置：

```bash
python main.py \
  --dataset-path "D:/Dataset/emg2pose_dataset_mini" \
  --fast-window-size 1.0 \
  --correction-window-size 4.0 \
  --max-latency 500 \
  --enable-online-learning \
  --encoder-lr 1e-6 \
  --decoder-lr 1e-4 \
  --log-level INFO
```

## 监控和日志

### TensorBoard监控

启动TensorBoard查看实时监控：
```bash
tensorboard --logdir ./logs
```

访问 http://localhost:6006 查看：
- 预测延迟统计
- 缓冲区利用率
- 在线学习进度
- 系统性能指标

### 日志信息

系统会记录以下信息：
- 预测性能统计
- 在线学习触发和结果
- 缓冲区状态
- 系统资源使用
- 错误和警告信息

## 性能优化

### 延迟优化
1. 调整`fast_window_size`到最小可行值
2. 减少`correction_frequency`以降低校正开销
3. 优化缓冲区大小平衡内存使用和性能
4. 使用GPU加速（如果可用）

### 内存优化
1. 限制缓冲区大小
2. 调整`keep_last_n`减少检查点存储
3. 定期清理日志文件

### 学习效果优化
1. 调整触发条件平衡学习频率和性能
2. 使用适当的学习率避免过拟合
3. 监控学习统计调整参数

## 故障排除

### 常见问题

**1. 数据集路径错误**
```
错误: 数据集路径不存在
解决: 检查configs/realtime_config.yaml中的dataset_path设置
```

**2. 延迟过高**
```
警告: 预测延迟超标
解决: 减小窗口大小或关闭在线学习
```

**3. 内存不足**
```
错误: CUDA out of memory
解决: 减小批次大小或缓冲区大小
```

**4. 模型加载失败**
```
错误: 检查点加载失败
解决: 检查checkpoint_path设置或使用strict=False
```

### 调试模式

启用详细日志进行调试：
```bash
python main.py --log-level DEBUG
```

## 系统扩展

### 添加新的触发策略
1. 修改`OnlineLearner.should_trigger_learning()`方法
2. 添加新的配置选项
3. 更新配置文件模板

### 添加新的预测模式
1. 继承`RealtimePredictor`类
2. 实现新的预测策略
3. 在配置中添加模式选择

### 自定义数据变换
1. 在`src/data/transforms.py`中添加新变换
2. 在数据加载流程中应用变换
3. 更新配置选项

## 支持与维护

系统设计遵循科研代码风格，优先考虑：
- 代码可读性和可复现性
- 模块化和可扩展性
- 配置驱动的实验管理
- 端到端的自动化流程

如需修改或扩展功能，建议：
1. 保持模块间的低耦合
2. 通过配置文件管理参数
3. 添加适当的日志和监控
4. 进行充分的测试验证
