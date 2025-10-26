# 第一阶段实现计划（严格对齐离线 + 新增研发规范）

## 0. 背景与目标

- 以 `refer/referConfig/regression_model.yaml` 与 `refer/emg2pose/*` 为“黄金标准（离线参考）”，在不引入 Lightning/Hydra 的前提下，构建可配置、模块化的端到端系统，严格复现离线模型的推理与训练行为，用于数值对齐与研究迭代。
- 第一阶段聚焦“严格数值对齐”，不做延迟优化；输出以 5s 整窗为单位批量产生。

## 1. 范围与非目标

- 范围（必须实现）
  - 配置驱动系统（YAML + argparse），参数与代码解耦。
  - 数据加载（HDF5）、窗口化切片（10000 点）、有效区间裁剪与对齐、模型推理与（可选）训练、日志记录与可视化、端到端主程序。
  - 数值对齐验收（与离线参考逐时刻对齐，在严格容差内）。
- 非目标（本阶段不做）
  - 实时低延迟优化（小窗快路径、缓存/增量卷积、状态跨窗复用）。
  - 企业级健壮性与异常全覆盖处理。

## 2. 研发规范与新增需求落地

### 2.1 配置文件驱动（Configuration-Driven）

- 采用 YAML + argparse：
  - 默认从 `configs/realtime_config.yaml` 读取全量配置（模型、数据、训练、运行时、日志）。
  - 命令行可通过 `--set key.subkey=value` 方式覆写（仅作为便捷入口，不改变 YAML 为主的设计）。
  - 代码中不硬编码实验参数；任何超参改动仅在 YAML/命令行完成。

#### 2.1.1 第一阶段配置缺省（固化）

- 在线学习：`online_learning.enabled = false`（不实现 online_learner，仅保留配置占位）
- 数值对齐验证：`eval.alignment_check = true`（在 `main.py` 内集成，对比参考 checkpoint）
- 训练优先级：`train.enabled = false`（先完成推理；训练作为后续可选）
- 指标模块：`eval.metrics = [angle_mae, angle_mse]`（实现并用于评估）
- 参考权重：`reference_checkpoint = refer/referCheckpoint/regression_vemg2pose.ckpt`

示例配置片段（仅示意关键位）：

```yaml
train:
  enabled: false

online_learning:
  enabled: false

eval:
  alignment_check: true
  metrics: [angle_mae, angle_mse]

reference_checkpoint: refer/referCheckpoint/regression_vemg2pose.ckpt
```

### 2.2 科研代码风格（Research-Oriented）

- 结构清晰、逻辑简洁、模块边界明确；优先可读性、可复现性与可扩展性。
- 适度断言与错误信息，避免过度防御；注释聚焦设计动机与关键约束，避免赘述显而易见逻辑。

### 2.3 模块化与解耦（Modular & Decoupled）

- 组件划分与可替换：
  - 数据加载器（HDF5 读取、过滤、窗口化）。
  - 模型结构（Encoder/Decoder/Head），按配置切换具体实现或超参。
  - 实时预测器（流式窗口聚合、块状前向、输出对齐）。
  - 在线学习器（可选模块，默认关闭；接口预留，配置可启用）。
- 依赖方向：上层仅依赖接口，不依赖具体实现；通过配置选择实现类。

### 2.4 端到端执行（End-to-End Execution）

- 单一入口脚本 `main.py`：一键启动，自动完成配置加载、数据准备、（可选）训练、离线对齐验证、实时模拟推理、日志记录与可视化。
- 所有流程分支（训练/只推理/对齐验证）由配置开关控制。

### 2.5 无缝覆盖更新（Direct Overwrite）

- 迭代时直接在原有模块上覆盖更新；不做向后兼容适配层。

### 2.6 参考源与差异控制（Source of Truth）

- 模型结构、超参数与对齐相关设置的任何修复/调整，必须以 `refer/` 目录下的配置与代码为唯一参考源；严禁从网络检索未验证的模型设置或参数作为依据。
- 覆盖范围包括但不限于：网络拓扑（卷积 kernel/stride、TDS stages 参数、线性投影维度等）、LSTM 维度与层数、head 行为（position/velocity 切换逻辑、rollout_freq、插值方式、left/right context 计算方法）。
- 如需变更，需在提交说明中引用对应 `refer/` 文件与定位信息（文件路径/段落），以便审计对齐性。

## 3. 禁止项与最终交付体验

### 3.1 严格禁止

- 创建任何测试文件或硬编码示例。
- 创建额外快速启动脚本（shell/bash/Makefile 等）。
- 除外部资源引用外，最终代码中不得出现 `emg2pose` 标识。
- 禁用 PyTorch Lightning 与 Hydra。
 - 从网络检索未验证的模型结构或参数并用于实现/修复（以 `refer/` 为唯一来源）。

### 3.2 最终用户体验

1) 打开 `configs/realtime_config.yaml` 即可理解与修改全部实验参数。
2) 代码结构清晰、职责分明、易于扩展。
3) 终端执行 `python main.py` 即可完整运行实验。
4) 运行中可用 `tensorboard --logdir ./logs` 实时查看曲线与结构图。

### 3.3 环境说明

- 默认所需 Python 库已安装，代码不做环境校验。
- 运行环境示例：`conda activate D:\env\condaEnv\emg2Pose`。

## 4. 数据特征与加载策略

- 数据集：可配置，默认 `D:/Dataset/emg2pose_dataset_mini/`；目录下为多个 HDF5 文件。
- HDF5 结构：每文件包含 `emg2pose/timeseries` 复合数据集；每行含字段：
  - `time: float64`
  - `joint_angles: float32[20]`（监督信号）
  - `emg: float32[16]`（输入信号）
- 过滤规则：加载时自动丢弃 `joint_angles` 全 0 的帧（视为 IK 失败或无效标签）。
- 窗口与采样：
  - 采样率 `2000 Hz`；窗口 `10000` 点（5s），无重叠（stride=window_length）。
  - 左右 padding 固定为 `(0,0)`；有效输出区间为 `[left_context, window_length)`。
  - 计算约束：`left_context = 1790`，`right_context = 0`（与离线参考一致）。

## 5. 模型与推理规范（严格对齐离线）

- Encoder：`TdsNetwork`，拓扑与超参严格等同 YAML。
- Decoder：`SequentialLSTM`（in=84=特征64+状态20，hidden=512，layers=2，out=40，scale=0.01）。
- Head：`VEMG2PoseWithInitialState`（`rollout_freq=50 Hz`，`num_position_steps=500`，`state_condition=true`，`provide_initial_pos=false`）。
- 推理流程：
  1) 编码器输出特征（BCT）。
  2) 特征重采样至 50Hz；逐步自回归：输入为特征（64）与上一步状态（20，若启用 state_condition）。
  3) 每步 LSTM 输出 40 维，拆分为 pos(20)/vel(20)；前 ~12–13 步取 pos，之后对 vel 做积分（pos[t] = pos[t-1] + vel）。
  4) 将 50Hz 预测线性插值回“有效区间”的原采样步数，与标签严格对齐。
- 输出节奏：块状输出（每窗一次性产出约 4.105s 有效预测）。

## 6. 端到端流程（main.py）

1) 读取 YAML 配置，解析命令行覆写；设定随机种子与设备。
2) 数据管线：
   - 训练/验证：HDF5 读取 → 过滤零标签帧 → 窗口切片（训练可选抖动，验证/测试不抖动）。
   - 实时模拟：从本地数据流按 2kHz 重放 → `InputBuffer` 累积至 10000 点触发推理。
3) 构建模型（按配置装配 Encoder/Decoder/Head）。
4) 训练（可选）：标准监督回归；损失权重与学习率由配置控制（第一阶段默认关闭）。
5) 数值对齐验证（集成于 main.py）：同一窗口数据，加载 `reference_checkpoint` 得到离线参考输出，并与本实现输出在有效区间逐时刻比较，达成容差阈值。
6) 日志记录：使用 `torch.utils.tensorboard` 写入标量、直方图与结构图；日志目录由配置指定。

## 7. 模块边界与接口（建议）

- `src/data/dataset.py`
  - 读取 HDF5、过滤零标签帧、切窗（window_length/stride/padding/jitter）。
  - 返回批字典：`{"emg": CT, "joint_angles": CT, "time": T}`。
- `src/core/buffer_manager.py`
  - 流式聚合 2kHz 样本；`append(samples)`、`has_full_window()`、`get_window()`。
- `src/models/networks.py` / `src/models/modules.py`
  - `TdsNetwork`、`SequentialLSTM`、`VEMG2PoseWithInitialState` 的等价实现（名称可不同，但行为一致）。
- `src/core/realtime_predictor.py`
  - 从 `InputBuffer` 取满窗 → 执行前向 → 返回该窗有效区间预测。
- `src/core/online_learner.py`（可选）
  - 预留在线更新接口，默认关闭，由配置控制启用与否。
- `src/utils/logger.py` / `src/utils/config_manager.py`
  - TensorBoard 日志；YAML 加载与 argparse 覆写。
 - `src/utils/metrics.py`
   - 实现并导出 `angle_mae`、`angle_mse` 等基础指标（在有效区间上计算，支持掩码）。

> 注意：最终代码文件与类名中不得出现 `emg2pose` 标识；本计划书中提及的“离线参考”属于外部资源引用范畴。

## 8. 验收标准（必须全部满足）

1) 数值对齐：对同一 10000 点窗口，离线参考与本实现的有效区间预测逐时刻差异满足：
   - `MSE < 1e-6` 且 `max|diff| < 1e-5`（或团队确认的等效严格阈值）。
2) 行为对齐：
   - `left_context=1790`、`right_context=0`、`rollout_freq=50 Hz`、`num_position_steps=500`、`state_condition=true`、`provide_initial_pos=false`。
   - 输出节奏为块状；无窗间状态复用；插值方法为线性。
3) 配置体验：
   - 打开 `configs/realtime_config.yaml` 可理解并调整所有关键参数；命令行可覆写。
4) 端到端：
   - `python main.py` 可完成训练（可选）、对齐验证与实时模拟推理；
   - `tensorboard --logdir ./logs` 可查看曲线与结构图。
5) 规范约束：
   - 不使用 Lightning/Hydra；无测试文件与硬编码示例；不出现 `emg2pose` 标识（外部引用除外）。
6) 指标模块：
   - 已实现并集成 `angle_mae`、`angle_mse`，用于评估与对齐验证。

## 9. 里程碑与交付

- M1（配置与骨架）：
  - YAML + argparse 管线、目录结构、模块接口完成；能加载数据并切窗。
- M2（模型装配与前向）：
  - Encoder/Decoder/Head 等价实现；端到端前向跑通；日志输出。
- M3（数值对齐）：
  - 离线参考 vs 本实现 对齐验证脚本集成进主流程；对齐达标。
- M4（训练与文档）：
  - 基本训练回路可用（可选）；完善 `USAGE.md` 与配置说明；交付演示。

## 10. 风险与应对

- 浮点差异：不同硬件/库版本导致微小偏差；通过固定随机种子与严格容差控制，并在插值与上下文裁剪环节做一致性自检。
- 数据时间轴偏差：若真实设备非 2kHz，需要前置重采样至 2kHz；本阶段建议用本地数据流“等时重放”。
- 窗口边界对齐：严格以样本索引定义边界，避免时间戳累计误差；在日志中输出每窗的起止索引与有效区间长度。

---

附：命令行覆写建议（示例）

```bash
python main.py \
  --set data.dataset_root=D:/Dataset/emg2pose_dataset_mini \
  --set train.enabled=true \
  --set train.learning_rate=5e-4 \
  --set runtime.buffer.max_samples=10000
```

以上为第一阶段的执行蓝本。若需进入“第二阶段（低延迟优化）”，需在完成数值对齐后，增量评审并补充对齐回归测试策略。

