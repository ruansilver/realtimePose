# 第二阶段优化计划（低延迟流式 + 对齐回归）

## 执行摘要（S2 - 低延迟流式）

- 目标：在 S1 基线上提供 50Hz 连续输出的流式推理；保留 exact 基线可随时回退。
- 关键改动：增量卷积（overlap-save）、增量重采样（2kHz→50Hz）、LSTM 跨步状态；预热 left_context 后每 20ms 输出。
- 对齐回归：对同一数据 exact vs streaming 在对齐时刻比较误差（阈值略宽于 S1）。

### exact vs streaming（行为对照）

- 重置：exact 每窗重置；streaming 仅会话级重置。
- 输出节奏：exact 5s 块状；streaming 每 20ms 连续。
- 卷积：exact 整窗卷积；streaming 增量卷积 + 缓存。
- 插值：均为线性；streaming 需与整窗插值周期性对表。
- 验收：S2 容差 MSE ≤ 1e-5、max|diff| ≤ 5e-5；P95 延迟 ≤ 20ms（设备允许）。

## 目标变更通告（S2 策略调整）

- 原目标（废弃本阶段实现）：基于 Overlap-Save 的复杂增量卷积与全链路增量计算。
- 新目标（本阶段实施）：采用“滑动窗口”流式推理，用可控冗余换取稳定性与快速落地。
  - 每 20ms 触发一次；从尾部提取长度为 `left_context + step_samples = 1790 + 40 = 1830` 的窗口；
  - 原样送入 TdsNetwork 完整前向；对产出的特征序列做“单点线性插值”到 50Hz（n_time=1，align_corners=true）；
  - 仅将该单点特征输入 stateful LSTM 做单步解码，持续保留 hidden/cell。
  - 可选冗余：窗口可取 2000 样本以加安全边界，但 seconds 计算需基于 1830 保持对齐。

## AI 执行合同（S2 给编程AI）

请严格按以下步骤在当前仓库直接实施第二阶段（低延迟流式），无需额外询问：

1) 配置与模式（滑动窗口方案）
   - 读取并必要时覆写 `configs/realtime_config.yaml`：
     - `model.mode=streaming`；`runtime.streaming.enabled=true`；`runtime.streaming.step_ms=20`；`runtime.streaming.warmup_samples=1790`；
     - 新增字段（若无则添加）：
       - `runtime.streaming.window_samples=1830`（= 1790+40）
       - `runtime.streaming.step_samples=40`
       - `runtime.streaming.feature_single_point_interpolation=true`（线性插值到 n_time=1，align_corners=true）
     - 若无法使用 TensorBoard，则确保 `logging.latency.enabled=true` 并启用 Headless 报告（见本文件“指标与验收标准”）。
   - 保留 exact 基线：`model.mode=exact` 时走第一阶段整窗流程，不修改其行为。

2) 新增/更新模块（若不存在则创建，已存在则直接覆盖更新）
   - `src/core/ring_buffer.py`：2kHz 环形缓冲；接口：`append(x: Tensor[time, channel])`、`can_step(step_ms)`、`slice_tail(n_samples)`。
   - `src/core/streaming_predictor.py`：滑动窗口推理：
     - 触发：每新增 40 样本触发一次；预热 1790 样本后开始输出；
     - 窗口：从尾部提取 `window_samples=1830`；形状 [B=1, C=16, T=1830]；
     - 编码：调用原 TdsNetwork 完整前向，得到特征序列（BCT）；
     - 单点插值：按 seconds=(1830-1790)/2000=0.02 计算 n_time=1，线性插值（align_corners=true）得到单点特征；
     - 解码：若 `state_condition=true`，与上一步状态（20维）拼接；LSTM 单步输出 40 维；前 ~12–13 步取 pos，之后对 vel 做积分；
     - 状态：跨步保留 hidden/cell；会话切换/异常时 `reset()`；
     - 计时：统计“编码器前向 + 单点插值 + LSTM 单步”的总时间 T_compute。
   - `src/models/tds_streaming.py`（可选后续优化）：Overlap-Save 增量包装器，当前阶段无需实现，仅保留占位。

3) 集成到 `main.py`
   - 根据 `model.mode` 分支：`exact` 走原整窗流程；`streaming` 走 `streaming_predictor`；
   - 记录延迟：对每个 20ms 步进行计时，输出 P50/P95/P99（若 `logging.latency.enabled=true`）。
   - Headless 落盘：`streaming_alignment.json`（exact vs streaming 容差对比）、`latency_stats.json`、`streaming_metrics.jsonl`（逐步 MAE/MSE）、`summary.json`、`config_snapshot.yaml`、`model_info.json`。

4) 约束（必须遵守）
   - 严禁使用 Lightning/Hydra；最终代码中不出现 `emg2pose` 标识（外部引用除外）。
   - 一切模型结构/参数/行为以 `refer/` 为唯一来源；涉及拓扑/参数更改必须与 `refer/` 一致并在提交中标注引用路径。

5) 验收（自动判定）
   - 数值回归：在同一数据上比较 exact vs streaming 的对齐时刻输出，`MSE ≤ 1e-5` 且 `max|diff| ≤ 5e-5`；预热边界处容许 2× 阈值；
   - 延迟：对“编码器前向 + 单点插值 + LSTM 单步”的 T_compute，P95 ≤ 20ms、P50 ≤ 10ms（设备允许时）；
   - 稳定：连续运行≥30min 无内存泄漏与累计延迟增长；
   - 机读工件齐全且 `passed=true`。

6) 非目标（不要做）
   - 不开启在线学习；不改动第一阶段 exact 行为；不尝试新架构；不新增启动脚本/测试样例。


## 0. 目标与范围

- 在第一阶段严格对齐的基础上，引入“低延迟、持续输出”的流式推理能力，同时保留“严格对齐基线模式（exact）”。
- 所有优化均以 `refer/` 为唯一参考源进行实现和验证（参见 plan.md 2.6）。
- 不引入 Lightning/Hydra；维持单入口 `main.py`；配置依旧由 YAML + argparse 驱动；直接覆盖更新。

不在本阶段范围：
- 新模型架构尝试（除非以配置选择，在 exact 基线上证明对齐/收益）。
- 复杂在线学习策略（默认关闭，仅保留配置占位）。

## 1. 输出节奏与时序（流式）

- 步进频率：50Hz（每 20ms 触发一次推理步）。
- 预热策略：在累计到 `left_context=1790` 个样本后，开始按 50Hz 连续输出；之前不产出结果。
- 对齐模式：
  - exact（基线）：维持第一阶段 5s 整窗、块状输出与每窗重置。
  - streaming（优化）：持续输出，跨步保持内部状态；用“回归对齐测试”监控数值漂移。

## 2. 架构变更与新增组件

### 2.1 输入与上下文缓冲

- `RingBuffer`（2kHz 原始样本）：
  - 维护一条长度≥`left_context + streaming_horizon` 的滑动窗口。
  - 每到达可整除 20ms 的边界，触发一次 50Hz 步。

- `ContextManager`：
  - 管理跨步所需的历史上下文（至少 `left_context` 个样本）。
  - 在 exact 模式下禁用（每窗重置）。

### 2.2 滑动窗口方案（本阶段主路径）

- 以 `window_samples=1830` 进行整网前向，确保多层 stride 后仍满足卷积 kernel 尺寸（不再出现“kernel 大于输入”的报错）。
- 对编码器输出特征序列做“单点线性插值”到 50Hz（n_time=1，align_corners=true），以保证与 exact 的 50Hz 对齐方式完全一致，避免直接取“最后一帧特征”带来的偏差。

### 2.3 TdsNetwork 增量计算（Overlap-Save，S2-Advanced 可选）

- 保持为未来优化方向；当前阶段不实现，不影响验收。

### 2.4 单点特征插值器

- 对单次窗口前向得到的特征序列进行线性插值（n_time=1，align_corners=true），返回单点特征供 LSTM 单步解码。

### 2.4 解码器与状态管理

- `SequentialLSTM` 本身支持逐步；在 streaming 模式下：
  - 跨步保留 hidden/cell 状态；
  - 输入为“最新一刻的 encoder 特征（64）+ 上一步状态（20，若 state_condition=true）”；
  - 输出 40 维，前半为 pos，后半为 vel；`num_position_steps` 在 50Hz 下生效（约 12–13 步后切换到速度积分）。

- exact 模式：每窗开始时 `reset_state()`；streaming 模式：仅在系统重置/会话切换时清零。

### 2.5 双模式切换与回归

- `model.mode: exact | streaming`：
  - exact：完全复用第一阶段逻辑（块状、重置、整窗后输出）。
  - streaming：增量卷积 + 增量重采样 + 跨步 LSTM 状态；20ms 连续输出。
- 回归对齐：对同一录制数据，比较 exact 与 streaming 在对齐时刻的误差（允许微小差异，阈值略宽于第一阶段）。

## 3. 配置扩展（YAML + argparse）

```yaml
model:
  mode: streaming   # exact | streaming

runtime:
  streaming:
    enabled: true
    step_ms: 20               # 50Hz
    warmup_samples: 1790      # = left_context
    window_samples: 1830      # = left_context + step_samples（滑动窗口）
    step_samples: 40
    feature_single_point_interpolation: true  # 对特征序列做单点线性插值（n_time=1, align_corners=true）
    max_buffer_seconds: 10    # 防护上限

eval:
  streaming_alignment_check: true
  streaming_tolerance:
    mse: 1.0e-5
    max_diff: 5.0e-5

logging:
  latency:
    enabled: true
    percentiles: [50, 95, 99]
```

命令行覆写示例：

```bash
python main.py \
  --set model.mode=streaming \
  --set runtime.streaming.enabled=true \
  --set eval.streaming_alignment_check=true
```

## 4. 指标与验收标准（S2）

必须满足：
1) 延迟与吞吐：
   - 单步（20ms）核心计算 T_compute（编码器前向 + 单点插值 + LSTM 单步）P95 ≤ 20ms；P50 ≤ 10ms（设备允许时）。
   - 内存占用随时有界（不随时间线性增长）。
2) 数值回归：
   - 在同一数据上比较 exact vs streaming 的对齐时刻输出：
     - `MSE ≤ 1e-5` 且 `max|diff| ≤ 5e-5`（允许略宽于 S1）。
   - 边界处（预热后第一个输出）单独记录并不高于 2×上述阈值。
3) 稳定性：
   - 长时运行（≥30min）无内存泄漏、无状态爆炸、无累计延迟增长。
4) 规范遵循：
   - 不使用 Lightning/Hydra；不新增启动脚本；不出现 `emg2pose` 标识（外部引用除外）；以 `refer/` 为唯一参考源。

## 5. 里程碑

- S2-M1：流式步进与重采样
  - 实现 `RingBuffer` + `IncrementalResampler`，能以 50Hz 推进并产出步进时间戳。
  - 在 exact 模式下验证插值一致性（对同窗插值结果一致）。

- S2-M2：TdsNetwork 增量卷积
  - 基于 overlap-save 引入逐层缓存与最小重算；
  - 用录制数据对比“整窗卷积 vs 增量卷积”的一致性（逐层输出误差小于阈值）。

- S2-M3：连续解码器与状态
  - streaming 模式下跨步保留 LSTM hidden/cell；
  - `num_position_steps` 行为在 50Hz 下验证；
  - 引入 streaming 对齐回归测试（exact vs streaming）。

- S2-M4：端到端性能与回归看板
  - main.py 中集成延迟/吞吐/内存监控；
  - TensorBoard 面板：延迟分布（P50/P95/P99）、误差曲线（exact vs streaming）、内存曲线。

## 6. 风险与缓解

- 层级 stride 对齐与边界效应：
  - 采用层级步长对齐与最小公倍数推进，减少重算；
  - 在边界处引入“保护窗口”，对比整窗结果并记录差异。

- 插值累计误差：
  - 使用固定精度与确定性实现，周期性与整窗插值结果对表；
  - 出现漂移即回退至 exact 模式继续运行，并记录事件。

- 状态漂移：
  - 周期性软重置（可配置，如每 N 秒在不影响输出的情况下做轻量对齐校正）；
  - 引入漂移监控指标（近窗 MSE、低频偏移）。

## 常见坑 / FAQ（S2）

- 为什么流式输出与整窗略有差异？增量卷积/插值的边界处理可能引入微小数值差异，需满足 S2 容差并通过回归看板观察。
- 何时回退 exact？一旦 streaming 对齐回归超阈值或延迟异常上升，应立即切换 exact 模式并记录事件。
- 步长如何对齐？以层级 stride 的最小公倍数推进，减少重算并避免错位。
- 预热多长？至少 `left_context=1790` 样本，确保第一帧输出有效。

## 7. 实现指引与模块边界

- 仍由 `main.py` 单入口控制；通过配置切换 exact/streaming；
- 新增/扩展模块（名称示意，最终命名遵循现有结构）：
  - `src/core/ring_buffer.py`：2kHz 环形缓冲与步进触发；
  - `src/core/incremental_resampler.py`：2kHz→50Hz 增量线性插值；
  - `src/models/tds_streaming.py`：TdsNetwork 的增量前向包装器（内部调用现有层，实现缓存与 overlap-save）；
  - `src/core/streaming_predictor.py`：封装流式端到端推理（依赖上述模块）。

> 保持与 plan.md 一致的禁止项与“参考源为 refer/”的约束；所有参数仍通过 YAML 管理，并支持命令行覆写。

## 8. 交付物

- 更新后的 `main.py`（流式模式集成与切换）。
- 新增核心模块（见 7），无额外启动脚本/测试文件。
- 配置扩展：`configs/realtime_config.yaml` 增补 streaming 与对齐回归相关字段。
- 文档：在 `USAGE.md` 增补“流式模式使用说明与指标看板指引”。



## 修改方案（S2 修订指令，给编程AI）

请按以下修订项更新流式实现，确保通过 S2 验收：

1) 立即修复项（对齐一致性）
   - 解码器状态：在 streaming 会话开始与每次对齐回归测试开始时，调用 `decoder.reset_state()`，与 exact 起点一致。
   - 位置步数换算：
     - 将 `rollout_position_steps = int(num_position_steps * rollout_freq / 2000)` 改为：
       - `rollout_position_steps = round(num_position_steps * (rollout_freq / 2000))`
     - 原因：与 refer/ 一致，避免切换处少 1 步导致局部尖峰（Max Diff 偏大）。
   - 单点特征插值与取样时刻：
     - 使用“线性插值 + align_corners=true + n_time=1”的单点插值（左缘取样），不要加 half-step 偏移；
     - seconds 计算：`seconds = (window_samples - left_context - right_context) / 2000`；在 `window_samples=1830, left_context=1790, right_context=0` 时，`seconds=0.02`。

2) 滑动窗口参数（配置）
   - `runtime.streaming.warmup_samples = 1790`
   - `runtime.streaming.step_samples = 40`
   - `runtime.streaming.window_samples = 1830`（=1790+40）
   - `runtime.streaming.feature_single_point_interpolation = true`

3) 执行步骤（伪代码）
```
for each 20ms step (every step_samples=40 new EMG):
  if not warmed_up(1790): continue
  x = ring_buffer.slice_tail(window_samples=1830)         # shape [1, C=16, T=1830]
  feats = tds_network_forward(x)                          # shape [1, F=64, T_f]
  feats_1 = linear_interpolate_single_point(feats,        # n_time=1, align_corners=true
               seconds=(1830-1790)/2000)                  # -> shape [1, 64]
  if state_condition: inp = concat([feats_1, prev_state]) # [1, 84]
  out = lstm_step(inp)                                    # [1, 40] -> split pos/vel
  if t_step < rollout_position_steps: state = pos
  else: state = state + vel
  emit(state)
```

4) 性能剖析（任务A）
   - 计时范围：`T_compute = 编码器前向 + 单点插值 + LSTM 单步`；
   - GPU 上使用 `torch.cuda.synchronize()` 包围计时代码；预热后统计 P50/P95/P99；
   - 验收：`T_compute P95 < 20ms`（P50 ≤ 10ms 为目标）。

5) 对齐回归（任务B）
   - 使用同一底层数据，比较 streaming 单步输出与 exact 同时刻输出；
   - 阈值：`MSE ≤ 1.0e-5` 且 `max|diff| ≤ 5.0e-5`；
   - 若失败，先核对两点：`round()` 是否生效；插值是否为 `n_time=1, align_corners=true`（左缘取样）。

6) 机读工件（Headless）
   - 更新/生成：`streaming_alignment.json`（含阈值与 passed）、`latency_stats.json`、`streaming_metrics.jsonl`、`summary.json`、`config_snapshot.yaml`、`model_info.json`。

7) 回退策略
   - 任一验收不达标：将 `model.mode=exact`，保留日志与工件；
   - 修复后重新跑回归并提交结果。
