# 第二阶段优化计划（低延迟流式 + 对齐回归）

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

### 2.2 TdsNetwork 增量计算（Overlap-Save）

- 针对含 stride 的时序卷积/子采样链，采用“重叠-保存（overlap-save）”策略：
  - 为每一层维护所需的历史输入切片与边界缓存。
  - 新到样本段仅计算“增量部分”，并拼接到层级输出缓存。
  - 按层处理 stride，引入最小化重算的对齐规则（以层级 stride 的最小公倍数为对齐步长）。

- 产出特征序列（BCT）后，暴露“新增时刻”的特征切片给解码器。

### 2.3 50Hz 增量重采样器

- `IncrementalResampler`（2kHz→50Hz）：
  - 维护插值器状态（线性插值），支持逐步推进、每 20ms 产出一个采样点。
  - 与第一阶段保持同一插值方法（线性），确保数值一致性在窗口级别满足容差。

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
   - 单步（20ms）端到端推理延迟 P95 ≤ 20ms（设备允许时）；P50 ≤ 10ms。
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


