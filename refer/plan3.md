# 第三阶段修复计划（编码器边界效应隔离与缓解）

## 执行摘要
- 目标：定位并缓解 streaming 模式中由“短窗重复编码”引入的编码器边界效应，修复对齐回归失败（MSE≈5e-4、MaxDiff≈0.13）。
- 策略：先隔离问题（用 exact 的“黄金特征”驱动流式解码器验证解码器路径正确），再用“窗口增厚”降低边界误差，保持总体延迟≤500ms。
- 约束：不改变 exact 基线；不引入新框架；一切拓扑/参数严格以 `refer/` 为唯一来源；提交需溯源。

## AI 执行合同（给编程AI）
请严格按 A→B 顺序实施；A 用于诊断（临时代码），B 为最终修复（可合并）。

---

## 步骤 A：诊断测试（隔离编码器，验证解码器）

目标：用 exact 模式一次性编码+重采样得到的“黄金特征（features_golden_exact）”驱动 streaming 的 LSTM 单步逻辑；若此时对齐通过，说明问题在“短窗重复编码”的边界效应。

实施位置：`main.py` → `streaming_alignment_check()` 内（仅对齐测试路径，勿改主推理路径）。

实现指令：
1) 生成黄金特征
- 对 `test_emg` 的 10000 样本窗口，调用 exact 的编码器前向，得到整窗特征序列。
- 按 exact 的 50Hz 线性插值（align_corners=true）重采样到 `[T_rollout, 64]`，命名为 `features_golden_exact`（通常 T_rollout≈205）。

2) 用“黄金特征”驱动 streaming LSTM 单步
- `decoder.reset_state()`，`state = zeros(20)`，`rollout_position_steps = round(500 * 50 / 2000) = 12`。
- for k in [0..T_rollout-1]:
  - `feat_k = features_golden_exact[k]`（形状 [64]），若 `state_condition=true` 则拼接 `[feat_k, state] → [84]`。
  - `out = decoder(feat_k|[feat_k,state]) → [40]`，拆分 `pos, vel`。
  - `state = pos` 若 `k < 12`，否则 `state = state + vel`。
  - 收集 `state` 到 `streaming_from_golden`。

3) 对齐验证
- 将 `streaming_from_golden` 与 exact 的（相同时刻）输出做逐步对齐比较；阈值：`MSE ≤ 1e-5`、`max|diff| ≤ 5e-5`。
- 通过：证明问题根源在编码器边界；失败：检查 state_condition 拼接顺序、切换步数是否 round、初始化是否全零。

4) 临时代码
- 步骤 A 的“混合模式”逻辑仅在 `streaming_alignment_check()` 内使用，诊断完成后需回退。

---

## 步骤 B：务实修复（窗口增厚以降低边界误差）

目标：在 streaming 正常路径中保留 1830 取样对齐关系的同时，增厚编码窗口，为卷积链提供更稳定上下文，降低边界误差；评估最小可通过阈值的增厚因子 n。

实施位置：`src/core/streaming_predictor.py`

实现指令：
1) 配置扩展（`configs/realtime_config.yaml`）
```yaml
runtime:
  streaming:
    window_samples_base: 1830   # 固定对齐基准（=1790+40）
    step_samples: 40            # 已存在
    warmup_samples: 1790        # 已存在
    window_thicken_factor: 5    # 新增，起始建议值 n=5
```

2) 窗口增厚逻辑（仅编码器输入改动）
- 将原 `window_samples = 1830` 替换为：
  - `L = window_samples_base + window_thicken_factor * step_samples`（例如 1830+5*40=2030）。
- 从环形缓冲的尾部提取 `L` 样本送入编码器完整前向。
- 其余逻辑保持不变：单点线性插值 `size=1, mode=linear, align_corners=true`；`state_condition` 拼接；`round` 切换；时间戳仍按当前步左缘。
- 注意：`seconds` 与取样时刻仍基于 base 窗口（1830 对齐），不随 L 变化。

3) 试验与收敛
- 从 `window_thicken_factor = 5` 开始，重跑 `streaming_alignment_check()`；若 MSE/MaxDiff 仍超阈值，逐步升高 n（6/7/8），找到最小可过阈值的 n（同时确保 P95 延迟≤20ms，总体延迟≤500ms）。

---

## 验收与工件
- 对齐回归：
  - 步骤 A（诊断）：`passed=true`（MSE ≤ 1e-5，max|diff| ≤ 5e-5）。
  - 步骤 B（修复）：`passed=true`，并给出最终的 `window_thicken_factor`。
- 性能：
  - `latency_stats.json`：P50≤10ms、P95≤20ms、P99≤50ms；记录 n 值。
- 工件（logs/<run_id>/）：
  - `streaming_alignment.json`，`latency_stats.json`，`summary.json`，`config_snapshot.yaml`，`model_info.json`（含 n）。

## 回退策略
- 任一步骤失败：保持 `model.mode=exact` 可用；记录失败日志与对比曲线，提交观察。

## 代码指引（概要）
- `main.py`：在 `streaming_alignment_check()` 内做好 A 的“混合模式”分支（诊断）；完成后回退。
- `src/core/streaming_predictor.py`：实现 B 的窗口增厚与读取；其余不变。
- 仅使用 `torch.nn.functional.interpolate(..., size=1, mode='linear', align_corners=True)` 做单点插值；严禁“取最后一帧特征”。

## 提交说明模板（要求溯源 refer/）
```
变更摘要：
- [A] 诊断：exact 编码器 + 流式 LSTM 单步对齐通过；
- [B] 修复：窗口增厚（n=__）后 streaming 对齐通过；

refer/ 溯源：
- 编码器/解码器/头部：refer/emg2pose/networks.py、pose_modules.py（position/velocity 逻辑、rollout_freq 等）；

验证：
- A：MSE/MaxDiff 结果与日志；
- B：最小 n 值、对齐结果、延迟统计与工件路径；
```


## 编码要求（Coding Requirements）
- 语言与版本：Python 3.10+；新增/修改函数须提供类型注解（返回值与参数）。
- 命名与风格：遵循 PEP8 与本仓库现有风格；变量/函数名需含义清晰，避免 1–2 字母短名与魔数（写入配置/常量）。
- 配置驱动：`YAML + argparse` 为唯一入口；新增键如 `runtime.streaming.window_thicken_factor`、`window_samples_base` 必须来自配置；严禁硬编码。
- 禁止项：不使用 Hydra/PyTorch Lightning；代码与类名中不得出现 `emg2pose` 标识（外部溯源文档除外）。
- 可复现性：设置随机种子；`torch.backends.cudnn.deterministic=True` 且 `benchmark=False`；Windows 下 HDF5/多进程限制 `num_workers=0`。
- 精度与计时：对齐阶段禁用 AMP；统一 `torch.float32`；计时段在 GPU 上使用 `torch.cuda.synchronize()` 包围。
- 设备无关：优先使用 `cuda`，否则回退 `cpu`；在 `summary.json` 中记录实际设备。
- 日志与工件：保留 TensorBoard；必须生成 Headless 工件（JSON/JSONL），以满足无人值守验收。
- 单入口：仍使用 `main.py`；不新增 shell/Makefile/测试样例。
- 注释原则：仅保留必要设计意图与关键约束说明，避免陈述性冗余注释。

## 环境与运行（Environment & Usage）
- 运行环境：默认依赖已安装；示例 `conda activate D:\env\condaEnv\emg2Pose`；无需在代码中做环境校验。
- 数据路径：`D:\Dataset\emg2pose_dataset_mini\`（可通过 `configs/realtime_config.yaml` 覆写 `data.dataset_root`）。
- 设备选择：配置 `runtime.device=cuda`（若可用），否则自动切换 `cpu`；实际设备在 `summary.json` 与 stdout 打印。
- 常用运行命令：
```bash
# 开启流式对齐回归（默认 1830 基线窗口）
python main.py --set model.mode=streaming --set eval.streaming_alignment_check=true

# 指定数据路径与窗口增厚（计划 B）
python main.py \
  --set data.dataset_root=D:/Dataset/emg2pose_dataset_mini \
  --set model.mode=streaming \
  --set eval.streaming_alignment_check=true \
  --set runtime.streaming.window_thicken_factor=5

# 仅输出 Headless 工件（禁用 TensorBoard）
python main.py --set logging.tensorboard.enabled=false --set eval.output_format=json
```
- 结果产出：`logs/<run_id>/` 下包含 `streaming_alignment.json`、`latency_stats.json`、`summary.json`、`config_snapshot.yaml`、`model_info.json`、`streaming_metrics.jsonl`。
- 验收口径：
  - 步骤 A：`streaming_alignment.json.passed=true`（MSE ≤ 1e-5，max|diff| ≤ 5e-5）。
  - 步骤 B：`passed=true`，并在 `model_info.json` 或 `summary.json` 标注最终 `window_thicken_factor`；`latency_stats.json` 满足 P50≤10ms、P95≤20ms、P99≤50ms。


