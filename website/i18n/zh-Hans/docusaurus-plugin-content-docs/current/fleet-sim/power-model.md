---
title: 功耗模型
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/fleet-sim/power-model.md"
  outdated: false
---

# 功耗模型

Fleet Sim 可以在已校准的性能 profile 上叠加估计功耗曲线。这支持两类规划问题：

- 每焦耳输出 token 如何随池组成和利用率变化；
- 并发上限如何在建模板级功耗与排队延迟之间取舍。

功耗结果是估计值。DES 可以校验场景的延迟一侧，但不会测量或独立验证功耗曲线。

## 功耗模型变体 {#power-model-variants}

### 手工 profile {#manual-profile}

`ManualProfile` 可以定义：

- `power_idle_w`：低并发端点的建模板级功耗；
- `power_nominal_w`：高并发端点的建模板级功耗；
- `power_logistic_k`：曲线陡度；以及
- `power_logistic_x0`：`log2(concurrency)` 轴上的中点。

当 `power_logistic_k` 大于零时，并发为 `b` 时的功耗为：

```text
P_range = P_nominal - P_idle
P(b) = P_idle + P_range / (1 + exp(-k * (log2(max(1, b)) - x0)))
```

当 `k` 为零时，模型用 `b / max_slots` 在 `P_idle` 和 `P_nominal` 之间线性插值。

两种变体都用活跃序列作为负载变量。它们不建模时钟状态、温度、内核组合、主机功耗、网络或冷却。

### 计算 profile {#computed-profile}

`ComputedProfile` 根据硬件 TDP 以及 profile 的 KV 流量和 tensor-core 活动估计功耗。它在建模活动上升时，在 TDP 的固定比例之间插值。

这是粗粒度的迁移模型，不能替代目标 GPU 上的批次与功耗测量。即使硬件规格本身正确，把相同的 TDP 比例套到另一种架构、模型、精度或并行布局也会增加不确定性。

## 内置 profile 边界 {#built-in-profile-boundary}

CLI 包含名为 `h100`、`a100` 和 `a10g` 的手工 profile。它们让示例可以运行，但其常量应视为可编辑的规划假设：

- 价格是源码中的静态值，不是实时云价格源；
- 性能和功耗曲线不是你部署上的测量结果；
- H100/A100 profile 意在代表 70B 级、多 GPU 布局，而 A10G 代表更小的单 GPU 模型；以及
- profile 名称并不编码所有影响功耗的服务参数。

因此，默认的 `h100` 与 `a10g` 每瓦 token 结果比较的是两套建模系统（包含模型规模）。它并不能证明某一 GPU 对同一模型更高效。

## 每瓦 token {#tokens-per-watt}

对单个同构池，Fleet Sim 用到达率和平均输出长度估计输出吞吐，再除以建模池功耗：

```text
pool_output_tokens_per_second = lambda_pool * mean_output_tokens
pool_power_watts = N_pool * P(mean_active_sequences)
pool_tokens_per_watt = pool_output_tokens_per_second / pool_power_watts
```

对多池机队，正确的汇总是：

```text
fleet_tokens_per_watt =
  sum(lambda_i * mean_output_tokens_i) /
  sum(N_i * P_i(mean_active_sequences_i))
```

在固定工作点上，单个同构池的机队规模可以在代数上约掉。对不同规模、流量比例或功耗曲线的池，它不会约掉。

CDF 工作流用默认的总 token 占比 20% 估计平均输出长度。如果你的输出比例不同，即使功耗曲线不变，每瓦 token 也可能变化。

## 运行能耗研究 {#run-an-energy-study}

单池模式会把全部工作负载应用到每个所选 profile：

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpus h100 a100 \
  --rho-sweep \
  --out energy.json
```

只有在为同一模型、精度、并行布局、上下文和 vLLM 配置创建 profile 之后，才把它当作硬件比较。

双池模式把路由拓扑与同构长池基线比较：

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a10g \
  --gpu-long h100 \
  --out routed-energy.json
```

这种形式下，模型切换可以是有意的。请把它报告为路由、模型和硬件的组合比较，并单独验证答案质量。

## 并发上限分析 {#concurrency-cap-analysis}

`grid-flex` 在固定机队上降低建模并发上限：

```bash
vllm-sr-sim grid-flex \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --n-gpus 32 \
  --gpu h100 \
  --slo 500 \
  --flex-pcts 0 10 20 30 \
  --verify-des 20000 \
  --out flex.json
```

对每个请求的降幅，Fleet Sim：

1. 相对 `power_nominal_w` 计算每个 profile 的目标功耗；
2. 反解功耗曲线以选择并发上限；
3. 在该上限下重新校准解析队列；
4. 估计 P99 TTFT；以及
5. 可选地用 `--verify-des` 运行 DES。

请求的百分比可能被空闲功耗下限截断，因此请查看报告的瓦数，不要假定恰好达到目标。

该命令不会更改 `max_num_seqs`、控制在线 vLLM 服务器，或与电力系统通信。它为外部控制器设计生成一条建模权衡曲线。

## 校准 profile {#calibrate-a-profile}

先校准性能，再校准功耗；若吞吐模型错误，每瓦 token 没有意义。

1. 固定模型、精度、张量并行、vLLM 版本、上下文组合、时钟和功耗上限。
2. 在多个稳态负载下测量 TTFT、token 吞吐、活跃序列和 KV 容量。拟合 `W`、`H`、`calibration_ctx` 和 `max_slots`。
3. 在相同负载下用平台遥测记录板级功耗，包括低并发和最高可持续工作点。
4. 用这些点拟合线性或 logistic 曲线。不要把 TDP 当作测得的标称值。
5. 在留出负载和生产形态的 token 分布上验证。
6. 将测量日期和环境与 profile 一起保存；运行时或模型发生实质变化后重新测量。

可以显式构造源码定义的手工 profile：

```python
from fleet_sim.gpu_profiles.manual import ManualProfile

profile = ManualProfile(
    name="measured-model-on-target-gpu",
    W=0.006,
    H=0.0004,
    calibration_ctx=8192,
    chunk=512,
    blk_size=16,
    total_kv_blks=50000,
    max_slots=96,
    cost_per_hr=0.0,
    power_idle_w=180.0,
    power_nominal_w=360.0,
    power_logistic_k=0.9,
    power_logistic_x0=3.5,
)
```

上面的数字只说明必填字段，不是任何设备的推荐值。

## 报告清单 {#reporting-checklist}

能耗结果应说明：

- 瓦数指 GPU 板级功耗还是整机/机房功耗；
- 模型、dtype、并行布局、运行时、时钟和功耗上限；
- 工作负载和输出长度分布；
- 测量来源和拟合曲线误差；
- 每个池的机队数量和利用率；
- 延迟是解析、DES 还是负载测试测得的；以及
- 任何未测量 profile 引入的不确定性。

在未计入主机、网络、存储、冷却、功率转换、工作负载迁移和真实控制器行为之前，不要把建模的板级功耗节省变成机房能耗、排放或需求响应承诺。
