---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/latency-aware.md"
  outdated: true
---

# 按延迟选择

## 概览

要选择最快的可用候选，只在延迟上为 [Multi Factor](./multi-factor) 加权。它按配置的百分位上观测到的延迟为候选排序，并选择最快的一个。这取代了已移除的 `latency_aware` 选择器。

## 主要优势

- 使用对该路由重要的延迟百分位来比较候选。
- 无需管理模型状态：完全由运行时指标驱动。
- 之后可以在不改变算法类型的情况下加入 SLO 上限、成本或负载。

## 它解决什么问题？

有些路由更在意响应速度而不是模型的绝对质量。只按延迟加权的 `multi_factor` 使用观测到的延迟统计，而不是静态假设，来比较已经可用的候选。

## 何时使用

- 路由有多个可行候选，且应由延迟决定胜出者。
- 你希望得到快速响应，又不想硬编码某一个模型。

## 配置

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights:
      latency: 1
    latency_percentile: 90
```

`latency_percentile` 选择比较每个候选时使用的观测延迟百分位。默认情况下，`multi_factor` 比较每个输出 token 的时间（TPOT）；当某个候选没有 TPOT 样本时，比较其首 token 时间（TTFT）。设置 `latency_metric: ttft` 或 `tpot` 可只比较一种测量值。延迟观测值只在单个 Router 进程内有效；它不是集群级的调度器。SLO 上限和其他因素见 [Multi Factor](./multi-factor)。

### 从 latency_aware 迁移

`vllm-sr config migrate` 把 `latency_aware` 改写为这种 `multi_factor` 加权。只设置了一个百分位的块会保留其度量（`latency_metric`）和百分位。同时设置了 `tpot_percentile` 和 `ttft_percentile` 的块会在 TPOT 百分位上比较 TPOT，并附带一条需要复核的说明：`multi_factor` 只比较一种测量值，而 `latency_aware` 会对两者取平均。

可复用片段：[`config/fragments/algorithm/selection/latency-aware.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/latency-aware.yaml)。
