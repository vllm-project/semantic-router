---
translation:
  source_commit: "bb72437b"
  source_file: "docs/tutorials/algorithm/selection/multi-factor.md"
  outdated: false
---

# 多因素

## 概览

`multi_factor` 是一种选择算法，它将四种原始运行时信号，即**质量**、**延迟**、**成本**和**负载**，合成为每个候选项的单一加权分数；还可选择设置 SLO 硬上限，在评分前剔除候选项。

配置归属于声明它的决策，并且各决策的配置相互隔离。如果多个决策使用 `multi_factor`，则每个匹配到的决策都会使用各自的权重、SLO、分位数和无候选项策略进行评估。

它与 `config/fragments/algorithm/selection/multi-factor.yaml` 保持一致，并解决了议题 [#37](https://github.com/vllm-project/semantic-router/issues/37)。

## 主要优势

- 无需编排多个选择器，即可在单个决策中实现 SLO 感知路由。
- 每种信号都有实时数据源：质量来自 `quality_score` 配置，延迟来自 `pkg/latency` 的分位数，成本来自定价，负载来自 `pkg/inflight`。
- 在候选集内进行最小-最大归一化，因此无论信号的绝对尺度如何，权重都具有直观含义。
- 没有需要训练的模型状态，也不需要外部服务。
- 硬性 SLO 上限（TPOT、TTFT、成本、进行中请求数）会在评分前剔除不安全的候选项。

## 它解决什么问题？

实际路由会同时关注多个维度：候选池中可能既有更快、更便宜的模型，也有更慢但更好的模型；哪一个才是“正确”选择，取决于当前负载和 SLO 目标，而不只是静态配置。现有的单信号选择器（`latency_aware`、仅按成本路由、仅按质量路由）迫使用户作出非此即彼的选择。`multi_factor` 让一条决策规则能够在全部四个维度间表达平滑的权衡，还可通过硬性 SLO 上限排除不安全的候选项。

## 何时使用

- 一条决策有 2+ 个候选模型，它们在多个维度上存在差异（例如，一个模型更快且更便宜，另一个更慢但更好），而你希望用一个旋钮平滑调节取舍。
- 你希望强制执行 SLO（例如，“绝不路由到 p95 TPOT > 200ms 的模型”），但不想另写一条决策规则。
- 质量、延迟、成本和负载都很重要，且没有任何一个维度占据绝对主导地位。

## 同类算法

- `latency_aware` 是它的一个特例，即仅按延迟评分。当其他维度确实无关紧要时，请使用该算法。
- `hybrid` 将请求时选择器和只读学习证据合成为
  一个分数。`multi_factor` 则直接合成原始运行时信号。两者
  都有用且互为补充。

## 算法原理

对于候选集中的每个候选模型 $m$，在经过 SLO 筛选后：

$$\text{score}(m) = w_Q \cdot \hat{Q}(m) + w_L \cdot (1 - \hat{T}(m)) + w_C \cdot (1 - \hat{C}(m)) + w_{\text{load}} \cdot (1 - \hat{N}(m))$$

其中：

- $\hat{Q}(m)$、$\hat{T}(m)$、$\hat{C}(m)$、$\hat{N}(m)$ 分别是质量、延迟、成本和负载值，**在经过筛选后保留的候选集内通过最小-最大归一化映射到 [0, 1]**。
- 延迟、成本和负载经过反转（`1 - ...`），因为这些值越低越好。
- 质量不反转，因为质量越高越好。
- 权重会归一化，使其总和为 1（负权重会被截断为零）。恢复时默认采用等权重。

## SLO 筛选

评分前，任何超过非零上限的候选项都会被移除：

- `max_tpot_ms` — 通过 `pkg/latency` 观测到的 p95（或配置的分位数）TPOT
- `max_ttft_ms` — 通过 `pkg/latency` 观测到的 p95（或配置的分位数）TTFT
- `max_cost_per_1m` — 配置的提示定价
- `max_inflight` — 来自 `pkg/inflight` 的当前进行中请求数

如果所有候选项都被筛除，则行为由 `on_no_candidates` 控制：

| 值 | 行为 |
| --- | --- |
| `cheapest`（默认） | 返回配置的 `prompt_per_1m` 最低的候选项 |
| `first` | 返回所列的第一个候选项 |
| `fail` | 向调用方返回错误 |

## 配置

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
    slo:
      max_tpot_ms: 200       # 可选；省略则不设上限
      max_ttft_ms: 800       # 可选
      max_cost_per_1m: 5.0   # 可选；每 1M 个提示词元的成本（USD）
      max_inflight: 50       # 可选
    latency_percentile: 95   # 要读取的分位数（默认 95）
    on_no_candidates: cheapest
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `weights.quality` | float | `0.25` | 为每个模型配置的 `quality_score` 的权重 |
| `weights.latency` | float | `0.25` | 分位延迟的权重（越低越好，经过反转） |
| `weights.cost` | float | `0.25` | 提示定价的权重（越低越好，经过反转） |
| `weights.load` | float | `0.25` | 进行中请求数的权重（越低越好，经过反转） |
| `slo.max_tpot_ms` | float | `0`（关闭） | p95 TPOT 的硬上限，单位为毫秒 |
| `slo.max_ttft_ms` | float | `0`（关闭） | p95 TTFT 的硬上限，单位为毫秒 |
| `slo.max_cost_per_1m` | float | `0`（关闭） | 每 1M 个词元的提示成本硬上限 |
| `slo.max_inflight` | int | `0`（关闭） | 并发进行中请求数的硬上限 |
| `latency_percentile` | int | `95` | 从 `pkg/latency` 读取的分位数（1-100） |
| `on_no_candidates` | string | `cheapest` | SLO 筛除所有候选项时的回退策略：`cheapest`、`first`、`fail` |

## 已知限制

- 质量评分依赖于为每个模型配置 `quality_score`。未配置该项的模型对质量信号的贡献为零。
- 最小-最大归一化是**针对每次请求在候选集内进行的**，因此任何信号的绝对尺度都不重要；但如果所有候选项在某个维度上的值都相同，则该维度贡献 0.5（中性值）。
- 负载使用进程内跟踪器（`pkg/inflight`），因此在多副本部署中，每个副本只能看到自身的负载，而非整个集群的负载。对于典型的边车部署，这是可以接受的；未来可以接入外部状态存储，以实现真正的集群级负载感知。
- 进行中请求跟踪器通过 TTL 驱逐实现自愈（默认 10 分钟），以便从遗漏的 `End` 调用中恢复；但它无法识别运行时间超过该窗口且仍在执行的长请求，这些请求在选择器看来将是“空闲”的。如果你的工作负载经常出现单个请求超过 10 分钟的情况，请通过 `pkg/inflight.SetMaxAge` 调整。
