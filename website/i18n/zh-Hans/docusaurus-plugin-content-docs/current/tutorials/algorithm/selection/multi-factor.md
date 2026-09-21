---
translation:
  source_commit: "2b7519a84aec96963b02a3534e82908beba33f76"
  source_file: "docs/tutorials/algorithm/selection/multi-factor.md"
  outdated: false
---

# 多因子选择

## 概览

`multi_factor` 根据质量、延迟、成本和负载选择一个候选。硬资格规则先运行；幸存候选再用加权或字典序目标比较。

| 因素 | 来源 | 方向 |
| --- | --- | --- |
| 质量 | 版本化 Overall、能力或运维人员索引 | 越高越好 |
| 延迟 | 所选百分位的 TTFT 或 TPOT 观测值 | 越低越好 |
| 成本 | 按本次请求 token 预算应用的输入/输出定价 | 越低越好 |
| 负载 | 该 Router 进程中当前进行中的请求 | 越低越好 |

质量按候选的确切推理 effort 解析。绝不会借用不同 effort 的分数。

设置 `latency_metric: ttft` 优先缩短首字等待时间，设置 `tpot` 优先提高开始生成后的输出速度。
指定指标后，缺失的观测值保持未知，不会用另一种指标代替。只有当前剩余候选都具有观测值时，
字典序选择才比较延迟；完全没有观测或只有部分观测时，会跳过延迟，继续比较下一个优先级。
模型不会仅因为启动后先收到请求、先产生测量，就排除尚未测量的候选。
省略此设置时保留原有的 TPOT 优先、缺失时使用 TTFT 的行为。

## 解决什么问题？

模型池里常常同时有更强、更便宜和更快的模型。该选择器让资格保持显式，并把这种权衡变成一条可审计的决策策略。

## 何时使用

当决策至少有两个候选模型，并且必须优化质量/延迟/成本/负载权衡时，使用 `multi_factor`。当顺序本身就是策略时使用 `static`；当延迟是唯一选择信号时使用 `latency_aware`。

## 配置

### 均衡

默认的 `weighted` 策略对幸存池中每个可用因素做 min-max 归一化，并计算：

$$
S(m)=w_Q\hat Q(m)+w_T(1-\hat T(m))+w_C(1-\hat C(m))+w_L(1-\hat L(m))
$$

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    objective:
      strategy: weighted
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
      min_coverage: 1.0
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

权重归一化后和为一。如果每个配置权重都是零，选择器会恢复为等权。

### 准确率优先

`lexicographic` 按顺序应用优先级。每个阶段保留落在最佳观测值所声明相对容差内的候选，再把该带传给下一阶段。

延迟阶段要求当前候选全部具有观测值。这不会放宽质量资格或成本限制；缺少质量和成本证据时仍遵循各自原有的规则。

只有通过全部优先级筛选的候选才能参与后续自适应、会话防护和派发。这些步骤不能重新引入被先前质量或成本容差排除的模型。记录的分数仍可包含已排除模型，用于解释选择结果。

Preview 通过 `selection_trace` 返回字典序选择过程，Insights 在 `route_diagnostics.selection_trace` 中保存同样的结构：实际执行到的阶段、测量覆盖率、可用值、跳过原因、淘汰原因及最终保留的候选。未知值保持缺失，真实零值保留为零。这些是 Router Learning 之前的基础选择证据，实际选中的模型单独报告。实时测量可能在 Preview 后变化，因此 trace 用于解释当时的选择，不保证之后仍选择同一模型。

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    objective:
      strategy: lexicographic
      priorities:
        - {factor: quality, tolerance: 0.03}
        - {factor: cost, tolerance: 0.05}
        - {factor: latency, tolerance: 0.05}
    quality:
      index: vllm-sr/intelligence@1.0.0
      on_missing: exclude
      min_coverage: 1.0
```

这会保留质量分数在最佳值 3% 以内的模型，然后在该带内选择更便宜、更快的候选。

### 成本优先并设质量下限

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/intelligence@1.0.0
      on_missing: exclude
      min_coverage: 1.0
      min_score: 65
    objective:
      strategy: lexicographic
      priorities:
        - {factor: cost, tolerance: 0.05}
        - {factor: quality, tolerance: 0.03}
        - {factor: latency, tolerance: 0.05}
```

质量下限会在优化前排除低于 65 的模型。目标随后保留成本估计在最便宜值 5% 以内的模型，并在该成本带内选择最佳质量。

`weights` 不能与字典序目标组合。一个优先级因素只能出现一次。

## 质量资格与缺失数据

```yaml
quality:
  index: acme/clinical-quality@1.0.0
  on_missing: exclude
  min_coverage: 0.5
  min_score: 60
```

- `index` 选择版本化的内置或运维人员索引。
- `min_coverage` 在索引自身的缺失数据策略之后应用。它可以让路由比索引定义更严格。
- `min_score` 是索引声明量表上的硬下限，并要求 `on_missing: exclude`。
- `exclude` 会移除没有合格的精确 effort 证据的候选。
- `disable_quality` 保留池，但一个缺失候选会禁用整次比较的质量。它从不为单个模型改权重。

内置层级见 [Open Intelligence Index](../../../benchmarking/open-intelligence-index)，运维人员定义的证据见 [Custom evaluations](../../../benchmarking/custom-evaluations)。

## 成本与 SLO

输入成本按请求的预路由 token 估计计算。输出成本在调用方提供时使用 `max_output_tokens`。如果两者都不知道，选择器回退到配置的每百万 token 费率。同一请求混合用于成本因素、`max_cost_per_1m` 和 `cheapest` 回退。

```yaml
multi_factor:
  slo:
    max_tpot_ms: 200
    max_ttft_ms: 800
    max_cost_per_1m: 5.0
    max_inflight: 50
  latency_percentile: 95
  on_no_candidates: cheapest
```

仅当对应观测可用时，才强制 SLO 上限。
如果每个候选都被排除，`on_no_candidates` 选择 `cheapest`、`first` 或 `fail`。`fail` 是严格策略：Router 返回 HTTP 503，绝不会替换成第一个配置候选。Eval 干跑会对同一请求把选择报告为不可用。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `objective.strategy` | `weighted` | `weighted` 或 `lexicographic` |
| `objective.priorities[].factor` | — | `quality`、`latency`、`cost` 或 `load` |
| `objective.priorities[].tolerance` | `0` | 相对最佳值的带，范围 0 到 1 |
| `quality.index` | 目录默认值 | 版本化质量索引 |
| `quality.on_missing` | 显式时为 `exclude` | `exclude` 或池范围的 `disable_quality` |
| `quality.min_coverage` | 索引策略 | 允许的最低证据覆盖率，范围 0 到 1 |
| `quality.min_score` | 关闭 | 索引量表上的硬质量下限 |
| `weights.*` | `0.25` | 均衡的质量、延迟、成本和负载权重 |
| `latency_percentile` | `95` | 观测延迟百分位，范围 1 到 100 |
| `latency_metric` | TPOT，其次 TTFT | 统一使用 `ttft` 或 `tpot` 比较候选 |
| `on_no_candidates` | `cheapest` | `cheapest`、`first` 或 `fail` |

延迟和负载观测是每个 Router 进程本地的。因此副本可能做出不同选择。需要集群级容量时，使用共享遥测或基础设施路由器。

完整片段见：
[`config/fragments/algorithm/selection/multi-factor.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/multi-factor.yaml)。
