---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/adaptations.md"
  outdated: false
---

# 自适应（Adaptation）

## 概览

自适应（Adaptation）是一种在线模型选择学习机制。它在匹配的 decision 和基础选择器之后运行，然后从允许的候选集中提出一个模型。

## 主要优势

- 无需重写请求路径上的语义 decision，即可改进模型选择。
- 通过 `candidate_set: decision` 保持较小的默认搜索空间。
- 启用 `candidate_set: tier` 后，可以在同一路由 tier 内共享证据。
- 在随机采样之前以及最终切换模型之前使用 Protection。
- 写入紧凑的响应头和详细的 replay 诊断，供离线分析使用。

## 解决什么问题？

静态配方编码了运维人员在部署时掌握的信息。在生产环境中，路由器还会观察模型适配度、过度使用、提供商故障、延迟、缓存复用和实际成本。Adaptation 将这些有界证据转化为在线模型提案，同时配方仍是策略的事实来源。

## 何时使用

- 某个 decision 有多个候选模型，并且运行时结果能够改进模型选择。
- 相关 decision 共享一个 tier，并且应相互学习模型证据。
- 希望进行在线探索，但仅限于 Protection 判定当前 agent 状态可以安全探索时。
- 希望离线评估为模型经验提供初始种子，而无需立即更改配方。

## 配置

首个策略是 `routing_sampling`：

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: true
        strategy: routing_sampling
        candidate_set: decision
```

## 候选集

| 值 | 候选模型 |
| --- | --- |
| `decision` | 来自匹配 decision 的 `modelRefs` 的模型。 |
| `tier` | 具有相同 `decision.tier` 的 decision 的 `modelRefs` 并集。 |
| `global` | 配方的模型/提供商清单中的所有已部署模型。 |

`decision` 是最安全的默认值。`tier` 允许相关路由共享候选模型。`global` 的范围最广，可以提出未出现在匹配 decision 的 `modelRefs` 中的已部署模型，因此会使用更严格的成本和可靠性保护条件。

## 路由采样

`routing_sampling` 根据模型经验为每个候选模型评分：

- 离线质量种子或中性质量种子
- `good_fit`、`underpowered`、`overprovisioned` 和 `failed` 结果
- 延迟证据
- 缓存复用证据
- 实际输入成本
- 可靠性证据

当 Protection 允许探索时，该策略可以从候选模型的后验分布中采样。当 Protection 抑制探索时，该策略会按确定性的后验均值评分。

Protection 仍拥有最终决定权。通过采样或均值选出的候选模型必须通过切换保护条件，才能成为最终模型。

## 诊断

响应头保持紧凑：

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=keep_base,protection=hold_current
x-vsr-learning-reasons: adaptation=base_best,protection=cache_cost_high
```

Router Replay 会存储候选模型评分、后验均值、采样值、基础模型、提案模型、最终模型、候选集、策略和原因。
