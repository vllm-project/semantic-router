---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/algorithm/overview.md"
  outdated: false
---

# 算法 {#algorithms}

## 概览 {#overview}

算法在决策匹配之后运行。**选择器**（selector）在调用之前从决策的 `modelRefs` 中选出一个模型。**执行器**（executor）调用多个候选并返回一个响应：逐级升级、运行评审组，或执行工作流。
它并不决定路由是否可用；那是信号和决策先完成的事。

## 主要优势 {#key-advantages}

- 把路由的可用性与模型选择分开。
- 让选择与编排策略在每个决策上都可审查。
- 同时支持单次调用的选择和有界的多模型执行。

## 它解决什么问题？ {#what-problem-does-it-solve}

一个匹配的路由可能有多个有效的候选模型。算法让选择变得显式：固定顺序、质量、延迟和成本等运行时因素、决策模型的回答，或多模型执行。

## 何时使用 {#when-to-use}

当决策有多个候选，或有意运行多模型工作流时，添加算法。只有一个候选时，除非所选执行器支持并需要单模型执行计划，否则省略算法。

## 配置 {#configuration}

算法是决策本地的。每种类型的设置都是以该类型命名的块：

```yaml
recipes:
  - name: default
    decisions:
      - name: responsive-route
        description: Prefer the model with the best observed latency.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small-model
          - model: large-model
        algorithm:
          type: multi_factor
          minimum_candidates: 2
          multi_factor:
            weights:
              latency: 1
            latency_percentile: 90
```

从下面的清单中选择算法，然后按其指南配置必填字段和依赖。

`minimum_candidates` 是一个通用的算法字段：

- 不带模型的配方可以在绑定任何模型之前声明它。
- 入口发布该配方后，校验要求至少有这么多个不同的 `modelRefs`。
- Router 在请求时按上下文过滤之后会再次检查，因此评审组或选择器绝不会在比策略声明更小的候选池上悄悄运行。

## 算法清单 {#algorithm-inventory}

### 选择器 {#selectors}

选择器返回一个候选模型。打分之前，Router 会剔除无法处理请求的候选：上下文不足、后端协议格式不兼容，或缺少声明的能力。选择器随后只比较剩下的模型。

- **检查读取什么。** 检查会包含 Router 保留的对话内容，并预览决策的请求参数和无工具策略，而不会把这些插件执行两次。
- **未标注的模型**只保留协议兼容性检查。严格的质量和 SLO 约束仍然适用，Router Learning 也检查相同的能力。
- **失败关闭。** 候选池为空或少于 `minimum_candidates` 时，请求失败。
- **最终检查。** 插件修改请求之后，分派会再次校验最终请求。迟到的能力不匹配会返回错误，而不是重新打分或悄悄换模型；显式指定的模型绝不会被替换。

选择器失败时回退到第一个候选。

| 类型 | 状态 | 目标 | 主要依赖 | 指南 |
|---|---|---|---|---|
| `static` | supported | 使用声明的顺序或固定的领域分数 | 无 | [Static](./selection/static) |
| `random` | supported | 在可用的声明候选中均匀随机选择 | 无 | [Random](./selection/random) |
| `multi_factor` | supported | 在可选 SLO 过滤下平衡质量、延迟、成本和负载 | 模型元数据和本地实时指标 | [Multi Factor](./selection/multi-factor)，[按延迟](./selection/latency-aware) |
| `systemone` | supported | 询问决策模型应由哪个候选作答 | 支持 Choice 的 System One 部署 | [System One](./selection/systemone) |

### 执行器 {#executors}

执行器在进程内发起额外的模型调用。每次调用都会运行所匹配决策的插件，并发往被调用模型的 `providers.models[].backend_refs`，因此执行器调用的每个模型都需要后端。

:::warning 执行器成本更高，并会共享内容
额外的调用会增加延迟和 token，中间内容也会发给参与运行的每个工作模型。请先应用隐私和提供方边界决策。
:::

每个执行器都读取与类型并列的两个共享设置：

| 设置 | 含义 |
| --- | --- |
| `budget` | `deadline` 是整次运行的墙钟时间上限，`max_calls` 是它可发起的模型调用次数，且低于 Router 的单请求上限。耗尽预算的运行会失败，而不会返回部分答案。 |
| `on_error` | 调用失败时如何处理。`fusion`、`remom`、`ratings` 和 `workflows` 接受 `skip`（使用成功的调用继续）或 `fail`。`escalate` 接受 `last_attempt`（没有回答被接受时返回最后一个经过评判的回答）或 `fail`。 |

要为执行器提供独立的公开模型名，请发布一个入口，让它的 `model_names` 指向包含运行该执行器的决策的配方。

| 类型 | 状态 | 目标 | 指南 |
|---|---|---|---|
| `escalate` | supported | 按顺序调用候选，直到某个回答被接受；适用于 Chat 或 System One | [Escalate](./looper/escalate) |
| `ratings` | supported | 在有界并发下从每个候选返回一个选择 | [Ratings](./looper/ratings) |
| `remom` | supported | 多轮探索多条推理路径，然后合成 | [ReMoM](./looper/remom) |
| `fusion` | experimental | 运行分析评审组和评审/合成步骤 | [Fusion](./looper/fusion) |
| `workflows` | experimental | 执行有界的静态或由规划器生成的工作流 | [Router Flow](./looper/workflows) |

请把实验性算法视为评估功能：在用于生产路由之前，先在你的流量上验证它们。

### System One 请求 {#system-one-requests}

以 `api: systemone` 发布的配方使用原生执行：其匹配的决策运行带 `escalate.stages` 的 `escalate`，返回完整的类型化响应。

- **预算从到达时开始。** 请求的截止时间和调用次数在请求到达时开始计算。选中决策之前，上限取配方中各原生决策最大的 `algorithm.budget`。
- **决策会收紧预算。** 选中之后，该决策自己的预算适用于它的阶段、评审和传输重试，已用于信号的调用也计入它的 `max_calls`。
- **不回退到对话模型。** 原生执行使用显式的候选名单和质量约定，从不运行对话插件，也不会回退到对话模型。

<details>
<summary>已移除的算法类型及其替代</summary>

仍然使用这些类型的配置会在加载时失败，`vllm-sr config migrate` 会改写它们。迁移会记录需要复核的内容，例如被丢弃的权重或不再使用的训练工件。

| 已移除的类型 | 变为 |
|---|---|
| `confidence`（Looper） | 带 `escalate.accept.confidence` 的 `escalate` |
| `cascade`（System One） | 带 `escalate.stages` 的 `escalate` |
| `prompt`、`router_dc` | `systemone` |
| `knn`、`kmeans`、`svm`、`mlp` | 不需要训练模型的 `systemone` |
| `automix`、`hybrid`、`latency_aware` | `multi_factor` |

</details>

## 运行边界 {#operational-boundaries}

- **候选是完整的引用。** 当算法接受重复的模型引用时，候选包含其 LoRA 和推理控制，而不仅仅是模型名。打分和分派都保留精确的胜出引用，与多个候选匹配的仅模型名结果会被拒绝。
- **LoRA 候选。** 前端按基础模型路由，而发往提供方的请求使用适配器的名称。
- **会话保护保持精确的选择。** Router Learning 记住所选候选的控制项，保护机制可以在工具循环中保持这一选择。如果它在循环中途，或在无法迁移的续接中被策略排除，请求会被拒绝，而不是换模型。见[保护](../learning/protection)。
- 在完整配置中，候选模型名必须能通过 `providers.models` 解析。
- 延迟和负载观测值只在单个 Router 进程内有效；它们不是集群级的调度器。
- 执行器生成的规划、工作、校验、评审和合成提示词会在分派前对照每个目标模型的已知上下文窗口进行检查。缺少上下文元数据的模型仍保持兼容可用。
- 使用 `vllm-sr config validate --config config.yaml` 校验完整配置。
