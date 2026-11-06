---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/overview.md"
  outdated: false
---

# 算法

## 概览

算法在决策匹配之后运行。**选择器**（selector）在调用之前从决策的 `modelRefs` 中选出一个模型。**执行器**（executor）调用多个候选并返回一个响应：逐级升级、运行评审组，或执行工作流。
它并不决定路由是否可用；那是信号和决策先完成的事。

## 主要优势

- 把路由的可用性与模型选择分开。
- 让选择与编排策略在每个决策上都可审查。
- 同时支持单次调用的选择和有界的多模型执行。

## 它解决什么问题？

一个匹配的路由可能有多个有效的候选模型。算法让选择变得显式：固定顺序、质量、延迟和成本等运行时因素、决策模型的回答，或多模型执行。

## 何时使用

当决策有多个候选，或有意运行多模型工作流时，添加算法。只有一个候选时，除非所选执行器支持并需要单模型执行计划，否则省略算法。

## 配置

算法是决策本地的。每种类型的设置都是以该类型命名的块：

```yaml
routing:
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

`minimum_candidates` 是一个通用算法字段。不绑定模型的配方可以在绑定任何模型之前声明它；一旦入口实例化该配方，校验就要求至少有这么多个不同的 `modelRefs`。在请求时按上下文进行可用性过滤之后，会再次检查同一边界，因此评审组或选择器不会在比策略声明更小的候选池上悄悄运行。

## 算法清单

### 选择器

选择器返回一个候选模型。对于完整的推理请求，Router 在打分**之前**按上下文、后端协议支持和声明的模型任务能力过滤精确的候选引用。所配置的选择器比较剩下的候选池；它不会先选出一个不具备能力的胜出者，再用第一个兼容的同级候选替换它。

能力检查包括 Router 保留的对话内容，并会预览决策的请求参数和无工具策略，而不会把这些插件执行两次。未标注的模型只保留协议兼容性检查。严格的质量/SLO 约束仍然适用，Router Learning 在考虑其他候选时也检查相同的能力。

空的候选池或违反 `minimum_candidates` 要求会以失败关闭。分派会在修改之后再次校验最终请求：迟到的能力不匹配会返回错误，而不是重新评估决策、重新打分或悄悄更换所选候选。显式指定的模型使用同样的最终校验，但不会被其他模型替换。选择器失败时回退到第一个候选。

| 类型 | 状态 | 目标 | 主要依赖 | 指南 |
|---|---|---|---|---|
| `static` | supported | 使用声明的顺序或固定的领域分数 | 无 | [Static](./selection/static) |
| `random` | supported | 在可用的声明候选中均匀随机选择 | 无 | [Random](./selection/random) |
| `multi_factor` | supported | 在可选 SLO 过滤下平衡质量、延迟、成本和负载 | 模型元数据和本地实时指标 | [Multi Factor](./selection/multi-factor)，[按延迟](./selection/latency-aware) |
| `systemone` | supported | 询问决策模型应由哪个候选作答 | 支持 Choice 的 System One 部署 | [System One](./selection/systemone) |

### 执行器

执行器会发起额外的模型调用。Router 在进程内完成每次调用：调用会运行所匹配决策的插件，并发往被调用模型的 `providers.models[].backend_refs`，因此执行器调用的每个模型都需要后端。这些调用会增加延迟和 token 用量，并且中间内容会发送给参与运行的每个已配置工作模型。

每个执行器都读取与类型并列的两个共享设置：

- `budget` 约束整次运行：`deadline` 是其墙钟时间上限，`max_calls` 是它可发起的模型调用次数，且低于 Router 的单请求上限。耗尽预算的运行会失败，而不会返回部分答案。
- `on_error` 说明调用失败时如何处理。`fusion`、`remom`、`ratings` 和 `workflows` 接受 `skip`（使用成功的调用继续）或 `fail`。`escalate` 接受 `last_attempt`（没有回答被接受时返回最后一个经过评判的回答）或 `fail`。

`global.integrations.looper.endpoint` 已弃用并被忽略，因为执行器调用不再通过网关回环。请删除它，或运行 `vllm-sr config migrate`。

| 类型 | 状态 | 目标 | 指南 |
|---|---|---|---|
| `escalate` | supported | 按顺序调用候选，直到某个回答被接受；适用于 Chat 或 System One | [Escalate](./looper/escalate) |
| `ratings` | supported | 在有界并发下从每个候选返回一个选择 | [Ratings](./looper/ratings) |
| `remom` | supported | 多轮探索多条推理路径，然后合成 | [ReMoM](./looper/remom) |
| `fusion` | experimental | 运行分析评审组和评审/合成步骤 | [Fusion](./looper/fusion) |
| `workflows` | experimental | 执行有界的静态或由规划器生成的工作流 | [Router Flow](./looper/workflows) |

请把实验性算法视为评估功能：在用于生产路由之前，先在你的流量上验证它们。

### System One 请求

以 `api: systemone` 发布的配方使用原生执行。其匹配的决策运行带 `escalate.stages` 的 `escalate`，保持完整的类型化响应。请求截止时间和物理调用账本在请求到达时开始。选中决策之前，上限取配方中各原生决策最大的 `algorithm.budget` 截止时间和 `max_calls`；各信号自身的超时仍是上限。选中之后，该决策的 `algorithm.budget` 为其阶段、评审和传输重试收紧同一个截止时间和账本。已用于信号的调用计入该决策的 `max_calls`。原生执行使用显式的候选名单和质量约定；它不会运行 Chat 插件，也不会悄悄回退到 Chat 模型。

### 已移除的算法

这些类型已被上面的选择器和执行器取代。仍然使用它们的配置会在加载时失败，并提示运行 `vllm-sr config migrate` 进行改写：

| 已移除的类型 | 变为 |
|---|---|
| `confidence`（Looper） | 带 `escalate.accept.confidence` 的 `escalate` |
| `cascade`（System One） | 带 `escalate.stages` 的 `escalate` |
| `prompt`、`router_dc` | `systemone` |
| `knn`、`kmeans`、`svm`、`mlp` | 不需要训练模型的 `systemone` |
| `automix`、`hybrid`、`latency_aware` | `multi_factor` |

迁移会记录需要人工复核的内容，例如被丢弃的权重或不再使用的训练工件。

## 运行边界

- 当算法接受重复的模型引用时，候选包含其 LoRA 和推理控制，而不仅仅是模型名。打分、组合和分派都保留精确的胜出引用。与多个不同候选匹配的旧式仅模型名结果会被拒绝，而不是解析为第一个引用。对于 LoRA 候选，服务前端按所选的基础模型路由，而提供商请求使用适配器的名称。
- Router Learning 的会话记忆保留所选候选的控制项。即使之后的基础选择偏好同一模型的另一种推理强度，保护机制也可以在工具循环续接中保持这一精确选择。如果在仍处于接纳它的决策上的活动工具循环中，或在不可移植的续接中，该精确所有者被排除，路由会拒绝请求，而不是恢复该所有者或回退到不同的推理强度。决策漂移会释放工具循环锁，并从新决策的候选中选择；不透明的提供商状态仍绑定到创建它的模型。可移植的轮次可以选择新的可用候选；观察和旁路模式不会强制执行保护决定。
- 在完整配置中，候选模型名必须能通过 `routing.modelCards` 和 `providers.models` 解析。
- 延迟和负载观测值只在单个 Router 进程内有效；它们不是集群级的调度器。
- 执行器会与其配置的工作模型共享请求内容。请在选择执行器之前先应用隐私和提供商边界决策。
- 执行器生成的规划、工作、校验、评审和合成提示词会在分派前对照每个目标模型的已知上下文窗口进行检查。缺少上下文元数据的模型仍保持兼容可用。
- 使用 `vllm-sr config validate --config config.yaml` 校验完整配置。
