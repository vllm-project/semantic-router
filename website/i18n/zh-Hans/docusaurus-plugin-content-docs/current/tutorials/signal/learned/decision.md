---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/signal/learned/decision.md"
  outdated: false
---

# Decision 信号

## 概述

`decision` 向决策模型提出有关请求的类型化问题，并将答案转换为路由事实。问题使用自然语言编写，可以选择选项（`choice`）、回答是或否（`noul`），或在等级范围内评分（`score`）。支持相应能力的模型，例如 Vela 2.0，还能回答哪些标签适用（`set`）以及标签在原文中的位置（`span`）。模型在 Router 启动的[内置模型运行时](../../../model-runtime/overview.md)中运行。

## 主要优势

- 编写新问题即可使用，无需训练新的分类器。
- 答案包含概率，路由可以要求足够置信度。
- 共享部署且输入兼容的问题可以合并为一次请求阶段调用，包括 [`pii` 信号](pii.md#vela-20)。
- 回答超时或失败时，信号为未知，决策的失败策略决定继续路由还是拒绝请求。

## 解决什么问题？

固定标签分类器只能回答训练时定义的问题。决策模型可以根据问题文本判断“是否需要逐步推理”或“是否与我们的产品有关”。

## 何时使用

用于需要判断整个请求的问题。长度、关键词和模态等结构性事实优先使用启发式信号；domain、PII 和 jailbreak 等已有专用模型的问题优先使用相应的学习信号。

## 配置

没有指定 `deployment` 的问题使用 `global.model_catalog.system.decision_model`，默认为 Vela 2.0 0.3B，也可以[选择其他规模](../../../model-runtime/choose-a-model.md#choose-a-size)。它会与输入兼容的内置问题一起进入请求阶段批处理。复用部署可以避免加载另一份模型，但不保证整个路由请求只执行一次 forward。

```yaml
routing:
  signals:
    decision:
      - name: needs_tools
        question:
          type: noul
          instructions: Does answering this request need a tool call?
        predicate:
          gte: 0.7
```

绑定通过 `{deployment: primary}` 指定，可以选择 Vela 或 Decision 1.0/2.0。模型必须支持请求中的每种题型。[Decision 选模算法](../../algorithm/selection/decision.md)也使用同一个默认绑定，因此模型资源还可以用于选择后端。

要使用另一个模型，例如 Decision 2.0，先声明 `model_runtime` 部署，再为每个问题指定 `deployment`：

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto

routing:
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
        timeout_ms: 1000
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: math
              description: Mathematics or quantitative reasoning
            - key: chat
              description: Anything else
      - name: difficulty
        deployment: decision-kai
        question:
          type: score
          instructions: How difficult is this request?
          levels: [Trivial, Moderate, Hard]
        predicate:
          gte: 1.5

  decisions:
    - name: hard-code
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: decision
            name: request_kind
            label: code
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: large-coder
```

| 题型 | 匹配条件 | 路由可以读取的值 |
| --- | --- | --- |
| `noul` | 回答“是”的概率满足 `predicate`，默认 `gte: 0.5` | `decision:<name>` = P(yes) |
| `score` | 期望等级满足必填的 `predicate`，等级从 0 开始 | `decision:<name>` = 期望等级 |
| `choice` | 条件的 `label` 是选中的选项，设置 `predicate` 时还需满足概率条件 | `decision:<name>:<key>` = P(key)，`decision:<name>` = P(chosen) |
| `set` | 条件的 `label` 概率满足 `predicate`；未设置时由模型选择标签 | `decision:<name>:<key>` = P(key)，`decision:<name>` = 最高概率 |
| `span` | 条件的 `label` 对应文本片段满足 `predicate`；未设置时由模型识别片段 | `decision:<name>:<key>` = 该标签最高的片段概率，无片段时为 0；`decision:<name>` = 任一词的最高概率 |

条件可以指定自己的 `predicate`。带 `label` 的 `choice`、`set` 或 `span` 条件读取对应标签的值。

[投影分数](../../projection/scores.md)通过 `value_source: raw` 读取同样的值：`name: <question>` 读取 `decision:<name>`，`name: <question>:<key>` 读取 `choice`、`set` 或 `span` 的某个选项或标签。使用中的投影读取问题时，Router 会提出该问题：

```yaml
routing:
  signals:
    decision:
      - name: difficulty
        question:
          type: score
          instructions: How much reasoning does a strong expert need to answer well?
          levels: [none, a little, multi-step, expert]
        predicate:
          gte: 2
      - name: needs
        question:
          type: set
          instructions: What does a good answer need?
          labels:
            - key: deliberation
              description: a derivation, proof or careful step-by-step check
            - key: tools
              description: calling external tools or functions
  projections:
    scores:
      - name: effort
        method: weighted_sum
        inputs:
          - type: decision
            name: difficulty
            weight: 0.3
            value_source: raw
          - type: decision
            name: needs:deliberation
            weight: 0.4
            value_source: raw
    mappings:
      - name: effort_band
        source: effort
        method: threshold_bands
        outputs:
          - name: effort_high
            gte: 0.9
  decisions:
    - name: deliberate
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: projection
            name: effort_high
      modelRefs:
        - model: large-reasoner
```

### Set 和 span 问题 {#set-and-span-questions}

`set` 问题判断哪些标签适用，`span` 问题查找各标签在请求中的位置。两者使用 `labels` 而非 `choices`，每项包含 `key` 和可选的 `description`，相应条件指定标签：

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu

routing:
  signals:
    decision:
      - name: support_topics
        deployment: vela2
        question:
          type: set
          instructions: Which topics does the request mention?
          labels:
            - key: billing
              description: payments, invoices or refunds
            - key: shipping
              description: deliveries, tracking or returns
      - name: account_ids
        deployment: vela2
        question:
          type: span
          instructions: Which spans are account or order numbers?
          labels:
            - key: account_number
              description: a customer account or order number
          head: router

  decisions:
    - name: billing-with-account
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: decision
            name: support_topics
            label: billing
          - type: decision
            name: account_ids
            label: account_number
      modelRefs:
        - model: support-model
```

- `threshold`（0 到 1）覆盖模型对该问题的决策阈值。未设置时使用模型校准的阈值，决定选中的标签和识别的片段。
- `head`（仅适用于 `span`）选择模型提供的文本片段头：`router` 用于 PII、无依据声明和有害片段，`broad` 用于开放提取。未设置时由模型按自己的规则选择。
- 规则的 `predicate` 覆盖模型的选择。例如 `set` 规则上的 `gte: 0.8` 匹配概率至少为 0.8 的标签。
- 模型支持原生 `set` 时使用原生能力。否则 Router 可以为每个标签提出一个 `noul` 问题来组合 `set`，任务目录将其标为 `composed_noul`，而非原生 Set 支持。

准备阶段会检查模型的实际能力。Decision 1.0/2.0 可以通过 Noul 能力回答路由 `set` 任务，但没有支持 span 的模型就不能生成位置。不支持的任务会导致准备失败。结构上支持任务不代表任务准确率，应使用选定模型评估问题和阈值。公开的 System One 原生 API 仍只接受该模型原生支持的题型。

模型正在加载、过载或响应超过 `timeout_ms` 时，信号为未知。决策的 `rules.on_unknown` 或条件的 `on_error: match | no_match` 决定如何处理未知结果。兼容的问题可以共享批处理；不同输入状态及后续选模或响应阶段可能需要额外调用和 forward。当其他调用方仍需要结果时，共享调用可能继续运行。截止时间结束调用方等待，但不保证已开始的模型 forward 立即停止。

Decision 信号可能按照部署的扫描预算截取路由判断视图。这不会缩短发送给选中 Chat 后端的请求。需要完整输入覆盖时，使用专用 PII 或 Reask 任务。匹配的 Decision 信号列在响应头 `x-vsr-matched-decision-model` 中。

选择模型、规模、硬件或在自己的 GPU 服务器上运行模型，请参阅 [Decision 模型指南](../../../model-runtime/guides/decisions.md)。
