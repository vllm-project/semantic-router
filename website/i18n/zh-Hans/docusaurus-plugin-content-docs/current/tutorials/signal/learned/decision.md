---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/signal/learned/decision.md"
  outdated: false
---

# Decision 信号

## 概览 {#overview}

`decision` 向判断模型提出有关请求的类型化问题，并将答案转化为路由事实。

你可以用自然语言描述问题：从几个选项中选一个（`choice`）、回答是或否（`noul`），或在量表上给出等级（`score`）。支持相应能力的模型（如 Vela 2.0）还可以回答 `set` 问题（哪些标签适用）和 `span` 问题（各标签在文本中的位置）。模型在 Router 为你启动的[内置模型运行时](../../../model-runtime/overview.md)中运行。

## 主要优势 {#key-advantages}

- 写好新问题即可使用，无需训练分类器。
- 答案带有概率，路由可以要求答案达到置信度阈值。
- 共享部署且输入兼容的问题可以合并为一次请求阶段调用，包括 [`pii` 信号](pii.md#vela-20)。
- 答案超时或失败时，信号变为未知；决策的失败策略决定继续路由还是拒绝请求。

## 解决什么问题？ {#what-problem-does-it-solve}

固定标签分类器只能回答训练时定义的问题。判断模型则可以直接根据问题文本回答“这是否需要逐步推理？”或“这是否与我们的产品有关？”。

## 何时使用 {#when-to-use}

适用于需要对整个请求作出判断的问题。对于长度、关键词、模态等结构性事实，优先使用启发式信号；如果领域、PII、越狱等专用学习信号已经能够回答你的问题，优先使用这些信号。

## 配置 {#configuration}

未指定 `deployment` 的问题会调用 Router 的判断模型，即 `global.model_catalog.system.decision_model`。默认模型为 Vela 2.0 0.3B，也可以[选择其他规模](../../../model-runtime/choose-a-model.md#choose-a-size)。这类问题与兼容的内置问题共同组成请求阶段批次。复用部署可以避免加载另一份模型，但不保证整个路由请求只执行一次 forward：

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

绑定使用 `{deployment: primary}`，可以选择 Vela 或 Decision 1.0/2.0。模型必须支持所有请求的题型。[`decision` 选模算法](../../algorithm/selection/decision.md)使用相同的默认绑定，因此同一资源也能用于选择后端。

若要调用其他模型（例如 Decision 2.0），先声明一个 `model_runtime` 部署，再为每个问题指定 `deployment`：

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

| 题型 | 匹配条件 | 路由可读取的值 |
| --- | --- | --- |
| `noul` | 回答“是”的概率满足 `predicate`，默认为 `gte: 0.5` | `decision:<name>` = P(yes) |
| `score` | 期望等级满足必填的 `predicate`，等级从 0 开始计数 | `decision:<name>` = 期望等级 |
| `choice` | 条件的 `label` 是选中的选项，且设置了 `predicate` 时，其概率满足该条件 | `decision:<name>:<key>` = P(key)，`decision:<name>` = P(chosen) |
| `set` | 条件的 `label` 概率满足 `predicate`；未设置时，该标签被模型选中 | `decision:<name>:<key>` = P(key)，`decision:<name>` = 最高概率 |
| `span` | 带有条件指定 `label` 的片段概率满足 `predicate`；未设置时，模型找到了带有该标签的片段 | `decision:<name>:<key>` = 该标签概率最高的片段的概率，无片段时为 0；`decision:<name>` = 任意词达到的最高概率 |

条件也可以设置自己的 `predicate`。对于带有 `label` 的 `choice`、`set` 或 `span` 条件，读取的是该标签对应的值。

[投影分数](../../projection/scores.md)可通过 `value_source: raw` 读取相同的值：`name: <question>` 读取 `decision:<name>`，`name: <question>:<key>` 读取 `choice`、`set` 或 `span` 问题中的一个选项或标签。只要启用的投影读取了某个问题，Router 就会提出该问题：

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

`set` 问题列出标签并询问哪些适用；`span` 问题询问各标签在请求中的位置。两者都使用 `labels` 而不是 `choices`，每个标签包含 `key` 和可选的 `description`，相应的路由条件需要指定标签：

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

- `threshold`（0 到 1）覆盖模型针对该问题的默认判断阈值。未设置时，模型使用其校准阈值，决定哪些标签被 `selected` 以及哪些片段被检出。
- `head`（仅用于 `span`）为具有两个片段头的模型指定回答问题的头：`router`（在 PII、无依据声明和有害片段上训练）或 `broad`（开放抽取）。未设置时，模型按自身规则选择。
- 规则的 `predicate` 会替代模型的选择结果。例如，`set` 规则上的 `gte: 0.8` 匹配概率至少为 0.8 的标签。
- 模型支持原生 `set` 时直接使用；否则，Router 可以为每个标签提出一个 `noul` 问题来组合出 set。任务目录将其标为 `composed_noul`，而不是原生 Set 支持。

准备阶段会检查模型的实际能力。因此，Decision 1.0/2.0 可以通过 Noul 能力回答路由中的 `set` 任务，但没有支持 span 的模型就不能生成 `span` 位置。不支持的任务会在准备阶段失败。结构上的支持不代表任务准确率，应使用所选模型评估问题和阈值。公开的原生 System One API 仍只接受该模型本身支持的题型。

模型正在加载、过载或响应时间超过 `timeout_ms` 时，信号为未知。决策上的 `rules.on_unknown`，或条件上的 `on_error: match | no_match`，决定如何处理未知答案。兼容的问题共享批次；不同的输入状态、后续选模或响应阶段可能需要额外调用和 forward。只要另一个调用方仍需要结果，共享调用就可能继续执行。截止时间结束的是调用方的等待，不保证已经开始的模型 forward 立即停止。

Decision 信号可能将其用于路由判断的输入截断到部署的扫描预算。这不会缩短发送给所选 Chat 后端的请求；需要完整输入覆盖时，应使用专用的 PII 或 Reask 任务。匹配的 Decision 信号会列在 `x-vsr-matched-decision-model` 响应头中。

关于模型、规模、硬件选择以及在自己的 GPU 服务器上运行模型，请参阅 [Decision 模型指南](/docs/model-runtime/guides/decisions)。
