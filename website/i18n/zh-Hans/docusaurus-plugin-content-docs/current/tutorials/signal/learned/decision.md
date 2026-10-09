---
translation:
  source_commit: "fddd53d7c446c30ae183d16b6d63ec54f7a00a3e"
  source_file: "docs/tutorials/signal/learned/decision.md"
  outdated: false
---

# 决策信号

## 概览 {#overview}

`decision` 就请求问决策模型一个带类型的问题，把答案变成一个路由事实。问题用大白话写：几个选项里挑一个（`choice`）、是或否（`noul`）、或刻度上的某一级（`score`）。能答这些问题的模型——比如 Vela 2.0——还收 `set` 问题（这些标签里哪些适用？）和 `span` 问题（每个标签在文本的哪里？）。模型跑在[内置模型运行时](../../../model-runtime/overview.md)里，runtime 由 router 替你拉起来。

## 关键优势 {#key-advantages}

- 新问题写下来就能用，没有分类器要训。
- 答案带概率，路由可以要求一个够有把握的答复。
- 一个请求问同一个模型的所有问题——[`pii` 信号](../../../tutorials/signal/learned/pii.md)用那个模型时的 PII 问题也算——都在一次调用里送去。
- 答迟了或答挂了，信号未知；请求照常走完。

## 解决什么问题 {#what-problem-does-it-solve}

固定标签的分类器只会答它训过的问题。决策模型凭问题文本就能答这一步需不需要多步推理、这段是不是在说我们的产品。

## 什么时候用 {#when-to-use}

需要对整个请求做判断的问题，用它。结构性事实（长度、关键词、模态）用启发式信号；领域、PII、越狱这些已有专门学习信号能答的，用它们。

## 配置 {#configuration}

没写 `deployment` 的问题，问的是 router 的决策模型 `global.model_catalog.system.decision_model`（不[选大小](../../../model-runtime/choose-a-model.md)的话是 Vela 2.0 0.3B）。它加入回答内置信号的那次调用，于是一个模型在一次调用里答完 router 对请求问的每个问题：

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

用 `decision_model: Vela-1.0` 时，Vela 1.0 的专家们只答内置信号，这样的问题就是一个加载错误，要你给出 `deployment`。

要问别的模型——比如一个 Decision 2.0 模型——把它声明成 `model_runtime` 部署，并给每个问题写上它的 `deployment`：

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

| 问题类型 | 何时命中 | 路由可读的值 |
| --- | --- | --- |
| `noul` | 为是的概率满足 `predicate`（默认 `gte: 0.5`） | `decision:<name>` = P(yes) |
| `score` | 期望级位满足 `predicate`（必填；级位从 0 数起） | `decision:<name>` = 期望级位 |
| `choice` | 条件的 `label` 是被选中的选项，设了 `predicate` 时其概率也要满足 | `decision:<name>:<key>` = P(key)，`decision:<name>` = P(chosen) |
| `set` | 条件的 `label` 的概率满足 `predicate`；没设时，模型选中了该标签 | `decision:<name>:<key>` = P(key)，`decision:<name>` = 最高的 P |
| `span` | 一个带条件 `label` 的片段，其概率满足 `predicate`；没设时，模型找到了带该标签的片段 | `decision:<name>:<key>` = 该标签最可能的片段（没有时为 0），`decision:<name>` = 任何词达到的最高概率 |

条件可以自带 `predicate`；带 `label` 的 `choice`、`set` 或 `span` 条件，读的是那个标签的值。

### Set 和 span 问题 {#set-and-span-questions}

`set` 问题列一批标签，问哪些适用；`span` 问题问每个标签在请求的哪里出现。两者都用 `labels` 而不是 `choices`，每项带 `key` 和可选的 `description`，条件则指名一个标签：

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

- `threshold`（0 到 1）替换模型自己的判定阈值，只作用于这个问题。不设的话用模型校准过的阈值——`selected` 出哪些标签、找到哪些片段，由它决定。
- `head`（仅 `span`）在有两个 span 头的模型上指定用哪个：`router`（在 PII、无依据论断和有害片段上训的）或 `broad`（开放抽取）。不指定时模型按自己的规则选。
- 规则的 `predicate` 替换模型的选择：`set` 规则上的 `gte: 0.8` 匹配概率不低于 0.8 的标签。
- 模型在同一次调用里把每个 `set` 标签答在 `<name>.<label>` 下，所以该部署上别的决策信号不能用这个名字。

只有声明了这些问题类型的模型才答它们。router 准备配置时——启动时或重载时——会把每个用到的 `set` 或 `span` 问题对着它部署的模型检查；模型只答 `choice`、`noul` 和 `score`（Decision 1.0 和 2.0）时，就带着规则名失败。

模型加载中、过载、或比 `timeout_ms` 还慢，信号都是未知的。决策的 `rules.on_unknown`，或条件的 `on_error: match | no_match`，决定未知答案意味着什么。一个请求问一个部署的所有问题——内置信号的也算——都在一次调用里发出；调用一旦发出，每个问题等它等到其中最晚的那个，毕竟请求本来就在等这次调用。命中的决策信号列在 `x-vsr-matched-decision-model` 响应头里。

选模型、定大小和硬件，或把模型跑在自己的 GPU 服务器上，见[决策模型](../../../model-runtime/guides/decisions.md)。
