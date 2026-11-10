---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/systemone.md"
  outdated: false
---

# System One 信号 {#system-one-signal}

## 概述 {#overview}

`systemone` 向决策模型提出一个你编写的类型化问题，再把答案变成一个路由事实。你用自然语言写问题：从几个选项中选一个（`choice`）、是或否（`noul`）、刻度上的等级（`score`）、这些标签中哪些适用（`set`），或者每个标签出现在文本的什么位置（`span`）。它是[应用信号](./overview#application-signals)编译后的原语，[System One](../../system-one/overview) 一章介绍了背后的模型。

## 主要优势 {#key-advantages}

- 问题写好就能用，无需训练分类器。
- 答案带有概率，路由可以要求答案足够可信。
- 规则读取问题需要的内容：当前轮次、之前的轮次、工具结果、请求中的图片，或模型的答案。
- 针对同一请求的问题，与该请求向同一部署提出的其他问题共享一次调用。

## 解决什么问题？ {#what-problem-does-it-solve}

固定标签的分类器只能回答训练时定义的问题。System One 模型仅凭问题文本，就能回答“这需要逐步推理吗？”或“这和我们的产品有关吗？”。

## 何时使用 {#when-to-use}

当你需要对请求做出某种判断，而现有的[应用信号](./overview#application-signals)都不涵盖时使用它。长度、关键词、输入媒体、身份等结构性事实，优先使用启发式信号。[编写自定义问题](../../system-one/use/custom-questions)从一个想法一路讲到调优好的规则。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate:
            gte: 0.7
        - name: request_kind
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
          labels: [code, math]
        - name: difficulty
          deployment: scorer
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
            - type: systemone
              name: request_kind
              label: code
            - type: systemone
              name: difficulty
        modelRefs:
          - model: large-coder
global:
  model_catalog:
    deployments:
      scorer:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
```

没有 `deployment` 的规则询问决策模型 `global.model_catalog.system.decision_model`（除非另行选择，否则为 vela3-fast）；[选择部署](../../system-one/deploy/deployments)介绍了本地、挂接和远程模型。部署必须能回答规则的问题类型，否则配置加载时会被拒绝。vela3-fast 回答 `choice`、`noul` 和 `span` 问题，并把 `set` 问题拆成每个标签一个 `noul` 来回答，但不回答 `score` 问题，所以 `difficulty` 询问的是一个 Decision 2.0 部署。

为早期版本编写的配置把这个信号命名为 `decision`（`recipes[].signals.decision` 和 `type: decision` 条件）。它们仍能加载，但会给出 `renamed_signal_type` 警告，`vllm-sr config migrate` 会把它们改写为 `systemone`。

### 规则字段 {#rule-fields}

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供条件和投影使用。 |
| `question` | 类型化问题：`type`、`instructions`，以及它的 `choices`、`levels` 或 `labels`。 |
| `labels` | `choice`、`set` 或 `span` 规则匹配的选项。条件仍可以指定任意选项。 |
| `ignore_labels` | 仅 `set` 和 `span`：永不计入规则的标签。 |
| `predicate` | 答案必须满足的概率或等级（`gt`、`gte`、`lt`、`lte`）。 |
| `deployment` | 回答问题的部署；不写时为决策模型。 |
| `stage` | `request`（默认）或 `response`。 |
| `input` | 规则在当前用户轮次之外读取的内容；见[读取当前轮次之外的内容](#read-more-than-the-current-turn)。 |
| `coverage` | `truncate`（默认）读取能放进模型输入的部分；`full` 读取全部。 |
| `on_unscanned` | 仅 `coverage: full`：无法读完整个输入的答案意味着什么：`match`、`no_match` 或 `unknown`（默认）。 |
| `timeout_ms` | 答案的截止时间，最长 60000 毫秒；否则使用部署的值，再否则为 1000 毫秒。 |

### 规则匹配什么 {#what-a-rule-matches}

条件 `{type: systemone, name: <rule>}` 在规则匹配时匹配；带 `label` 的条件在该选项或标签匹配时匹配：

| 问题类型 | 规则匹配的条件 | 条件的 `label` 匹配的条件 |
| --- | --- | --- |
| `noul` | “是”的概率满足 `predicate`（默认 `gte: 0.5`） | （不接受 label） |
| `score` | 期望等级满足 `predicate`（必填；等级从 0 开始计） | （不接受 label） |
| `choice` | 选中的选项属于 `labels`，并且设置了 `predicate` 时其概率满足它；没有 `labels` 的规则只能通过 label 匹配 | 该 label 就是选中的选项，并且设置了 `predicate` 时其概率满足它 |
| `set` | 有计入的标签匹配 | 该标签的概率满足 `predicate`；不设置时，模型选中了该标签 |
| `span` | 有计入的标签匹配 | 带该标签的某个 span 概率满足 `predicate`；不设置时，模型找到了带该标签的 span |

`set` 或 `span` 规则计入 `labels`（不写时为问题的所有标签），再减去 `ignore_labels`。条件可以加上自己的 `predicate`；带 `label` 的条件读取该标签的值。以下情况配置会被拒绝：`label`、`labels` 或 `ignore_labels` 中的某项不是问题的选项；条件指定了被忽略的标签；不带 label 的条件指定了没有 `labels` 的 `choice` 规则。

规则发布以下值，供条件的 `predicate` 和[投影得分](../projection/scores)读取：

| 问题类型 | `systemone:<name>` | `systemone:<name>:<key>` |
| --- | --- | --- |
| `noul` | P(是) | — |
| `score` | 期望等级 | — |
| `choice` | P(选中) | 每个选项的 P(key) |
| `set` | 计入标签中最高的 P | 每个标签的 P(key) |
| `span` | 计入标签中最高的概率 | 该标签最可能的 span（没有时为 0） |

投影得分用 `value_source: raw` 读取它们：`name: <question>` 读取 `systemone:<name>`，`name: <question>:<key>` 读取 `choice`、`set` 或 `span` 问题的某个选项或标签。只要有被使用的投影读取它，问题就会被提出：

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: difficulty
          deployment: scorer
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
            - type: systemone
              name: difficulty
              weight: 0.3
              value_source: raw
            - type: systemone
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
global:
  model_catalog:
    deployments:
      scorer:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
```

### 读取当前轮次之外的内容 {#read-more-than-the-current-turn}

规则读取当前用户轮次。`input` 把对话中的指定部分加入模型读取的内容：

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: escalation
          question:
            type: noul
            instructions: Has the user asked for the same fix more than once?
          input:
            history: true
            previous_turns: 6
            tool_results: true
```

- `history`：之前的用户轮次。
- `previous_turns`（0 到 32）：最近几个轮次（用户和助手），以对话记录形式给出。
- `tool_results`：请求中的工具结果。
- `image` 和 `video`：当前轮次的图片和视频。部署必须能读取该输入，否则配置以 `input_modality_unsupported` 被拒绝。目前还没有部署能读取音频，所以 `input.audio` 总会被拒绝。
- `answer` 和 `context`：模型的答案及其依据上下文，仅用于 `stage: response` 规则。

读取请求相同部分的规则共享一次调用；规则读取不同部分时，这次调用是批量请求，规则读取的每段文本各占一条。

### 读取完整输入 {#read-the-whole-input}

规则读取能放进模型输入的部分并丢弃其余部分（`coverage: truncate`）。设置 `coverage: full` 时，模型读取全部内容，超过一次输入长度时分窗口读取。当答案无法覆盖完整输入时（扫描预算耗尽、模型无法判断覆盖范围，或输入超过模型长度），`on_unscanned` 决定规则报告什么：`match`、`no_match` 或 `unknown`（默认，交给决策的失败策略处理）：

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: account_ids
          question:
            type: span
            instructions: Which spans are account or order numbers?
            labels:
              - key: account_number
                description: a customer account or order number
          coverage: full
          on_unscanned: match
```

### 关于答案的问题 {#questions-about-the-answer}

`stage: response` 针对模型的答案提问，`input.answer` 和 `input.context` 指定它读取的内容。[`response_guard`](../plugin/response-guard) 等响应插件读取这类规则；决策的规则不能读取，因为决策在答案产生之前就完成了路由，读取它的配置会被拒绝。

### Set 和 span 问题 {#set-and-span-questions}

`set` 问题给出一组标签，询问哪些适用；`span` 问题询问每个标签出现在请求的什么位置。两者都用 `labels` 代替 `choices`，每个标签有 `key` 和可选的 `description`；针对它们的条件指定一个标签，或者不指定标签以匹配任意计入的标签：

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: support_topics
          question:
            type: set
            instructions: Which topics does the request mention?
            labels:
              - key: billing
                description: payments, invoices or refunds
              - key: shipping
                description: deliveries, tracking or returns
              - key: other
                description: anything else
          ignore_labels: [other]
        - name: account_ids
          question:
            type: span
            instructions: Which spans are account or order numbers?
            labels:
              - key: account_number
                description: a customer account or order number
    decisions:
      - name: billing-with-account
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: support_topics
              label: billing
            - type: systemone
              name: account_ids
        modelRefs:
          - model: support-model
```

- `threshold`（0 到 1）为该问题替换模型自己的判定阈值。不写时模型使用其校准阈值，由它决定 `selected` 的标签和找到的 span。
- `head`（仅 `span`）在有两个 span head 的模型上指定由哪个回答：`router`（在个人数据、无依据陈述和有害片段上训练过）或 `broad`（开放式抽取）。不写时由模型按自己的规则选择。只有 Vela 2.0 0.8B、4B 和 9B 有 broad head；在其他内置模型（vela3-fast、其他 Vela 3.0 模型、Vela 2.0 0.3B）上写 `head: broad` 的规则会以 `span_head_unsupported` 加载失败。`systemone` 部署的 head 在加载时未知，由其 API 拒绝它没有的 head。
- 规则的 `predicate` 取代模型的选择：`set` 规则上的 `gte: 0.8` 匹配概率至少为 0.8 的标签。
- 没有原生 `set` 能力的模型，会在同一次调用中把 set 当作每个标签一个 `noul` 问题来回答。
- 部署不支持任何 span 问题时，`span` 规则以 `span_required` 被拒绝。默认决策模型能返回 span；d3 家族以及 Decision 1.0 和 2.0 不能。

### 模型没有给出答案时 {#when-the-model-gives-no-answer}

当模型正在加载、过载或慢于 `timeout_ms` 时，规则为未知。决策上的 `rules.on_unknown`，或条件上的 `on_error: match | no_match`，决定未知答案意味着什么。截断到部署的扫描预算只缩短规则读取的内容，不会缩短发给所选后端的请求。

要查看哪些规则匹配了，使用路由预览（`POST /api/v1/routing/preview`）；对允许查看调试响应头的调用方，也可以查看 `x-vsr-trace` 响应头的 `signals` 成员。
