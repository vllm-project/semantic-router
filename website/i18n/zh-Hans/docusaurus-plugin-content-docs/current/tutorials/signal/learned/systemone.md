---
is_mtpe: false
translation:
  source_commit: "085b44d375765d4935a1e152a2d14d59c1e64b75"
  source_file: "docs/tutorials/signal/learned/systemone.md"
  outdated: true
---

# System One 信号

## 概述 {#overview}

`systemone` 向 System One 模型提出一个关于请求的类型化问题，再把答案转换成路由事实。你可以用自然语言描述问题：从几个选项中选一个（`choice`）、回答是或否（`noul`），或在一组等级上评分（`score`）。Vela 2.0 等支持更多题型的模型还能回答 `set`（哪些标签适用？）和 `span`（各标签出现在文本的什么位置？）问题。模型运行在 Router 自动启动的[内置模型运行时](/model-runtime/overview.md)中，或运行在远程 [System One API](#ask-a-remote-system-one-model) 之后。

## 主要优势 {#key-advantages}

- 写好问题即可使用，无需训练新的分类器。
- 答案带有概率，可以要求达到指定置信程度后再匹配路由。
- 规则读取问题需要的内容：当前轮次、之前的轮次、工具结果，或请求中的图片和视频。
- 使用同一部署且输入兼容的问题可以合并为一次请求阶段调用，包括 [`pii` 信号](pii.md#vela-20)。
- 超时或失败的答案会使信号变为未知；decision 的失败策略决定继续路由还是拒绝请求。

## 解决什么问题？ {#what-problem-does-it-solve}

固定标签的分类器只能回答训练时定义的问题。System One 模型则可以直接根据问题描述，回答“是否需要逐步推理？”或“是否与我们的产品有关？”。

## 何时使用 {#when-to-use}

当问题需要理解整个请求时，使用 System One 信号。长度、关键词和模态等结构性事实优先使用启发式信号；领域、PII、越狱等已有专用学习信号能够解决的问题，也优先使用对应信号。

## 配置 {#configuration}

未指定 `deployment` 的问题使用 `global.model_catalog.system.decision_model` 指向的 Router 判断模型，默认为 Vela 3.0 vela3-fast，也可以[选择其他模型](/model-runtime/choose-a-model.md#choose-a-size)。它会与输入兼容的内置问题一起加入请求阶段的批处理。复用部署可以避免重复加载模型，但不保证完整路由请求只执行一次前向计算：

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
```

决策模型默认为 `{deployment: primary}`，也可以是 Vela 或 Decision 1.0/2.0 部署。所选模型必须支持所有请求的题型。[`systemone` 选模算法](../../algorithm/selection/systemone.md)默认询问同一个决策模型，因此它也可以选择后端。

要使用另一个模型，例如 Decision 2.0，先声明一个 `model_runtime` 部署，再为每个问题指定 `deployment`：

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
recipes:
  - name: default
    signals:
      systemone:
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
          labels: [code, math]
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
            - type: systemone
              name: request_kind
              label: code
            - type: systemone
              name: needs_reasoning
        modelRefs:
          - model: large-coder
```

为早期版本编写的配置把这个信号称为 `decision`（`recipes[].signals.decision` 和 `type: decision` 条件）。这些配置仍可加载，但会给出 `renamed_signal_type` 警告；`vllm-sr config migrate` 会把它们改写为 `systemone`。

### 规则字段 {#rule-fields}

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供条件和投影引用。 |
| `question` | 类型化问题：`type`、`instructions`，以及 `choices`、`levels` 或 `labels`。 |
| `labels` | `choice`、`set` 或 `span` 规则匹配的选项。条件仍可指定任意选项。 |
| `ignore_labels` | 仅适用于 `set` 和 `span`：永远不计入规则的标签。 |
| `predicate` | 答案必须满足的概率或等级（`gt`、`gte`、`lt`、`lte`）。 |
| `deployment` | 回答问题的部署；未指定时为决策模型。 |
| `stage` | `request`（默认）或 `response`。 |
| `input` | 除当前用户轮次外规则还读取的内容，见[读取当前轮次之外的内容](#read-more-than-the-current-turn)。 |
| `coverage` | `truncate`（默认）只读取模型输入能容纳的部分；`full` 读取全部内容。 |
| `on_unscanned` | 仅适用于 `coverage: full`：答案未能读完整个输入时的含义：`match`、`no_match` 或 `unknown`（默认）。 |
| `timeout_ms` | 答案的截止时间，最多 60000 ms；未设置时使用部署的值，否则为 1000 ms。 |

### 规则匹配什么 {#what-a-rule-matches}

条件 `{type: systemone, name: <rule>}` 在规则匹配时匹配；带 `label` 的条件在该选项或标签匹配时匹配：

| 题型 | 规则匹配条件 | 条件的 `label` 匹配条件 |
| --- | --- | --- |
| `noul` | 回答“是”的概率满足 `predicate`，默认 `gte: 0.5` | （不接受标签） |
| `score` | 期望等级满足必填的 `predicate`；等级从 0 开始计数 | （不接受标签） |
| `choice` | 选中的选项属于 `labels`，且其概率满足已配置的 `predicate`；没有 `labels` 的规则只能通过标签匹配 | 该标签是选中的选项，且其概率满足已配置的 `predicate` |
| `set` | 某个计入的标签匹配 | 该标签的概率满足 `predicate`；未配置时，以模型是否选中该标签为准 |
| `span` | 某个计入的标签匹配 | 存在一个带有该标签的片段，其概率满足 `predicate`；未配置时，以模型是否找到带有该标签的片段为准 |

`set` 或 `span` 规则计入 `labels` 中的标签（未设置时为问题的全部标签），再去掉 `ignore_labels`。条件可以配置自己的 `predicate`；对于带 `label` 的条件，它读取该标签的值。以下情况配置会被拒绝：`label`、`labels` 或 `ignore_labels` 中的条目不是问题的选项；条件指定了被忽略的标签；不带标签的条件引用了没有 `labels` 的 `choice` 规则。

规则发布以下值，供条件的 `predicate` 和[投影分数](../../projection/scores.md)读取：

| 题型 | `systemone:<name>` | `systemone:<name>:<key>` |
| --- | --- | --- |
| `noul` | P(yes) | — |
| `score` | 期望等级 | — |
| `choice` | P(chosen) | 每个选项的 P(key) |
| `set` | 计入标签中的最高概率 | 每个标签的 P(key) |
| `span` | 计入标签中的最高概率 | 该标签最高概率片段的概率（未找到时为 0） |

投影分数通过 `value_source: raw` 读取这些值：`name: <question>` 读取 `systemone:<name>`，`name: <question>:<key>` 读取 `choice`、`set` 或 `span` 问题的一个选项或标签。只要启用的投影读取某个问题，就会执行该问题：

```yaml
recipes:
  - name: default
    signals:
      systemone:
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
```

### 读取当前轮次之外的内容 {#read-more-than-the-current-turn}

规则默认读取当前用户轮次。`input` 把对话中具名的部分加入模型读取的内容：

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
- `previous_turns`（0 到 32）：对话最后几轮（用户和助手）组成的对话记录。
- `tool_results`：请求中的工具结果。
- `image` 和 `video`：当前轮次的图片和视频。部署必须能读取这种输入：模型卡列出该模态的模型运行时模型，或在 `capabilities.modalities` 中声明它的 `systemone` 部署。否则配置会以 `input_modality_unsupported` 被拒绝。目前没有部署读取音频，因此 `input.audio` 总会被拒绝。
- `answer` 和 `context`：模型的回答及其依据上下文，仅适用于 `stage: response` 规则。

读取请求相同部分的规则共享一次调用；规则读取不同部分时，这次调用是批量请求，规则读取的每段文本各占一条。

### 读取完整输入 {#read-the-whole-input}

规则默认读取模型输入能容纳的部分并丢弃其余内容（`coverage: truncate`）。设置 `coverage: full` 时，模型读取全部内容，超过一次输入长度时分窗口读取。当答案未能覆盖整个输入（扫描预算用尽、模型无法判断覆盖范围，或输入超过模型长度）时，`on_unscanned` 决定规则的结果：`match`、`no_match`，或 `unknown`（默认，交给 decision 的失败策略处理）：

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

### 关于回答的问题 {#questions-about-the-answer}

`stage: response` 针对模型的回答提问，并用 `input.answer` 和 `input.context` 指定读取的内容。这样的规则由响应插件读取；decision 的规则不能引用它，因为 decision 在回答产生之前就完成路由，引用它的配置会被拒绝。

### 询问远程 System One 模型 {#ask-a-remote-system-one-model}

问题也可以询问 Router 不管理的 System One API：托管的 Decision API、另一个 vLLM-SR Engine，或位于它们前面的网关。在 `global.model_catalog.deployments` 中把它声明为 `systemone` 部署，并像 `model_runtime` 部署一样把它写进问题的 `deployment`。它只回答 System One 问题：它不属于请求会被路由到的 `providers.models`，任何模型列表都不会显示它。

```yaml alternative
global:
  model_catalog:
    deployments:
      d3-ultra-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth:
          api_key_env: VSR_APP_KEY
        timeout_ms: 3000
recipes:
  - name: default
    signals:
      systemone:
        - name: support_team
          deployment: d3-ultra-remote
          question:
            type: choice
            instructions: Which team should handle this ticket?
            choices:
              - key: billing
                description: Payments and refunds
              - key: engineering
                description: Outages and bugs
          labels: [billing]
```

`endpoint` 是 API 接收 `POST` 请求的 URL，包含路径前缀；`model` 是 API 用来回答的模型。密钥来自 `auth.api_key_env`，或来自 `auth.api_key_file`（绝对路径，在调用需要时读取），从不写在配置里，并以 `Authorization: Bearer <key>` 发送。如果网关要求把密钥放在自己的请求头中，设置 `auth.header`；设置 `auth.prefix: ""` 可以不带前缀发送密钥：

```yaml
global:
  model_catalog:
    deployments:
      gateway-d3:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        capabilities: {question_types: [choice, noul, score], modalities: [text, image]}
        auth:
          header: Ocp-Apim-Subscription-Key
          prefix: ""
          api_key_env: GATEWAY_KEY
```

- 同一请求阶段发给该部署的问题（包括询问它的内置信号的问题）会合并为一次 System One 调用：选项以 System One `criteria` 发送，调用的截止时间以 `options.deadline_ms` 发送。关于一段文本的问题是一个标准请求；关于多段文本的问题是一个批量请求，每段文本在 `states` 下各占一条，没有顶层的 `state` 或 `questions`（见模型运行时的 [decisions 参考](../../../model-runtime/reference#decisions)）。API 必须接受这两种形式。System One API 会因一个失败的问题而整体拒绝调用，并指明该问题，因此 Router 会在同一截止时间内去掉它、把其余问题再问一次：只有失败问题对应的信号变为未知。其他任何失败都会使该次调用的所有信号变为未知。针对同一输入的相同问题，其答案最多复用 10 分钟，配置变更后即失效。
- 模型能回答的题型、现成问题（`pii`、`halu`）和能读取的输入模态优先取自配置加载时读取的 `GET <endpoint>/models` 列表中该模型的信息；列表未列出时使用 `capabilities.question_types`、`capabilities.presets` 和 `capabilities.modalities`；都没有时使用 System One 题型 `choice`、`noul` 和 `score`，并只读取文本。模型不原生支持 `set` 时，`set` 问题会组合为多个 `noul` 问题。
- 网络往返比本地模型更慢。未设置自身 `timeout_ms` 的问题使用部署的 `timeout_ms`，否则使用默认的 1000 ms。API 变慢、失败或不可达时，其信号为未知，请求不会因此失败。Router 对每类失败只记录一次日志，且不包含密钥。
- `global.model_catalog.system.decision_model.deployment` 也可以指向 `systemone` 部署。此时未指定部署的问题和 [System One 选择器](../../algorithm/selection/systemone.md) 会询问它，没有被绑定到其他部署的内置信号也会询问它。安全规则的 hazard 部分、embedding 和重排序仍使用各自的部署。配置未写出的模块阈值在这样的模型上为 0.5，并给出 `uncalibrated_preset` 警告。在不回答 span 问题的模型上，PII 和幻觉检测只给出不含位置的是/否判断，并在配置加载时给出 `span_fallback` 警告；此时对 PII 做 masking 的配置会被拒绝。[为信号选择模型](../../../installation/choose-a-model-for-a-signal.md#s2-everything-on-a-remote-system-one-model)展示了这种配置。
- `systemone` 部署不接受设备、引擎参数、副本或输入预算。

### Set 和 Span 问题 {#set-and-span-questions}

`set` 问题列出标签并询问哪些适用；`span` 问题询问请求中每个标签出现的位置。两者都使用 `labels` 而非 `choices`，每个标签包含 `key` 和可选的 `description`；规则条件通过标签名引用其中一个标签，不指定标签时匹配任一计入的标签：

```yaml alternative
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
recipes:
  - name: default
    signals:
      systemone:
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
              - key: other
                description: anything else
          ignore_labels: [other]
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
            - type: systemone
              name: support_topics
              label: billing
            - type: systemone
              name: account_ids
        modelRefs:
          - model: support-model
```

- `threshold`（0 到 1）替换模型对该问题使用的判断阈值。未设置时，模型使用自己的校准阈值，决定哪些标签被标记为 `selected`，以及哪些片段被找到。
- `head`（仅适用于 `span`）为具有两个片段检测头的模型选择任务头：`router` 用于 PII、无依据声明和有害文本片段，`broad` 用于开放式提取。未设置时，模型按自身规则选择。
- 规则的 `predicate` 替换模型的选择结果。例如，`set` 规则上的 `gte: 0.8` 会匹配概率至少为 0.8 的标签。
- 模型支持原生 `set` 时，Router 使用原生题型。否则，Router 可以为每个标签提出一个 `noul` 问题，再组合成集合结果。任务目录将这种能力标记为 `composed_noul`，不会标成原生 Set 支持。

只有规则的部署能回答它时，配置才会加载。部署不回答 span 问题时，`span` 规则会以 `span_required` 被拒绝；规则的 `input` 读取部署不支持的模态时，会以 `input_modality_unsupported` 被拒绝。模型只有运行后才能确定的模型运行时部署（attached 运行时）会在运行时准备问题时检查，错误信息相同。因此，Decision 1.0/2.0 可以通过 Noul 能力完成路由中的 `set` 任务，但无法生成 `span` 片段位置。具备题型能力并不代表任务准确率达标：请使用选定模型评估问题和阈值。公开的原生 System One API 仍只接受该模型原生提供的题型。

在 Vela 3.0 D3 模型上，没有训练形式的问题会在模型包带有 decision adapter 时使用它回答；其他模型包会忽略这个请求。

模型加载中、过载或响应超过 `timeout_ms` 时，信号为未知。decision 的 `rules.on_unknown` 或条件的 `on_error: match | no_match` 决定如何处理未知答案。输入兼容的问题可以共享批处理；不同输入状态、后续选模阶段或响应阶段可能需要额外的调用和前向计算。当其他调用方仍需要结果时，共享调用可以继续执行。截止时间会结束当前调用方的等待，但不保证立即停止已经开始的模型前向计算。

按部署的扫描预算截取只会缩短规则读取的内容，不会缩短发送给所选 Chat 后端的请求。匹配的 System One 规则会以 `systemone:<name>` 列在 `x-vsr-trace` 的 `signals` 成员中。

关于模型、规模、硬件的选择，以及如何在自己的 GPU 服务器上运行模型，请参阅 [Decision 模型指南](/model-runtime/guides/decisions.md)。
