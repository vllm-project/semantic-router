---
is_mtpe: false
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/signal/learned/decision.md"
  outdated: false
---

# Decision 信号

## 概述 {#overview}

`decision` 向判断模型提出一个关于请求的类型化问题，再把答案转换成路由事实。你可以用自然语言描述问题：从几个选项中选一个（`choice`）、回答是或否（`noul`），或在一组等级上评分（`score`）。Vela 2.0 等支持更多题型的模型还能回答 `set`（哪些标签适用？）和 `span`（各标签出现在文本的什么位置？）问题。模型运行在 Router 自动启动的[内置模型运行时](/model-runtime/overview.md)中，或运行在远程 [System One API](#ask-a-remote-system-one-model) 之后。

## 主要优势 {#key-advantages}

- 写好问题即可使用，无需训练新的分类器。
- 答案带有概率，可以要求达到指定置信程度后再匹配路由。
- 使用同一部署且输入兼容的问题可以合并为一次请求阶段调用，包括 [`pii` 信号](pii.md#vela-20)。
- 超时或失败的答案会使信号变为未知；decision 的失败策略决定继续路由还是拒绝请求。

## 解决什么问题？ {#what-problem-does-it-solve}

固定标签的分类器只能回答训练时定义的问题。判断模型则可以直接根据问题描述，回答“是否需要逐步推理？”或“是否与我们的产品有关？”。

## 何时使用 {#when-to-use}

当问题需要理解整个请求时，使用 Decision 信号。长度、关键词和模态等结构性事实优先使用启发式信号；领域、PII、越狱等已有专用学习信号能够解决的问题，也优先使用对应信号。

## 配置 {#configuration}

未指定 `deployment` 的问题使用 `global.model_catalog.system.decision_model` 指向的 Router 判断模型，默认为 Vela 3.0 vela3-fast，也可以[选择其他模型](/model-runtime/choose-a-model.md#choose-a-size)。它会与输入兼容的内置问题一起加入请求阶段的批处理。复用部署可以避免重复加载模型，但不保证完整路由请求只执行一次前向计算：

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

决策模型默认为 `{deployment: primary}`，也可以是 Vela 或 Decision 1.0/2.0 部署。所选模型必须支持所有请求的题型。[`decision` 选模算法](../../algorithm/selection/decision.md)默认询问同一个决策模型，因此它也可以选择后端。

要使用另一个模型，例如 Decision 2.0，先声明一个 `model_runtime` 部署，再为每个问题指定 `deployment`：

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

### 询问远程 System One 模型 {#ask-a-remote-system-one-model}

问题也可以询问 Router 不管理的 System One API：托管的 Decision API、另一个 vLLM-SR Engine，或位于它们前面的网关。在 `global.model_catalog.deployments` 中把它声明为 `systemone` 部署，并像 `model_runtime` 部署一样把它写进问题的 `deployment`。它只回答 decision 问题：它不属于请求会被路由到的 `providers.models`，任何模型列表都不会显示它。

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

routing:
  signals:
    decision:
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
        capabilities: {question_types: [choice, noul, score]}
        auth:
          header: Ocp-Apim-Subscription-Key
          prefix: ""
          api_key_env: GATEWAY_KEY
```

- 同一请求阶段发给该部署的问题（包括询问它的内置信号的问题）会合并为一次 System One 调用：选项以 System One `criteria` 发送，问题读取的其他文本作为额外的 `states` 条目发送，调用的截止时间以 `options.deadline_ms` 发送。System One API 会因一个失败的问题而整体拒绝调用，并指明该问题，因此 Router 会在同一截止时间内去掉它、把其余问题再问一次：只有失败问题对应的信号变为未知。其他任何失败都会使该次调用的所有信号变为未知。针对同一文本的相同问题，其答案最多复用 10 分钟，配置变更后即失效。
- 模型能回答的题型和现成问题（`pii`、`halu`）优先取自配置加载时读取的 `GET <endpoint>/models` 列表中该模型的信息；列表未列出时使用 `capabilities.question_types` 和 `capabilities.presets`；都没有时使用 System One 题型 `choice`、`noul` 和 `score`。模型不原生支持 `set` 时，`set` 问题会组合为多个 `noul` 问题。
- 网络往返比本地模型更慢。未设置自身 `timeout_ms` 的问题使用部署的 `timeout_ms`，否则使用默认的 1000 ms。API 变慢、失败或不可达时，其信号为未知，请求不会因此失败。Router 对每类失败只记录一次日志，且不包含密钥。
- `global.model_catalog.system.decision_model.deployment` 也可以指向 `systemone` 部署。此时未指定部署的问题和 [decision 选择器](../../algorithm/selection/decision.md) 会询问它，没有被绑定到其他部署的内置信号也会询问它。安全规则的 hazard 部分、embedding 和重排序仍使用各自的部署。配置未写出的模块阈值在这样的模型上为 0.5，并给出 `uncalibrated_threshold` 警告。在不回答 span 问题的模型上，PII 和幻觉检测只给出不含位置的是/否判断，并在配置加载时给出 `span_fallback` 警告；此时对 PII 做 masking 的配置会被拒绝。[为信号选择模型](../../../installation/choose-a-model-for-a-signal.md#s2-everything-on-a-remote-system-one-model)展示了这种配置。
- `systemone` 部署不接受设备、引擎参数、副本或输入预算。

| 题型 | 匹配条件 | 路由可读取的值 |
| --- | --- | --- |
| `noul` | 回答“是”的概率满足 `predicate`，默认 `gte: 0.5` | `decision:<name>` = P(yes) |
| `score` | 期望等级满足必填的 `predicate`；等级从 0 开始计数 | `decision:<name>` = 期望等级 |
| `choice` | 条件的 `label` 是选中的选项，且该选项的概率满足已配置的 `predicate` | `decision:<name>:<key>` = P(key)，`decision:<name>` = P(chosen) |
| `set` | 条件的 `label` 概率满足 `predicate`；未配置时，以模型是否选中该标签为准 | `decision:<name>:<key>` = P(key)，`decision:<name>` = 最高概率 |
| `span` | 存在一个带有条件 `label` 的片段，其概率满足 `predicate`；未配置时，以模型是否找到带有该标签的片段为准 | `decision:<name>:<key>` = 该标签最高概率片段的概率（未找到时为 0），`decision:<name>` = 所有词中的最高概率 |

条件可以配置自己的 `predicate`。对于指定了 `label` 的 `choice`、`set` 或 `span` 条件，它读取该标签的值。

[投影分数](../../projection/scores.md)通过 `value_source: raw` 读取同样的值：`name: <question>` 读取 `decision:<name>`，`name: <question>:<key>` 读取 `choice`、`set` 或 `span` 问题的一个选项或标签。只要启用的投影读取某个问题，就会执行该问题：

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

### Set 和 Span 问题 {#set-and-span-questions}

`set` 问题列出标签并询问哪些适用；`span` 问题询问请求中每个标签出现的位置。两者都使用 `labels` 而非 `choices`，每个标签包含 `key` 和可选的 `description`；规则条件通过标签名引用它们：

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

- `threshold`（0 到 1）替换模型对该问题使用的判断阈值。未设置时，模型使用自己的校准阈值，决定哪些标签被标记为 `selected`，以及哪些片段被找到。
- `head`（仅适用于 `span`）为具有两个片段检测头的模型选择任务头：`router` 用于 PII、无依据声明和有害文本片段，`broad` 用于开放式提取。未设置时，模型按自身规则选择。
- 规则的 `predicate` 替换模型的选择结果。例如，`set` 规则上的 `gte: 0.8` 会匹配概率至少为 0.8 的标签。
- 模型支持原生 `set` 时，Router 使用原生题型。否则，Router 可以为每个标签提出一个 `noul` 问题，再组合成集合结果。任务目录将这种能力标记为 `composed_noul`，不会标成原生 Set 支持。

准备阶段会检查模型的实际能力。因此，Decision 1.0/2.0 可以通过 Noul 能力完成路由中的 `set` 任务，但没有 Span 能力的模型无法生成片段位置。不支持的任务会导致准备失败。具备题型能力并不代表任务准确率达标：请使用选定模型评估问题和阈值。公开的原生 System One API 仍只接受该模型原生提供的题型。

模型加载中、过载或响应超过 `timeout_ms` 时，信号为未知。decision 的 `rules.on_unknown` 或条件的 `on_error: match | no_match` 决定如何处理未知答案。输入兼容的问题可以共享批处理；不同输入状态、后续选模阶段或响应阶段可能需要额外的调用和前向计算。当其他调用方仍需要结果时，共享调用可以继续执行。截止时间会结束当前调用方的等待，但不保证立即停止已经开始的模型前向计算。

Decision 信号可能按部署的扫描预算截取用于路由判断的输入。这不会缩短发送给所选 Chat 后端的请求；需要完整输入覆盖时，请使用专用的 PII 或 Reask 任务。匹配的 Decision 信号会列在 `x-vsr-matched-decision-model` 响应头中。

关于模型、规模、硬件的选择，以及如何在自己的 GPU 服务器上运行模型，请参阅 [Decision 模型指南](/model-runtime/guides/decisions.md)。

## Router 判断中的图像和视频

Decision 部署可以显式启用请求中的内联图像和视频。默认不启用。该设置按部署生效，适用于请求阶段的 Decision 信号、共享调用的内置问题以及 Decision 选模算法。响应阶段的检查不会继承用户附件。

```yaml
global:
  model_catalog:
    deployments:
      decision-d3:
        provider: model_runtime
        artifact: vllm-sr/d3-lite
        input:
          media:
            images: {max_count: 4}
            videos: {max_count: 1, max_bytes: 16000000}
```

将问题或选模器绑定到 `decision-d3`。启动时会验证模型卡是否支持已启用的模态。`max_bytes` 和 `max_count` 只能收紧模型卡或默认限制；省略某种模态即禁用该模态。

图像只接受内联 PNG、JPEG 或 WebP 数据 URL；视频只接受内联 MP4、WebM、QuickTime 或 Matroska 数据 URL。Router 会验证 base64 和容器签名，不会为 Decision 调用获取远程 URL 或文件。无效、不支持、过多或过大的附件会被跳过，原始后端请求仍保留其媒体。

`vllm-sr route preview --request-file request.json` 与实时路由使用相同的部署媒体策略。请将内联附件放在请求的 `messages` 中。预览响应不会包含附件，响应阶段检查也不会继承这些附件。
