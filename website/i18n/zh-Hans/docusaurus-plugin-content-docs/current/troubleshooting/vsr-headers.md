---
is_mtpe: false
translation:
  source_commit: "f0cbd8621f9d5e9c6e0e25ea86ac4c0e4bd16326"
  source_file: "docs/troubleshooting/vsr-headers.md"
  outdated: false
---

# VSR 路由头

Router 使用这些请求头和响应头来保持会话连续性、路由可观测性、回放关联，以及按需调试。本页末尾的[头参考](#头参考)列出 Router 读取或写入的每一个头。

## 默认会出现什么

Router 把响应头分到两个面：

- **公开面** — 每个未被响应缓存服务的路由推理响应都包含 `x-vsr-schema-version` 和 `x-vsr-response-path`。这两个关键头由路由流水线产生，因此改由 Router 自身处理器应答的响应（例如 `GET /v1/models`）会省略它们。唯一例外是管理端点 `POST /api/v1/routing/preview`：它在结果成功时返回 `x-vsr-config-hash`。成功路由的响应还可以包含最终配方、决策、置信度、算法、模型、路由延迟、成本和回放 id——仅当启用 Router Replay 且记录已持久化时才包含回放 id。发生协议转换时会出现协议标记；仅在存在警告时出现协议警告。
- **配置版本** — `x-vsr-config-version` 指明服务该请求的配置版本。它出现在每个未被响应缓存服务的路由响应上（无论成功还是出错），也出现在 Router 自己应答的每个响应上，例如拦截、校验错误、缓存命中或 `GET /v1/models`。请求始终保留开始时的版本，即使运行期间热重载激活了另一个版本。
- **调试面** — 中间分类细节、匹配信号、工具选择指标和 `x-vsr-retention-*` 指令仅在请求设置 `x-vsr-debug: true`、且 [`global.router.headers.debug`](#who-may-turn-on-the-debug-headers) 允许该调用方时内联出现；默认不允许任何调用方。启用回放时，同样的诊断上下文仍可通过 `x-vsr-replay-id` 获得。

决策和匹配信号头还要求同时满足以下全部条件：

1. 上游响应成功（`2xx`）。
2. 响应不是由响应缓存提供。
3. Router 为该请求评估了路由决策或信号。

缓存命中响应可以发出缓存头，但它们不会重新运行路由，因此不会附加新的匹配信号头。

客户端用参考表中的 `control` 头来控制请求，例如 `x-session-id`、`x-disable-router-memory` 或 `x-vsr-debug`。

只有参考表中方向包含 `provider request` 的头才会到达提供方：Router 会从每个发往提供方的请求中删掉它自己的控制头、内部头、身份头和密钥。

## 谁可以打开调试头 {#who-may-turn-on-the-debug-headers}

调试头会列出请求命中的信号规则和防护规则、决策排名以及路由学习的原因，所以客户端不能自己打开它们。由 `global.router.headers.debug` 决定谁可以打开；其他调用方发来的 `x-vsr-debug` 会被忽略。

| `debug` | 谁能拿到调试头 |
| ------- | -------------- |
| `none`（默认） | 任何人都不能。 |
| `any` | 每个发送 `x-vsr-debug: true` 的客户端。用于开发和测试，或者放在会替不受信任的客户端删掉该头的网关后面。 |
| `authenticated` | 只有 `debug_principals` 里的用户和组，以你的认证组件设置的身份头为准（`global.services.authz.identity`）。 |

```yaml
global:
  router:
    headers:
      debug: authenticated
      debug_principals:
        users: [oncall-bot]
        groups: [platform-sre]
```

`authenticated` 信任身份头。standalone 模式的 Router 只在设置了 `identity.trust_headers` 的 listener 上保留它们；没有任何 listener 信任身份头时，Router 拒绝 `authenticated`。放在基于 Envoy 的网关之后时，网关必须删掉客户端发来的身份头。Dashboard Playground、`vllm-sr route --debug` 和 sr-bench 会请求调试头；调用方不被允许时，它们只显示默认面。

## 响应警告

`x-vsr-response-warnings` 按固定顺序列出本次 completion 的响应质量警告码，没有适用警告时省略。每条警告的细节（例如幻觉跨度或越狱置信度）保留在回放记录中，而不是展开到响应头。

## 决策头

最终路由事实使用公开面。中间细节（包括路由学习可观测性）需要 `x-vsr-debug`。

- 没有决策匹配、请求转到默认模型时，省略 `x-vsr-selected-decision`；`x-vsr-response-path` 仍为 `upstream`，`x-vsr-selected-model` 给出默认模型。
- 当决策基于策略叶子、聚合多个证据叶子，或通过错误策略解析时，`x-vsr-selected-confidence` 缺省。
- `x-vsr-fallback-attempts` 在两种网关模式下计数方式相同；主模型直接返回时省略。
- `x-vsr-decision-ranking` 是说明性文字；请改为解析路由回放中的 `decision_ranking`。只有一个决策匹配时省略。

用于 UI 展示时，把 `x-vsr-learning-actions` 翻译成面向用户的短语，例如 `tool/protocol pinned`、`model switched` 或 `learning bypassed`。新对话或会话开始的诊断通常只在调试视图中有用，应显示为中性状态文本，而不是主要路由状态。

这两个有效 token 头只在自动输出已解析时，随成功的上游响应一起发出。流式和缓冲响应的初始头中都可获得它们。显式输出上限、缓存命中、跳过处理以及 Looper 响应会省略它们。它们的值来自请求变更和后端渲染之后的最终分发；它们不是估计值，也不是模型配置的最大上下文长度。

## 匹配信号头

匹配信号头包含逗号分隔的规则名。它们需要 `x-vsr-debug`，且在该信号族未匹配时省略。由决策模型回答的规则（应用信号或 `systemone` 规则）在匹配时以规则名列出；没有 labels 的 choice 规则由条件按 label 引用，列为 `<rule>:<option>`，即其答案选中的选项，例如 `task_difficulty:hard`。未使其规则匹配的选项从不列出。`x-vsr-matched-projections` 列出匹配该请求的投影映射输出。

投影分数和完整投影追踪保存在路由回放记录中，而不是展开到响应头。用 `x-vsr-replay-id` 在控制面板或经身份验证的 Router 管理 API 中检查这些细节；公开推理监听器不提供回放记录。

## 留存头

当匹配决策发出留存指令时，调试响应会暴露已设置的字段。这些头帮助运维人员验证策略接线；客户端不应把它们当作命令。

未设置的字段会省略。缓存命中不发出这些头，因为该响应没有评估决策。

## 跨模型 KV 传输头（issue #2976）

当 Router 在模型切换时尝试跨模型 KV 复用时，它会在上游调用上注入请求侧提示：`x-vsr-kv-source-pod`、`x-vsr-kv-cache-id` 和 `x-vsr-kv-mapper-id`。目标 vLLM KVConnector 插件在响应的 `x-vsr-kv-transfer-status` 中报告结果；缺省即 `unsupported`。这些头是 Router 与推理池内部的；客户端不应依赖它们。

## 成本头

在缓冲（非流式）响应上，Router 用已服务模型的 `pricing` 配置，对模型报告的用量计价。这是配置价格数字，不是提供方账单。流式响应和没有 `pricing` 的模型会省略这两个头。

## 缓存与插件头

`x-vsr-cache-hit` 和 `x-vsr-fast-response` 在公开面上标识立即响应。缓存相似度、工具选择指标和 prompt-cache 回执需要 `x-vsr-debug`。

prompt-cache 头不含内容：它们只报告计数和结果，从不包含缓存文本本身，也从不声称缓存命中或提供方侧节省。

## 响应示例

公开面 — 关键头、最终路由事实和回放 id 入口：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-config-version: 7
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-replay-id: replay_01J...
```

这里的 `x-vsr-replay-id` 之所以出现，是因为启用了 Router Replay 且记录已持久化。回放被禁用或持久化失败时，响应会携带其他头并省略该头。

`global.router.headers.debug` 允许的调用方在请求中带 `x-vsr-debug: true` 时，被降级的中间细节和匹配信号也会内联发出：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-session-phase: tool_loop
x-vsr-matched-context: long-context
x-vsr-matched-projections: use-reasoning-model
x-vsr-replay-id: replay_01J...
```

## 身份头 {#identity-headers}

这些头由 Router 前面的认证组件设置，Router 用它们做角色绑定、限流、记忆，以及 user、team、tenant 三种响应缓存范围。除非 listener 信任身份头（见 [Gateway 模式](../installation/gateway-modes#identity-headers)），否则客户端发来的这些头会被丢弃；任何身份头都不会到达模型提供方。

| 头 | 说明 |
| --- | --- |
| `x-authz-user-id` | 已认证的用户。可用 `global.services.authz.identity.user_id_header` 改用其他头。 |
| `x-authz-user-groups` | 用户所属的组，逗号分隔。可用 `global.services.authz.identity.user_groups_header` 改用其他头。 |
| `x-authz-team-id` | 团队，用于 `team` 响应缓存范围。 |
| `x-authz-tenant-id` | 租户，用于 `tenant` 响应缓存范围。 |

## 兼容性与解读

- 解析可选头之前先看 `x-vsr-schema-version`；当前值为 `2`。未由响应缓存返回的路由推理响应携带它。改由 Router 自身处理器应答的响应（例如 `GET /v1/models`）从不携带该头，因此那里缺失属于预期，而不是合约违规。
- `x-vsr-matched-projections` 是投影头。单数形式不属于公开合约。
- `x-vsr-config-version` 计数的是单个 Router 的激活次数，每个 Router 保留自己的配置历史。要检查各副本是否服务同一配置，请比较 `GET /api/v1/config/hash` 报告的 `active_runtime_hash` 文档哈希。
- 配方名限定本地信号、投影、决策、缓存、回放、指标以及学习/会话身份。把响应与 Insights 或指标关联时，把 `x-vsr-selected-recipe` 与本地决策/信号名一起使用。
- `event` 是决策和 DSL 使用的公开信号类型。规范 YAML 把 event 规则存在 `routing.signals.events` 下，与其他复数信号容器一致。
- 路由学习在内部使用 Router 自有的在线状态。用户通过 `global.router.learning.adaptation` 启用在线模型选择学习，通过 `global.router.learning.protection` 启用稳定性保护，传入稳定身份头，并可选设置 `routing.decisions[].adaptations.mode`、组件模式或 `adaptations.adaptation.candidate_set`。`scope: conversation` 保护一个 `x-conversation-id`；`scope: session` 保护更广的 `x-session-id`。旧的 `routing.decisions[].algorithm.session_aware` 形态不属于公开合约。

## 头参考

下表由 Router 的头注册表 `src/semantic-router/pkg/headers/registry.go` 生成；修改注册表后运行 `make headers-generate`。头名、方向、面和敏感度保持英文原文。

- **Direction（方向）** — `request`（客户端到 Router）、`response`（Router 到客户端）、`provider request`（Router 到提供方）和 `provider response`（提供方到 Router）。
- **Surface（面）** — `public` 出现在默认响应上；`debug` 只给获准查看调试头的调用方；`control` 头由客户端发送，用来控制请求。`internal` 头不属于客户端合约，Router 写出的任何响应都不携带它们。
- **Sensitivity（敏感度）** — `identity` 值标识已认证的调用方，`secret` 值是凭据。二者都不会到达客户端；每用户密钥只会作为提供方的凭据到达提供方。
- **Since（起始版本）** — 该头首次发布的版本。

<!-- BEGIN-GENERATED-HEADER-REFERENCE -->

| Header | Direction | Surface | Sensitivity | Owner | Since | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `x-content-type-options` | response | public | none | router_replay | v0.4 | `nosniff` on Router Replay management responses. |
| `x-envoy-overloaded` | response | public | none | upstream | v0.5 | `true` on the 503 a full circuit breaker returns, as Envoy sets it. |
| `x-envoy-upstream-service-time` | response | public | none | gateway_parity | v0.5 | Envoy's upstream time on responses in Envoy mode; gateway-parity checks ignore its value. |
| `x-ratelimit-limit` | response | public | none | rate_limit | v0.2 | Limit of the rate-limit window that rejected the request. |
| `x-ratelimit-remaining` | response | public | none | rate_limit | v0.2 | Requests left in that window. |
| `x-ratelimit-reset` | response | public | none | rate_limit | v0.2 | Unix time the window resets at. |
| `x-request-id` | request, response, provider request | public | none | observability | v0.1 | Request correlation id, forwarded to providers; the native gateway adds one when absent, and the management API echoes it. |
| `x-vsr-applied-unknown-policy` | response | public | none | routing | v0.4 | Decisions whose unknown result `rules.on_unknown` resolved, as `decision=policy` pairs; also on the `fail_request` 503. |
| `x-vsr-cache-hit` | response | public | none | response_cache | v0.1 | `true` when the response cache served the response. |
| `x-vsr-client-protocol` | response | public | none | protocol | v0.4 | Client wire protocol, such as `anthropic`; on cross-protocol responses and with the debug headers. |
| `x-vsr-config-hash` | response | public | none | config | v0.4 | SHA-256 of the active runtime document, on routing previews, configuration activations and config-bound evaluation requests. |
| `x-vsr-config-version` | response | public | none | config | v0.5 | Version of the configuration snapshot that served the request. |
| `x-vsr-cost` | response | public | none | cost | v0.4 | Usage priced with the served model's configured pricing; buffered responses only. |
| `x-vsr-cost-currency` | response | public | none | cost | v0.4 | Currency of `x-vsr-cost`, from `pricing.currency`. |
| `x-vsr-effective-input-tokens` | response | public | none | automatic_output | v0.4 | Rendered input tokens of the final automatic-output dispatch, chat template included. |
| `x-vsr-effective-max-output-tokens` | response | public | none | automatic_output | v0.4 | Output token limit of the final automatic-output dispatch, reasoning included. |
| `x-vsr-fallback-attempts` | response | public | none | fallback | v0.4 | Attempts cross-model fallback made, the primary included, when a fallback candidate served the response. |
| `x-vsr-fast-response` | response | public | none | fast_response | v0.2 | `true` when the `fast_response` plugin answered without a provider call. |
| `x-vsr-inference-call-count` | response | public | none | benchmark | v0.4 | Provider calls a config-bound evaluation request made. |
| `x-vsr-model-usage` | response | public | none | benchmark | v0.4 | Per-model usage receipt of a config-bound evaluation request. |
| `x-vsr-protocol-warnings` | response | public | none | protocol | v0.4 | Protocol translation warnings, as comma-separated `severity;reason;field` entries. |
| `x-vsr-replay-id` | response | public | none | router_replay | v0.2 | Router Replay record id, when the record was persisted. |
| `x-vsr-response-path` | response | public | none | response_contract | v0.4 | How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`. |
| `x-vsr-response-warnings` | response | public | none | response_guard | v0.4 | Codes of the `response_guard` rules that annotated the answer, in rule order: `hallucination`, `unverified_factual`, `response_jailbreak` or `response_guard`. |
| `x-vsr-routing-latency-ms` | response | public | none | routing | v0.4 | Time spent choosing the model, in milliseconds, such as `0.412`. |
| `x-vsr-schema-version` | response | public | none | response_contract | v0.4 | Response-header contract revision; currently `2`. |
| `x-vsr-selected-algorithm` | response | public | none | routing | v0.4 | Model-selection algorithm used after the decision matched, such as `static`. |
| `x-vsr-selected-confidence` | response | public | none | routing | v0.2 | Model-derived score of the selected decision, such as `0.9100`. |
| `x-vsr-selected-decision` | response | public | none | routing | v0.1 | Decision the decision engine selected; omitted when no decision matched. |
| `x-vsr-selected-model` | response | public | none | routing | v0.1 | Model alias the Router selected. |
| `x-vsr-selected-recipe` | response | public | none | routing | v0.4 | Recipe the effective entrypoint selected; omitted for concrete backend models. |
| `x-vsr-upstream-protocol` | response | public | none | protocol | v0.4 | Provider wire protocol; on cross-protocol responses and with the debug headers. |
| `x-vsr-cache-similarity` | response | debug | none | response_cache | v0.3 | Similarity score of the response-cache lookup. |
| `x-vsr-context-token-count` | response | debug | none | signals | v0.2 | Token count the `context` signal measured. |
| `x-vsr-decision-ranking` | response | debug | none | routing | v0.5 | The matched decision the selected one beat, and the comparison that settled it. |
| `x-vsr-injected-system-prompt` | response | debug | none | system_prompt | v0.1 | Whether a `system_prompt` plugin injected text into the request. |
| `x-vsr-learning-actions` | response | debug | none | router_learning | v0.4 | Learning action per method, such as `adaptation=propose_switch`. |
| `x-vsr-learning-methods` | response | debug | none | router_learning | v0.4 | Router Learning methods the learning headers summarize, such as `adaptation,protection`. |
| `x-vsr-learning-reasons` | response | debug | none | router_learning | v0.4 | Reason per method, such as `adaptation=sampled_win`. |
| `x-vsr-learning-scopes` | response | debug | none | router_learning | v0.4 | Identity scope per method, such as `protection=conversation`. |
| `x-vsr-looper-algorithm` | response | debug | none | looper | v0.2 | Looper algorithm, such as `confidence`. |
| `x-vsr-looper-completion-tokens` | response | debug | none | looper | v0.4 | Completion tokens across the Looper's calls. |
| `x-vsr-looper-iterations` | response | debug | none | looper | v0.2 | Model calls the Looper made. |
| `x-vsr-looper-latency-ms` | response | debug | none | looper | v0.4 | Looper wall-clock latency, in milliseconds. |
| `x-vsr-looper-model` | response | debug | none | looper | v0.2 | Final model of the Looper. |
| `x-vsr-looper-models-used` | response | debug | none | looper | v0.2 | Models the Looper called, comma-separated. |
| `x-vsr-looper-prompt-tokens` | response | debug | none | looper | v0.4 | Prompt tokens across the Looper's calls. |
| `x-vsr-looper-total-tokens` | response | debug | none | looper | v0.4 | Total tokens across the Looper's calls. |
| `x-vsr-matched-action` | response | debug | none | signals | v0.5 | Matched `action` rule. |
| `x-vsr-matched-authz` | response | debug | none | signals | v0.2 | Matched `authz` rules. |
| `x-vsr-matched-complexity` | response | debug | none | signals | v0.2 | Matched `complexity` rules, and `rule:option` for a rule without labels, such as `task_difficulty:hard`. |
| `x-vsr-matched-context` | response | debug | none | signals | v0.2 | Matched `context` rules. |
| `x-vsr-matched-conversation` | response | debug | none | signals | v0.3 | Matched `conversation` rules. |
| `x-vsr-matched-decision-model` | response | debug | none | signals | v0.5 | Matched `systemone` rules, and `rule:option` for a choice rule without labels. |
| `x-vsr-matched-domains` | response | debug | none | signals | v0.1 | Matched `domain` rules. |
| `x-vsr-matched-event` | response | debug | none | signals | v0.3 | Matched `event` rules. |
| `x-vsr-matched-fact-check` | response | debug | none | signals | v0.1 | Matched `fact_check` rules. |
| `x-vsr-matched-hallucination` | response | debug | none | signals | v0.4 | Matched `hallucination` rules, once the answer is checked. |
| `x-vsr-matched-input-modality` | response | debug | none | signals | v0.4 | Matched `input_modality` rules. |
| `x-vsr-matched-jailbreak` | response | debug | none | signals | v0.2 | Matched `jailbreak` rules, of the request and of the response. |
| `x-vsr-matched-keywords` | response | debug | none | signals | v0.1 | Matched `keyword` rules, comma-separated. |
| `x-vsr-matched-language` | response | debug | none | signals | v0.2 | Matched `language` rules. |
| `x-vsr-matched-modality` | response | debug | none | signals | v0.3 | Matched `modality` rules, and `rule:option` for a rule without labels. |
| `x-vsr-matched-pii` | response | debug | none | signals | v0.2 | Matched `pii` rules. |
| `x-vsr-matched-projections` | response | debug | none | signals | v0.3 | Matched projection outputs. |
| `x-vsr-matched-safety` | response | debug | none | signals | v0.4 | Matched content `safety` rules. |
| `x-vsr-matched-structure` | response | debug | none | signals | v0.3 | Matched `structure` rules. |
| `x-vsr-matched-turn-relation` | response | debug | none | signals | v0.5 | Matched `turn_relation` rules, and `rule:option` for a rule without labels. |
| `x-vsr-matched-user-feedback` | response | debug | none | signals | v0.1 | Matched `user_feedback` rules, and `rule:option` for a rule without labels. |
| `x-vsr-prompt-cache-action` | response | debug | none | prompt_cache | v0.4 | Outcome of the `prompt_cache` plugin: `inserted`, `preserved`, `skipped` or `rejected`. |
| `x-vsr-prompt-cache-inserted` | response | debug | none | prompt_cache | v0.4 | Cache markers the Router inserted, at most `2`. |
| `x-vsr-prompt-cache-preserved` | response | debug | none | prompt_cache | v0.4 | Cache markers the caller supplied. |
| `x-vsr-prompt-cache-reason` | response | debug | none | prompt_cache | v0.4 | Reason for an outcome other than `inserted`: `caller_markers`, `no_eligible_target` or `unsupported_target`. |
| `x-vsr-retention-drop` | response | debug | none | retention | v0.4 | Whether the matched decision's retention directive keeps the response out of the response cache. |
| `x-vsr-retention-keep-current-model` | response | debug | none | retention | v0.4 | Whether the directive asks later routing to keep the current model. |
| `x-vsr-retention-prefer-prefix` | response | debug | none | retention | v0.4 | Whether the directive prefers prefix retention. |
| `x-vsr-retention-ttl-turns` | response | debug | none | retention | v0.4 | Retention lifetime the directive sets, in conversation turns. |
| `x-vsr-selected-category` | response | debug | none | classification | v0.1 | Domain category the classifier chose, such as `math`. |
| `x-vsr-selected-modality` | response | debug | none | classification | v0.2 | Modality result and optional method, such as `AR;classifier`. |
| `x-vsr-selected-reasoning` | response | debug | none | classification | v0.1 | Reasoning mode chosen for the request: `on` or `off`. |
| `x-vsr-session-phase` | response | debug | none | router_learning | v0.3 | Protection phase: `user_turn`, `tool_loop` or `provider_state`. |
| `x-vsr-tools-confidence` | response | debug | none | tools | v0.3 | Highest similarity score among the selected tools. |
| `x-vsr-tools-latency-ms` | response | debug | none | tools | v0.3 | Tool-selection retriever latency, in milliseconds. |
| `x-vsr-tools-strategy` | response | debug | none | tools | v0.3 | Retriever strategy of semantic tool selection. |
| `x-claude-code-session-id` | request, provider request | control | none | session | v0.3 | Conversation id Claude Code sends; `x-session-id` takes precedence, and providers receive it. |
| `x-conversation-id` | request, provider request | control | none | router_learning | v0.4 | Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it. |
| `x-disable-router-memory` | request | control | none | memory | v0.3 | `true` keeps Router-managed memory out of the request. |
| `x-session-id` | request, provider request | control | none | session | v0.3 | Client session id; Router Learning `scope: session` protects it, and providers receive it. |
| `x-sr-bench-expected-config-hash` | request | control | none | benchmark | v0.4 | Configuration hash an evaluation request requires; another active configuration fails it with 412. |
| `x-sr-bench-max-inference-calls` | request | control | none | benchmark | v0.4 | Provider-call limit of an evaluation request; only `1` is supported. |
| `x-vsr-cache-control` | request | control | none | response_cache | v0.4 | Response-cache directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-compression-control` | request | control | none | context_compression | v0.4 | Context-compression directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-debug` | request | control | none | debug_headers | v0.4 | `true` asks for the debug headers on the response; honored only for a caller `global.router.headers.debug` allows. |
| `x-vsr-protection-release` | request | control | none | router_learning | v0.5 | Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it. |
| `x-vsr-skip-processing` | request | control | none | skip_processing | v0.3 | `true` opts the request out of Router processing when `global.router.skip_processing.enabled` is set. |
| `api-key` | request, provider request | internal | secret | credentials | v0.2 | Azure OpenAI key: a client's authenticates to the Router and is removed, and the Router sets the provider's. |
| `api-subscription-key` | provider request | internal | secret | credentials | v0.4 | Provider credential a catalog provider names. |
| `authorization` | request, provider request | internal | secret | credentials | v0.1 | Bearer credential: a client's key for the Router, and the provider credential the Router sets. |
| `ocp-apim-subscription-key` | provider request | internal | secret | credentials | v0.5 | Azure API Management credential, which a model connection sets through `auth`. |
| `x-api-key` | provider request | internal | secret | credentials | v0.2 | Provider credential of Anthropic-style APIs. |
| `x-authz-team-id` | request | internal | identity | authz | v0.4 | Authenticated team id; partitions the response cache. |
| `x-authz-tenant-id` | request | internal | identity | authz | v0.4 | Authenticated tenant id; partitions the response cache. |
| `x-authz-user-groups` | request | internal | identity | authz | v0.2 | Authenticated user's groups, comma-separated; `identity.user_groups_header` names another header. |
| `x-authz-user-id` | request | internal | identity | authz | v0.2 | Authenticated user id; `identity.user_id_header` names another header. |
| `x-envoy-expected-rq-timeout-ms` | request | internal | none | ingress | v0.5 | Envoy timeout control; the edge drops it from client requests. |
| `x-envoy-force-trace` | request | internal | none | ingress | v0.5 | Envoy tracing control; the edge drops it from client requests. |
| `x-envoy-hedge-on-per-try-timeout` | request, provider request | internal | none | benchmark | v0.4 | `false` on evaluation requests, so a per-try timeout never hedges. |
| `x-envoy-internal` | request | internal | none | ingress | v0.5 | Envoy's internal-request marker; the edge drops it from client requests. |
| `x-envoy-ip-tags` | request | internal | none | ingress | v0.5 | Envoy IP tagging control; the edge drops it from client requests. |
| `x-envoy-max-retries` | request, provider request | internal | none | reliability | v0.4 | Retry count the Router sets from the decision's `reliability` block, and `0` on evaluation requests. |
| `x-envoy-original-url` | request | internal | none | ingress | v0.5 | Envoy redirect control; the edge drops it from client requests. |
| `x-envoy-ratelimited` | provider response | internal | none | upstream | v0.5 | Marks a provider response an Envoy rate limit rejected; the native upstream can retry it. |
| `x-envoy-retriable-header-names` | request | internal | none | ingress | v0.5 | Envoy retry control; the edge drops it from client requests. |
| `x-envoy-retriable-status-codes` | request, provider request | internal | none | reliability | v0.5 | Retriable status codes the Router sets from the decision's `reliability` block. |
| `x-envoy-retry-grpc-on` | request | internal | none | ingress | v0.5 | Envoy retry control; the edge drops it from client requests. |
| `x-envoy-retry-on` | request, provider request | internal | none | reliability | v0.5 | Retry conditions the Router sets from the decision's `reliability` block. |
| `x-envoy-upstream-alt-stat-name` | request | internal | none | ingress | v0.5 | Envoy stats control; the edge drops it from client requests. |
| `x-envoy-upstream-rq-per-try-timeout-ms` | request, provider request | internal | none | reliability | v0.5 | Per-try timeout the Router sets from the decision's `reliability` block. |
| `x-envoy-upstream-rq-timeout-alt-response` | request | internal | none | ingress | v0.5 | Envoy timeout control; the edge drops it from client requests. |
| `x-envoy-upstream-rq-timeout-ms` | request, provider request | internal | none | reliability | v0.5 | Route timeout the Router sets from the decision's `reliability` block. |
| `x-forwarded-for` | request, provider request | internal | none | native_gateway | v0.5 | Client address chain; the native gateway's access log records it, and providers receive it. |
| `x-forwarded-proto` | request, provider request | internal | none | native_gateway | v0.5 | Scheme of the client connection; the native gateway sets it when absent, as Envoy does. |
| `x-goog-api-key` | provider request | internal | secret | credentials | v0.4 | Provider credential of Google APIs. |
| `x-selected-model` | request, provider request | internal | none | routing | v0.1 | Route key: the model whose route the gateway sends the provider request on; the Router replaces or drops a client's value. |
| `x-user-anthropic-key` | request | internal | secret | authz | v0.2 | Per-user Anthropic key the authenticator injects. |
| `x-user-azure-openai-key` | request | internal | secret | authz | v0.2 | Per-user Azure OpenAI key the authenticator injects. |
| `x-user-bedrock-key` | request | internal | secret | authz | v0.2 | Per-user AWS Bedrock token the authenticator injects. |
| `x-user-cloudflare-workers-ai-key` | request | internal | secret | authz | v0.4 | Per-user Cloudflare Workers AI token the authenticator injects. |
| `x-user-gemini-key` | request | internal | secret | authz | v0.2 | Per-user Google Gemini key the authenticator injects. |
| `x-user-minimax-key` | request | internal | secret | authz | v0.3 | Per-user MiniMax key the authenticator injects. |
| `x-user-openai-key` | request | internal | secret | authz | v0.2 | Per-user OpenAI key the authenticator injects. |
| `x-user-vertex-ai-key` | request | internal | secret | authz | v0.2 | Per-user Vertex AI token the authenticator injects. |
| `x-vsr-fusion-depth` | request | internal | none | looper | v0.4 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-gateway-credential` | request | internal | secret | authz | v0.5 | Dedicated credential the authenticator attaches for a trusted upstream gateway; only a backend marked `trusted_gateway` receives its value, under its own auth header, and `gateway_credential.header` names another header. |
| `x-vsr-internal-auth` | request | internal | secret | looper | v0.4 | In-process request authentication; the Router drops it from every request. |
| `x-vsr-kv-cache-id` | provider request | internal | none | kv_transfer | v0.4 | Opaque id of the source KV block. |
| `x-vsr-kv-mapper-id` | provider request | internal | none | kv_transfer | v0.4 | Mapper artifact for the source and target model pair. |
| `x-vsr-kv-source-pod` | provider request | internal | none | kv_transfer | v0.4 | Address of the pod that holds the source model's KV cache. |
| `x-vsr-kv-transfer-status` | provider response | internal | none | kv_transfer | v0.4 | KV reuse outcome the target pod reports: `applied`, `fallback_reprefill` or `unsupported`. |
| `x-vsr-looper-decision` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-iteration` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-request` | request | internal | none | looper | v0.2 | Retired Looper hop header; the Router drops it from every request. |
| `x-vsr-looper-secret` | request | internal | secret | looper | v0.3 | Retired Looper hop secret; the Router drops it from every request. |
| `x-vsr-outcome-principal` | request | internal | identity | router_learning | v0.4 | Opaque attested identity that isolates outcome-ingest rate limits; never persisted. |
| `x-vsr-outcome-source` | request | internal | none | router_learning | v0.4 | Outcome provenance a trusted control plane attests to the management API. |
| `x-vsr-systemone-backend` | request, provider request | internal | none | systemone | v0.5 | Marks a System One request one Router forwards to another, which refuses to forward it again. |
| `x-vsr-upstream-host` | provider response | internal | none | kv_transfer | v0.5 | Endpoint Envoy reports on the provider response for KV registry writes; removed before the client. |

<!-- END-GENERATED-HEADER-REFERENCE -->
