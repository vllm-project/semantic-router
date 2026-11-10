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

路由流水线产生的每个响应都携带公开集合：上游模型的应答、响应缓存命中，以及 Router 自己产生的响应，例如拦截、快速响应或路由错误。

| 头 | 含义 |
| --- | --- |
| `x-vsr-schema-version` | 本头合约的修订号，`3`。 |
| `x-vsr-config-version` | 服务该请求的配置版本。 |
| `x-vsr-response-path` | 由谁应答：`upstream`、`cache`、`fast_response`、`looper`、`image_generation`、`blocked`、`rate_limited`、`error` 或 `fallback`。 |
| `x-vsr-selected-recipe`、`x-vsr-selected-decision`、`x-vsr-selected-model` | 最终路由事实，路由走到哪一步就给出哪一项。 |
| `x-vsr-replay-id` | Router Replay 记录；仅当启用回放且记录已持久化时出现。 |
| `x-vsr-warnings` | [响应警告](#response-warnings)；没有适用警告时省略。 |
| `x-vsr-cost`、`x-vsr-cost-currency` | 缓冲响应的[配置价格](#cost-headers)。 |

- 没有决策匹配、请求转到默认模型时，省略 `x-vsr-selected-decision`；此时 `x-vsr-selected-model` 给出默认模型。
- 请求始终保留开始时的配置版本，即使运行期间热重载激活了另一个版本。由 Router 自身处理器应答的响应（例如 `GET /v1/models`）只携带 `x-vsr-config-version`。
- 唯一设置 `x-vsr-*` 头的管理端点是 `POST /api/v1/routing/preview`：它在结果成功时返回 `x-vsr-config-hash`。

Router 关于请求知道的其他一切，例如匹配的信号、置信度和算法，都放在一个调试头 [`x-vsr-trace`](#the-trace) 里。启用回放时，同样的上下文仍可通过 `x-vsr-replay-id` 获得。

客户端用参考表中的 `control` 头来控制请求，例如 `x-vsr-session-id`、`x-vsr-memory` 或 `x-vsr-debug`。客户端发来的 Router 其他 `x-vsr-*` 头，以及提供方返回的每个 `x-vsr-*` 头，都会被 Router 丢弃。

只有参考表中方向包含 `provider request` 的头才会到达提供方：Router 会从每个发往提供方的请求中删掉它自己的控制头、内部头、身份头和密钥。

## 谁可以打开追踪头 {#who-may-turn-on-the-trace}

追踪头会列出请求命中的信号规则和防护规则、决策排名以及路由学习的原因，所以客户端不能自己打开它。客户端用 `x-vsr-debug: true` 请求它，由 `global.router.headers.debug` 决定谁可以；其他调用方发来的 `x-vsr-debug` 会被忽略。

| `debug` | 谁能拿到追踪头 |
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

`authenticated` 信任身份头。standalone 模式的 Router 只在设置了 `identity.trust_headers` 的 listener 上保留它们；没有任何 listener 信任身份头时，Router 拒绝 `authenticated`。放在基于 Envoy 的网关之后时，网关必须删掉客户端发来的身份头。Dashboard Playground、`vllm-sr route --debug` 和 sr-bench 会请求追踪头；调用方不被允许时，它们只显示公开集合。

## 追踪头 {#the-trace}

`x-vsr-trace` 是一个不超过 4 KB 的 [RFC 8941](https://www.rfc-editor.org/rfc/rfc8941) 字典。头参考之后的表按优先级从高到低列出它的成员；不适用的成员会省略。字符串会对 `%` 以及可打印 ASCII 之外的每个字节做百分号编码，所以去掉引号后要再解码。编码后的追踪头超过 4 KB 时，Router 从最低优先级开始省略成员，并在字典末尾加上 `truncated`。

```http
x-vsr-trace: confidence=0.91, algorithm=static, routing_latency_ms=3.2, signals=("context:long-context" "projection:use-reasoning-model"), session_phase=tool_loop
```

- 当决策基于策略叶子、聚合多个证据叶子，或通过错误策略解析时，`confidence` 缺省。
- `fallback_attempts` 在两种网关模式下计数方式相同；主模型直接返回时省略。
- 只有一个决策匹配时省略 `ranking`；完整排名请解析路由回放中的 `decision_ranking`。
- `learning` 列出路由学习的方法及其 `action`。展示时把这些动作翻译成面向用户的短语，例如 `tool/protocol pinned`、`model switched` 或 `learning bypassed`；新对话或会话开始的诊断通常只在调试视图中有用，应显示为中性状态文本。
- `effective_input_tokens` 和 `effective_max_output_tokens` 只在自动输出已解析时，随成功的上游响应一起出现；流式和缓冲响应的初始头中都可获得它们。显式输出上限、缓存命中、跳过处理以及 Looper 响应会省略它们。它们的值来自请求变更和后端渲染之后的最终分发；它们不是估计值，也不是模型配置的最大上下文长度。
- `config_hash`、`inference_calls` 和 `model_usage` 是绑定配置的评测请求（带 `x-vsr-bench-expected-config-hash` 发送的请求）的回执。
- `retention` 给出匹配决策的留存指令设置了哪些字段，便于运维人员验证策略接线；客户端不应把它们当作命令。
- `prompt_cache` 不含内容：它只报告计数和结果，从不包含缓存文本本身，也从不声称缓存命中或提供方侧节省。

投影分数和完整投影追踪保存在路由回放记录中，而不是展开到追踪头。用 `x-vsr-replay-id` 在控制面板或经身份验证的 Router 管理 API 中检查这些细节；公开推理监听器不提供回放记录。

## 响应警告 {#response-warnings}

`x-vsr-warnings` 是警告码的 RFC 8941 列表，没有适用警告时省略。先按规则顺序列出给应答加注的 `response_guard` 规则的警告码，再列出协议转换损失：

| 警告码 | 含义 |
| --- | --- |
| `hallucination` | 基于 `hallucination` 信号的规则标记了该应答。 |
| `unverified_factual` | 应答属于事实性内容，但没有可供核对的上下文。 |
| `response_jailbreak` | 基于响应阶段 `jailbreak` 信号的规则标记了该应答。 |
| `response_guard` | 基于其他信号的规则标记了该应答。 |
| `protocol` | 协议转换改动或丢弃了请求的一部分，附带 `action`、`reason` 和 `field`。 |
| `truncated` | 为使该头不超过 4 KB 而省略的警告，附带其数量 `count`。 |

每条警告的细节（例如幻觉跨度或越狱置信度）保留在回放记录中，而不是展开到响应头。

## 会话头

`x-vsr-session-id` 指明请求所属的会话：会话按它固定；除非 `identity.headers.session` 指定了别的头，路由学习也按它保护。已经用自己的头名发送会话 id 的客户端可以继续使用：`global.router.headers.session_id_headers` 列出 Router 同时接受的头名，默认是 `x-session-id` 和 `x-claude-code-session-id`。请求同时携带多个时，以 `x-vsr-session-id` 为准。

```yaml
global:
  router:
    headers:
      session_id_headers: [x-session-id, x-claude-code-session-id]
```

`x-conversation-id` 为路由学习的 `scope: conversation` 指明一个对话。`x-vsr-memory: off` 让请求不使用 Router 管理的记忆。

## 跨模型 KV 传输头（issue #2976）

当 Router 在模型切换时尝试跨模型 KV 复用时，它会在上游调用上注入请求侧提示：`x-vsr-kv-source-pod`、`x-vsr-kv-cache-id` 和 `x-vsr-kv-mapper-id`。这些头是 Router 与推理池内部的；客户端不应依赖它们。

## 成本头 {#cost-headers}

在缓冲（非流式）响应上，Router 用已服务模型的 `pricing` 配置，对模型报告的用量计价。这是配置价格数字，不是提供方账单。流式响应和没有 `pricing` 的模型会省略这两个头。

## 响应示例

公开集合：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 3
x-vsr-config-version: 7
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-model: reasoning-model
x-vsr-replay-id: replay_01J...
```

这里的 `x-vsr-replay-id` 之所以出现，是因为启用了 Router Replay 且记录已持久化。回放被禁用或持久化失败时，响应会携带其他头并省略该头。

`global.router.headers.debug` 允许的调用方在请求中带 `x-vsr-debug: true` 时，响应还会携带追踪头：

```http
x-vsr-trace: algorithm=static, routing_latency_ms=1.8, signals=("context:long-context" "projection:use-reasoning-model"), session_phase=tool_loop
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

- 解析其他头之前先看 `x-vsr-schema-version`；当前值为 `3`。由 Router 自身处理器应答的响应（例如 `GET /v1/models`）从不携带该头，因此那里缺失属于预期，而不是合约违规。
- `x-vsr-config-version` 计数的是单个 Router 的激活次数，每个 Router 保留自己的配置历史。要检查各副本是否服务同一配置，请比较 `GET /api/v1/config/hash` 报告的 `active_runtime_hash` 文档哈希。
- 配方名限定本地信号、投影、决策、缓存、回放、指标以及学习/会话身份。把响应与 Insights 或指标关联时，把 `x-vsr-selected-recipe` 与本地决策/信号名一起使用。
- `event` 是决策和 DSL 使用的公开信号类型。规范 YAML 把 event 规则存在 `recipes[].signals.events` 下，与其他复数信号容器一致。
- 路由学习在内部使用 Router 自有的在线状态。用户通过 `global.router.learning.adaptation` 启用在线模型选择学习，通过 `global.router.learning.protection` 启用稳定性保护，传入稳定身份头，并可选设置 `recipes[].decisions[].adaptations.mode`、组件模式或 `adaptations.adaptation.candidate_set`。`scope: conversation` 保护一个 `x-conversation-id`；`scope: session` 保护更广的 `x-vsr-session-id` 会话。旧的 `recipes[].decisions[].algorithm.session_aware` 形态不属于公开合约。

## 头参考

下表由 Router 的头注册表 `src/semantic-router/pkg/headers/registry.go` 生成；修改注册表后运行 `make headers-generate`。头名、方向、面和敏感度保持英文原文。

- **Direction（方向）** — `request`（客户端到 Router）、`response`（Router 到客户端）、`provider request`（Router 到提供方）和 `provider response`（提供方到 Router）。
- **Surface（面）** — `public` 出现在默认响应上；`debug` 只给获准查看追踪头的调用方；`control` 头由客户端发送，用来控制请求。`internal` 头不属于客户端合约，Router 写出的任何响应都不携带它们。
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
| `x-vsr-config-hash` | response | public | none | config | v0.4 | SHA-256 of the active runtime document, on routing previews and configuration activations. |
| `x-vsr-config-version` | response | public | none | config | v0.5 | Version of the configuration snapshot that served the request. |
| `x-vsr-cost` | response | public | none | cost | v0.4 | Usage priced with the served model's configured pricing; buffered responses only. |
| `x-vsr-cost-currency` | response | public | none | cost | v0.4 | Currency of `x-vsr-cost`, from `pricing.currency`. |
| `x-vsr-replay-id` | response | public | none | router_replay | v0.2 | Router Replay record id, when the record was persisted. |
| `x-vsr-response-path` | response | public | none | response_contract | v0.4 | How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`. |
| `x-vsr-schema-version` | response | public | none | response_contract | v0.4 | Response-header contract revision; currently `3`. |
| `x-vsr-selected-decision` | response | public | none | routing | v0.1 | Decision the decision engine selected; omitted when no decision matched. |
| `x-vsr-selected-model` | response | public | none | routing | v0.1 | Model alias the Router selected. |
| `x-vsr-selected-recipe` | response | public | none | routing | v0.4 | Recipe the effective entrypoint selected; omitted for concrete backend models. |
| `x-vsr-warnings` | response | public | none | response_contract | v0.5 | Warnings about the response, as an RFC 8941 list: the codes of the `response_guard` rules that annotated the answer, then each `protocol` translation loss with its `action`, `reason` and `field`. |
| `x-vsr-trace` | response | debug | none | debug_headers | v0.5 | Routing details as an RFC 8941 dictionary of at most 4 KB, for a caller `global.router.headers.debug` allows; the members follow this table. |
| `x-claude-code-session-id` | request, provider request | control | none | session | v0.3 | Conversation id Claude Code sends, which `session_id_headers` accepts by default after `x-session-id`; providers receive it. |
| `x-conversation-id` | request, provider request | control | none | router_learning | v0.4 | Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it. |
| `x-session-id` | request, provider request | control | none | session | v0.3 | Client session id `session_id_headers` accepts by default; providers receive it. |
| `x-vsr-bench-expected-config-hash` | request | control | none | benchmark | v0.5 | Configuration hash an evaluation request requires; another active configuration fails it with 412. |
| `x-vsr-bench-max-inference-calls` | request | control | none | benchmark | v0.5 | Provider-call limit of an evaluation request; only `1` is supported. |
| `x-vsr-cache-control` | request | control | none | response_cache | v0.4 | Response-cache directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-compression-control` | request | control | none | context | v0.4 | Context-compression directives for the request, when the plugin's `request_controls` allow them. |
| `x-vsr-debug` | request | control | none | debug_headers | v0.4 | `true` asks for `x-vsr-trace` on the response; honored only for a caller `global.router.headers.debug` allows. |
| `x-vsr-memory` | request | control | none | memory | v0.5 | `off` keeps Router-managed memory out of the request. |
| `x-vsr-protection-release` | request | control | none | router_learning | v0.5 | Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it. |
| `x-vsr-session-id` | request | control | none | session | v0.5 | Session id the client pins; without it the Router reads the headers `global.router.headers.session_id_headers` names, in order. |
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
| `x-user-anthropic-key` | request | internal | secret | authz | v0.2 | Per-user Anthropic key the authenticator injects. |
| `x-user-azure-openai-key` | request | internal | secret | authz | v0.2 | Per-user Azure OpenAI key the authenticator injects. |
| `x-user-bedrock-key` | request | internal | secret | authz | v0.2 | Per-user AWS Bedrock token the authenticator injects. |
| `x-user-cloudflare-workers-ai-key` | request | internal | secret | authz | v0.4 | Per-user Cloudflare Workers AI token the authenticator injects. |
| `x-user-gemini-key` | request | internal | secret | authz | v0.2 | Per-user Google Gemini key the authenticator injects. |
| `x-user-minimax-key` | request | internal | secret | authz | v0.3 | Per-user MiniMax key the authenticator injects. |
| `x-user-openai-key` | request | internal | secret | authz | v0.2 | Per-user OpenAI key the authenticator injects. |
| `x-user-vertex-ai-key` | request | internal | secret | authz | v0.2 | Per-user Vertex AI token the authenticator injects. |
| `x-vsr-delegated-credential` | request | internal | secret | authz | v0.5 | The requesting user's credential the authenticator attaches; only an upstream with `auth.credential: delegated` receives its value, under its own auth header, and `delegated_credential.header` names another header. |
| `x-vsr-internal-auth` | request | internal | secret | looper | v0.4 | In-process request authentication; the Router drops it from every client request. |
| `x-vsr-kv-cache-id` | provider request | internal | none | kv_transfer | v0.4 | Opaque id of the source KV block. |
| `x-vsr-kv-mapper-id` | provider request | internal | none | kv_transfer | v0.4 | Mapper artifact for the source and target model pair. |
| `x-vsr-kv-source-pod` | provider request | internal | none | kv_transfer | v0.4 | Address of the pod that holds the source model's KV cache. |
| `x-vsr-outcome-principal` | request | internal | identity | router_learning | v0.4 | Opaque attested identity that isolates outcome-ingest rate limits; never persisted. |
| `x-vsr-outcome-source` | request | internal | none | router_learning | v0.4 | Outcome provenance a trusted control plane attests to the management API. |
| `x-vsr-route-model` | request, provider request | internal | none | routing | v0.5 | Route key: the model whose route the gateway sends the provider request on; the Router drops a client's value and sets its own. |
| `x-vsr-systemone-backend` | request, provider request | internal | none | systemone | v0.5 | Marks a System One request one Router forwards to another, which refuses to forward it again. |
| `x-vsr-upstream-host` | provider response | internal | none | kv_transfer | v0.5 | Endpoint Envoy reports on the provider response for KV registry writes; removed before the client. |

The members of `x-vsr-trace`, highest priority first:

| Member | Owner | Description |
| --- | --- | --- |
| `config_hash` | benchmark | SHA-256 of the runtime document that served a config-bound evaluation request. |
| `inference_calls` | benchmark | Provider calls a config-bound evaluation request made. |
| `model_usage` | benchmark | Per-model usage receipt of a config-bound evaluation request, as JSON. |
| `confidence` | routing | Model-derived score of the selected decision, such as `0.91`. |
| `algorithm` | routing | Selector or executor that chose the model after the decision matched, such as `static`. |
| `routing_latency_ms` | routing | Time spent choosing the model, in milliseconds. |
| `fallback_attempts` | fallback | Attempts cross-model fallback made, the primary included, when a fallback candidate served the response. |
| `on_unknown` | routing | Decisions whose unknown result `rules.on_unknown` resolved, each with its `policy`. |
| `client_protocol` | protocol | Client wire protocol, such as `anthropic`. |
| `upstream_protocol` | protocol | Provider wire protocol, such as `openai`. |
| `effective_input_tokens` | automatic_output | Rendered input tokens of the final automatic-output dispatch, chat template included. |
| `effective_max_output_tokens` | automatic_output | Output token limit of the final automatic-output dispatch, reasoning included. |
| `signals` | signals | Matched signal rules as `type:name`, and `type:rule:option` for a choice rule without labels; projections are `projection:name`. |
| `context_tokens` | signals | Token count the `context` signal measured. |
| `category` | classification | Domain category the classifier chose, such as `math`. |
| `reasoning` | classification | Reasoning mode chosen for the request: `on` or `off`. |
| `modality` | routing | Output kinds of the selected decision's output contract, such as `(image)` or `(text image)`; omitted for a decision that declares none. |
| `system_prompt` | system_prompt | Present when a `system_prompt` plugin injected text into the request. |
| `cache_similarity` | response_cache | Similarity score of the response-cache lookup. |
| `retention` | retention | The matched decision's retention directive: the `drop`, `ttl_turns`, `keep_current_model` and `prefer_prefix` it sets. |
| `prompt_cache` | prompt_cache | Outcome of the `prompt_cache` plugin, with its `reason` and the `inserted` and `preserved` marker counts. |
| `tools` | tools | Outcome of the `tools` plugin's selection, with its `candidates`, `kept` and `latency_ms`. |
| `response_guard` | response_guard | Outcome of the selected decision's `response_guard`, such as `blocked`, with `buffered` when the Router held a streamed answer until the guard read it. |
| `session_phase` | router_learning | Protection phase: `user_turn`, `tool_loop` or `provider_state`. |
| `learning` | router_learning | Router Learning methods, each with its `action`, `scope` and `reason`. |
| `looper` | looper | Models the Looper called, with its `algorithm`, final `model`, `iterations`, `latency_ms` and token counts. |
| `ranking` | routing | The matched decision the selected one beat, with the `reason` that settled it. |
| `truncated` | response_contract | Present when members were left out to keep the trace within 4 KB. |

<!-- END-GENERATED-HEADER-REFERENCE -->
