---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/api/router.md"
  outdated: false
---

# 路由器接口 {#router-api}

Router 数据面在配置的监听器上接收模型请求。默认的 standalone 模式下由 Router 自己服务这些请求；使用 `--gateway extproc` 时由 Envoy 服务，并通过 ext_proc 调用 Router。在标准本地栈中，监听器为 `http://localhost:8899`；配方可在 `listeners` 下选择不同地址或端口。

推理请使用数据面。健康检查、配置、诊断和回放查询请使用管理 API，通常绑定到 `127.0.0.1:8080`。见 [Router 管理 API](./apiserver)。

## 支持的推理路径 {#supported-inference-paths}

| 方法 | 路径 | 客户端格式 | 说明 |
| --- | --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions | 主要的路由推理端点 |
| `POST` | `/v1/responses` | OpenAI Responses | 需要启用 Responses 服务 |
| `GET` | `/v1/responses/{id}` | OpenAI Responses | 读取已存储的 response |
| `DELETE` | `/v1/responses/{id}` | OpenAI Responses | 删除已存储的 response |
| `GET` | `/v1/responses/{id}/input_items` | OpenAI Responses | 读取已存储的 input items |
| `POST` | `/v1/messages` | Anthropic Messages | 当所选后端使用其他协议时，Router 会做转换 |
| `POST` | `/openai/deployments/{deployment}/chat/completions` | Azure OpenAI Chat Completions | URL 中的 deployment 是 Router 模型名；接受 `api-version` |
| `POST` | `/openai/responses` | Azure OpenAI Responses | 接受带日期的 `api-version`；模型名在请求体中，需要启用 Responses 服务 |
| `POST` | `/openai/v1/responses` | Azure OpenAI Responses | 模型名在请求体中，需要启用 Responses 服务 |
| `POST` | `/openai/v1/chat/completions` | Azure OpenAI Chat Completions | 模型名在请求体中 |
| `GET` | `/v1/models` | OpenAI Models | 列出当前 Router 配置暴露的模型 |
| `POST` | `/v1/systemone`、`/v1/decisions` | 原生 System One | standalone listener 上显式发布的决策模型回答问题 |
| `GET` | `/v1/systemone/models` | 原生模型发现 | 列出该 listener 发布的 System One 模型 |

Engine 模式关闭配方路由，并提供原生 System One 接口。在
`listeners[].systemone.models` 中发布原生模型 ID；Chat 的 `models` 名单不会授权
原生访问。两类接口都使用 listener 的 API keys。完整请求见
[模型运行时快速开始](../model-runtime/quickstart.md)。

其他 `/v1/*` 路径默认拒绝。特别是 `/v1/files`、`/v1/vector_stores` 和路由回放路径在公网推理监听器上不可用。Router 自有的文件和向量存储操作使用管理监听器上的 `/api/v1/storage/files` 和 `/api/v1/storage/vector-stores`。其他 `/openai/*` 操作，例如 embeddings 和读取已存储的 response，返回 `404`。

客户端到后端的转换矩阵、后端 `api_format` 值以及字段级可移植边界，见[协议兼容性](../installation/protocol-compatibility)。

## 发送路由请求 {#send-a-routed-request}

希望 Router 选择后端时，使用 `vllm-sr/auto` 或显式声明的配方入口。希望绕过语义模型选择并直接打到某个模型时，使用具体模型名。

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that merges two sorted lists."
      }
    ]
  }'
```

响应保持客户端协议的形状。其模型、内容、token 用量以及可选的 Router 头取决于所选后端和配方。稳定的可观测性契约见 [VSR 路由头](../troubleshooting/vsr-headers)。

Router 接受的模型名来自规范 provider 条目。`name` 是决策和客户端使用的逻辑别名，`provider_model_id` 发送给上游 provider，`providers.models[].backend_refs[]` 标识物理端点：

```yaml
providers:
  models:
    - name: local-small
      provider_model_id: served-model
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0
        completion_per_1m: 0
      backend_refs:
        - name: local-vllm
          endpoint: model-server:8000
          protocol: http
          provider: vllm
          weight: 1
```

定价是运维提供的部署元数据，不是实时报价。它保留在 `providers.models[]` 上；`routing.modelCards` 只描述语义能力。`currency` 可选，省略时记账解析为 `USD`。若设置，必须是大写三字母代码。所有每百万 token 费率必须为有限且非负。`cached_input_per_1m` 和 `cache_write_per_1m` 可选，显式零表示免费费率。

`api_format` 声明上游线路契约：`openai` 对应 Chat Completions，`responses` 对应 OpenAI Responses API，`anthropic` 对应 Anthropic Messages。客户端可以使用任何受支持的推理路径；Router 在 provider 边界转换一次，并返回客户端原来的线路格式。

### Responses API {#responses-api}

```bash
curl -sS http://localhost:8899/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "input": "Summarize the trade-offs of retrieval-augmented generation."
  }'
```

创建、检索和删除 Responses API 对象需要其配套服务和存储。禁用 Responses API 支持时，集合端点返回 `404`，已存储对象的处理也不可用。已配置的服务按其自身的存储和保留设置保留对象。

### Anthropic Messages {#anthropic-messages}

```bash
curl -sS http://localhost:8899/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "vllm-sr/auto",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": "Explain semantic routing in one paragraph."
      }
    ]
  }'
```

协议转换仅限于 Router 支持的字段。请求跨协议时，检查 `x-vsr-client-protocol`、`x-vsr-upstream-protocol` 以及任何 `x-vsr-protocol-warnings` 响应头。

### Azure OpenAI 客户端 {#azure-openai-clients}

deployment Chat 路径从 URL 读取模型名；Responses 和 v1 Chat 路径从请求体读取模型名。监听器配置 `api_keys` 时，Router 用客户端的 `api-key` 验证请求，并在转发给 provider 前移除该请求头。

GitHub Copilot CLI 使用 Azure 模式时，设置 `COPILOT_PROVIDER_TYPE=azure`、指向监听器的 `COPILOT_PROVIDER_BASE_URL`，以及作为 Router 模型名的 `COPILOT_PROVIDER_WIRE_MODEL`。设置 `COPILOT_PROVIDER_WIRE_API=responses` 后，CLI 使用 `/openai/v1/responses`；设置 `COPILOT_PROVIDER_AZURE_API_VERSION` 后使用 `/openai/responses`。Router 接受 Responses 请求中的 `reasoning.summary`：对 Responses 后端会转发该设置；对 Chat Completions 或 Messages 后端仍会处理请求，但丢弃摘要设置并在 `x-vsr-protocol-warnings` 中说明。

## 路由错误 {#routing-errors}

Router 无法路由某个请求时，会直接应答该请求，不调用任何后端。错误采用客户端所用的协议。在 OpenAI Chat Completions 和 Responses 的错误中，`error.code` 是稳定的原因码，`error.message` 是简短消息。除预算错误外，消息不包含模型、决策或请求内容：

```json
{"error":{"type":"invalid_request_error","code":"no_route","message":"no route matched the request","param":null}}
```

| 代码 | 状态码 | `error.type` | 含义 |
| --- | --- | --- | --- |
| `model_not_found` | 400 | `invalid_request_error` | 请求指定的模型不由该 Router 提供。 |
| `no_route` | 400 | `invalid_request_error` | 没有决策匹配，且没有可用的默认模型。配方入口会回退到已配置的 `providers.defaults.model`；Looper 入口遵循同样的配方规则，名称本身不会选择算法。 |
| `context_length_exceeded` | 400 或 422 | `invalid_request_error` | 请求超出了可服务它的模型的容量：400 来自[请求预算检查](#request-budget-errors)，422 来自模型的 `context_window_size`。 |
| `max_output_tokens_exceeded` | 400 | `invalid_request_error` | 请求的输出超过了配置的模型上限。见[请求预算错误](#request-budget-errors)。 |
| `decision_unresolved` | 503 | `server_error` | 某个决策所需的信号不可用，导致该决策无法评估，且其 `rules.on_unknown` 为 `fail_request`。`x-vsr-applied-unknown-policy` 会给出该决策。 |
| `no_eligible_model` | 503 | `server_error` | 选择策略拒绝了匹配决策的所有候选模型。 |

Router 会以 `WARN` 级别记录每一次此类失败，带上请求的 `x-request-id`、原因码和 Router 自己的原因：请求指定的模型、到达的配方和决策，以及错误本身。Anthropic Messages 客户端会在 Anthropic 的错误信封中收到相同的状态码和消息；该信封没有 code 字段。

## 请求预算错误 {#request-budget-errors}

设置 `candidate_requirements.context: known_limits` 后，Router 会用候选模型配置的上限检查估算的输入加上有效输出额度。如果所有候选模型都只因预算检查失败，Router 返回 HTTP 400：

| 错误码 | 含义 |
| --- | --- |
| `context_length_exceeded` | 准备好的输入与请求的输出放不下。 |
| `max_output_tokens_exceeded` | 请求的输出超过了配置的模型上限。 |

缺少能力、上限未知、选择证据不可用以及混合失败，保持选择错误的行为，即 `no_eligible_model`。预算检查本身不会截断请求；需要时请启用[上下文压缩](../tutorials/plugin/context-compression.md)。

这些计数是估算值。后端仍可能拒绝请求；其有效的 HTTP 状态码和有意义的消息会被保留。vLLM 的整数错误码在 OpenAI 兼容错误中以字符串形式暴露：`code: 400` 的 `BadRequestError` 会变成 `code: "400"` 的 `invalid_request_error`。

在生成开始前被拒绝的流式请求，收到的是同样的非 2xx JSON 错误，而不是成功的 SSE 流。启用回放时，回放会记录失败的状态码和响应体；Router 的预算拒绝使用 `terminal_reason: request_budget_exceeded`。

## 路由回放 {#router-replay}

路由回放记录路由决策和所选请求生命周期数据。它适用于调试、评测和路由学习，但读取记录本身不会改变路由。

除非启用该服务，否则回放默认关闭：

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: memory
```

内存后端适合本地检查。记录必须在进程重启后保留时，使用已配置的持久后端，并按所捕获数据设置合适的保留期。

回放查询发往管理 API：

```bash
curl -sS 'http://localhost:8080/api/v1/observability/replays?limit=20' \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/observability/replays` | 列出并过滤记录 |
| `GET` | `/api/v1/observability/replays/{id}` | 读取单条记录 |
| `GET` | `/api/v1/observability/replays/aggregate` | 聚合路由和成本元数据 |
| `GET` | `/api/v1/observability/replays/trajectory?session_id=...&recipe=...` | 重建指定配方的会话轨迹 |

列表和聚合请求接受 `recipe`、`decision`、`model`、`session_id`、`cache_status` 和 `search` 等过滤器。分页使用 `limit` 和 `offset`；`limit` 上限为 100。`showDetails=true` 会请求大体量字段，仅在需要这些字段时使用。

轨迹查询使用精确配方名。仅当会话记录属于一个配方时，才允许省略 `recipe`；同一会话跨配方时返回 `400`。显式空值 `recipe=` 选择旧的未分配配方记录。响应保留每次请求的路由、延迟和生命周期，包括同一轮次中的多次请求。

记录、轨迹中的路由和消息在具有显式对话身份时包含 `conversation_id`。消息按对话和轮次分组，因此同一会话内的不同对话都可以从第零轮开始。Insights 会显示对话边界和完整 ID。

Dashboard Insights 将这些路由与已记录的信号、投影、候选分数和会话切换原因一并展示。在 `observe` 模式下，候选及保持模型的解释表示保护策略本来会如何处理；所选模型和路由历史仍表示实际派发。保护策略的 `candidate_models` 独立于分数列出合格模型；未记录的分数显示为 `—`，已记录的零分仍显示为零。缺少身份或证据会明确显示。回放使用 `global.services.router_replay` 的采集默认值，并应用已选 decision 的 `router_replay` 插件覆盖；`enabled: false` 可关闭该 decision 的采集。尚未选中 decision 的被拒绝请求使用全局默认值。全局或插件设置 `capture_personal_data: false` 后，检测到 PII 或无法确认 PII 状态时会省略正文、提示词和工具内容，保留路由元数据。

启用 bearer 认证时，回放调用者需要 `replay.read`。提示词、响应、工具及其他敏感细节保持脱敏，除非主体还拥有 `replay.detail`。即使 API 通常返回脱敏视图，也应将回放存储视为可能敏感。

回放生命周期值描述记录器观察到的状态：

- `in_progress`：尚未记录到终止响应帧。
- `completed`：响应正常结束。
- `failed`：路由或上游响应失败。
- `aborted`：流在没有有效终止帧的情况下结束，例如断开或超时。

仅有 HTTP `200` 响应头并不会使流式记录变为 `completed`。

## 应使用哪个端口？ {#which-port-should-i-use}

| 任务 | 表面 |
| --- | --- |
| 发送模型流量 | 默认 standalone frontend，或 `--gateway extproc` 下的 Envoy；标准本地栈为 `8899` |
| 列出公开模型 | 推理监听器上的 `GET /v1/models` |
| 检查健康或就绪 | `8080` 上的管理 API |
| 读取或更改配置 | `8080` 上的管理 API |
| 检查回放记录 | `8080` 上的管理 API |
| 管理 Router 自有文件或向量存储 | `8080` 管理 API 下的 `/api/v1/storage/*` |

不要把管理端口当作公网推理监听器的替代品暴露出去。其端点可能泄露配置和运维数据，或发起会改变状态的请求。
