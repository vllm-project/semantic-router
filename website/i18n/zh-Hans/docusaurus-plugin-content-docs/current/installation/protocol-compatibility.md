---
is_mtpe: false
title: 协议兼容性矩阵
description: 将面向客户端的推理 API 与受支持的后端模型协议匹配，并理解跨协议功能边界。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/installation/protocol-compatibility.md"
  outdated: true
---

# 协议兼容性矩阵

Semantic Router 在数据平面两侧都支持三种推理线格式，以及用于图像生成的 Images API。客户端请求被解码为协议中立形式，路由策略选择模型，该模型的 `api_format` 选择后端编解码器。响应再被翻译回客户端的原始格式。

```text
client endpoint -> client codec -> routing -> backend codec -> model endpoint
```

协议兼容性独立于目标配置和部署支持：

- 使用[后端目标兼容性](backend-target-compatibility)了解 URL、权重、标头、发现和生成方保留；以及
- 使用[部署支持](support-matrix)了解项目维护的栈、集成和硬件配置。

客户端连接设置、虚拟模型限制和工具循环验证见[接入 Agent Harness](agent-harness)。

## 面向客户端的协议

| 客户端 API | 推理端点 | 缓冲 | 流式 | 可用性 |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Supported | 在公共推理监听器上可用。 |
| OpenAI Responses | `POST /v1/responses` | Supported | Supported | 需要 `global.services.response_api` 及其存储可用。Router 拥有的对象操作不会转发到模型后端。 |
| Anthropic Messages | `POST /v1/messages` | Supported | Supported | 发送 Anthropic 请求形状和适当的 `anthropic-version` 标头。客户端身份验证仍取决于部署。 |
| OpenAI Images | `POST /v1/images/generations` | Supported | Not supported | 只由 `output` 为图像的决策应答。参见 [Images API](../api/router#images-api)。 |

公共监听器还提供 `GET /v1/models`。完整的方法和路径清单、Responses 对象操作以及请求示例见 [Router API](../api/router)。

原生 **System One** API 是独立的类型化接口，用于直接调用判断模型，不经过本页的 Chat 协议转换矩阵。Router 和 Engine 模式都可以提供它，但 listener 必须通过 `apis.systemone.models` 显式授予模型访问权限。见[组件架构](../overview/component-architecture#keep-public-and-worker-apis-distinct)。

## 后端模型协议

在每个 `providers.models[]` 条目上设置 `api_format`。它描述该模型端点实现的线契约，而不是 provider 品牌。

| `api_format` | 后端请求和响应形状 | 默认上游路径 | 说明 |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | 省略 `api_format` 时的默认值。`backend_refs[].chat_path` 可以覆盖 Chat 路径。 |
| `responses` | OpenAI Responses | `/v1/responses` | 后端本身必须实现 Responses 线契约；启用 Router 的 Responses 服务不会为后端添加该 API。 |
| `anthropic` | Anthropic Messages | `/v1/messages` | 在后端 ref 上配置 provider 身份验证和所需的版本标头。 |
| `images` | OpenAI Images | `/v1/images/generations` | 只生成图像，用于[输出](../tutorials/decision/output)为图像的决策；从不流式输出。 |

这些字段容易混淆：

- 模型 `api_format` 选择请求、响应、错误和流式编解码器；
- 后端 ref 的 `protocol` 选择 HTTP 或 HTTPS 传输；以及
- 后端 ref 的 `provider` 提供 provider 专用的身份验证和路径默认值。它并不能证明端点实现了某种 API 格式。

对于每种后端格式，`endpoint` 中的完整 URL 命名完整的上游 API 根。其路径会被保留，并且协议操作后缀恰好追加一次；仅当 URL 没有路径时，才使用协议的默认 `/v1` 基路径。`chat_path` 仅适用于 Chat Completions。

使用 `--gateway extproc` 时，对于 HTTPS 后端，生成的 Envoy cluster 会同时验证服务器证书链及其 DNS 主机名。HTTPS 副本池必须保持一个主机名和一个 `tls` 块，因为受支持的 Envoy 运行时在 cluster 内共享其 TLS 上下文；对不同的 HTTPS 主机使用单独的模型别名。IP 字面量 HTTPS 目标会被拒绝，而不是悄悄削弱主机名验证。后端的 `tls.ca_file`（未设置时为 `providers.defaults.tls.ca_file`）指定其证书必须链接到的 CA 包，见[信任私有 CA](backend-target-compatibility#trust-a-private-ca)。两者都未设置时，与 `vllm-sr serve` 一起使用的自定义 Envoy 镜像必须在 `/etc/ssl/certs/ca-certificates.crt` 提供系统 CA 包；当该信任存储不可用时，启动校验会失败，而不是悄悄禁用验证。

## 客户端到后端矩阵

每种客户端格式都可以路由到每种后端格式。每个单元格都覆盖缓冲和流式模式。

| 客户端协议 | `openai` 后端 | `responses` 后端 | `anthropic` 后端 |
| --- | --- | --- | --- |
| OpenAI Chat Completions | Supported | 通过编解码器翻译支持 | 通过编解码器翻译支持 |
| OpenAI Responses | 通过编解码器翻译支持 | Supported | 通过编解码器翻译支持 |
| Anthropic Messages | 通过编解码器翻译支持 | 通过编解码器翻译支持 | Supported |

“Supported” 表示 Router 拥有请求、响应、传输错误和流式翻译路径。它并不表示一种协议的每个字段都能被另一种协议表示，也不表示端点背后的每个模型都支持所请求的能力。

## 功能可移植性

Router 在编码后端请求之前检查所需语义。所选后端格式无法表示的功能会显式失败，而不是被悄悄丢弃。

| 语义功能 | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| 文本、图像输入和文件输入 | Supported | Supported | Supported |
| 工具、并行工具调用和严格工具 schema | Supported | Supported | Supported |
| 自由格式的 custom tools 及其调用 | Supported | Supported | Not supported |
| 文本详细程度（`low`、`medium`、`high`） | Supported | Supported | 不转发；标记为 `dropped` |
| 严格 JSON Schema 输出 | Supported | Supported | Supported |
| 缓冲和流式响应 | Supported | Supported | Supported |
| 推理内容和 effort | Supported，但不支持带签名的 thinking 块 | Supported，但不支持带签名的 thinking 块 | Supported |
| 推理摘要请求（`reasoning.summary`） | 不转发；标记为 `dropped` | Supported；若提供方使用 `chat_template_kwargs` 控制推理，则标记为 `dropped` | 不转发；标记为 `dropped` |
| 没有 schema 的 JSON object 模式 | Supported | Supported | Not supported |
| 音频输入 | Supported | Not supported | Not supported |
| 托管图像生成生命周期 | Not supported | Supported | Not supported |
| 多个响应候选 | Supported | Not supported | Not supported |
| 提示词缓存指令 | Supported | Not supported | Supported |
| 提示词缓存键（`prompt_cache_key`） | Supported | Supported | 不转发；标记为 `dropped` |
| 推理 token 预算 | Supported 扩展 | Not supported | Supported |
| Seed 以及频率或存在惩罚 | Supported | Not supported | Not supported |
| `top_k` 采样 | Supported 扩展 | Not supported | 非负值 Supported |
| `min_p`、重复惩罚和缓存盐 | Supported 扩展 | Not supported | Not supported |
| Token logit 偏置（`logit_bias`） | Supported | Not supported | Not supported |
| 停止序列 | Supported | Not supported | Supported |
| 原生响应或会话状态字段 | Not supported | Supported | Not supported |

此表描述编解码器表示，而不是模型能力。例如，OpenAI 兼容服务器可以接受 Chat 请求形状，同时对特定模型拒绝图像或工具。在将它们加入路由池之前，先限定实际端点和模型 revision。

Messages 没有图像 detail 级别，也没有文档文件名。发往 `anthropic` 后端时，`auto` 以外的图像 `detail` 和文件 `filename` 会被省略，`x-vsr-warnings` 会将其分别报告为 `dropped`。

Chat 的 tool 消息只能携带文本。发往 Chat 后端时，工具结果中的图像（例如 Claude Code 用 `Read` 读取的图片、Codex 的 `view_image` 返回的图片）会放进紧跟在回答本轮的 tool 消息之后的一条 user 消息里，开头是 "The tool results above returned these images."；结果只有图像的 tool 消息会注明图像在下一条消息中。`x-vsr-warnings` 以字段 `messages[].content.tool_result.image` 将这一移动报告为 `approximated`。工具结果中的其他媒体（例如文档）仍以 `tool_result_media` 拒绝。

对于使用 Chat Completions 或 Responses 后端的 Anthropic Messages 客户端，
`thinking.type: adaptive` 使用后端模型的默认推理行为，`output_config.effort` 会保留。
`thinking.display: omitted` 会从转换后的响应中移除推理内容，包括流式输出。显式设置
`thinking.type: disabled` 需要已配置的推理家族和有效的后端关闭推理控制。不支持的
控制会返回类型化请求错误。`context_management` 中 `clear_thinking_20251015` 的
`keep: all` 编辑没有实际作用，会在这些后端上省略；会改变历史的编辑则被拒绝。
当所选后端使用 Responses 时，会省略它无法表示的 Anthropic `cache_control` 边界。
提示词和工具结果仍会派发，`x-vsr-warnings` 会为 `cache_control` 报告
`dropped` 诊断。

在所有后端上，`max_tokens` 都同时覆盖思考和回答。没有 `thinking` 且决策开启推理的
请求，可能把较小的 `max_tokens` 用于思考而返回不了文本。设置
`providers.defaults.reasoning_min_output_tokens` 后，`max_tokens` 低于该值的这类请求
会以模型推理关闭的方式发往后端；参见[模型推理](model-reasoning#turn-reasoning-off-for-small-output-limits)。

从 Anthropic 后端转换推理内容时，如果后端附带推理签名，转换会失败；其 thinking
响应默认携带签名。Chat Completions 或 Responses 客户端的缓冲请求会收到类型化的
`unsupported_capability` 错误；流式请求则会在流中途、响应头和可能的早期增量已经
发送后失败，因为签名随推理增量才到达编码器。同格式流量不受影响，包括 Messages
客户端读取 Anthropic 后端的情况。在 `global.router.allow_lossy_translation` 中列出
`reasoning.signature` 后，这些响应可以通过：Router 丢弃签名、保留推理文本，并将该
损失报告为 `approximated`。

Messages 客户端在之后的每一轮都会把这些 thinking 连同签名一起发回。当由 `anthropic`
后端回答过的会话转到 Chat Completions 或 Responses 后端时，请求会在路由前以
`unsupported_capability` 失败，错误消息指出第一个带签名的块，例如
`messages[1].content[0].signature: the selected model cannot accept signed thinking`。
同一设置也允许这类请求通过：这些后端收到每个 thinking 块的文本，但不带签名；
`x-vsr-warnings` 为 `reasoning.signature` 报告 `approximated` 诊断；`anthropic`
后端（无论是决策的选择还是回退候选）仍会收到签名。签名为空（`"signature": ""`）的
thinking 块不携带签名，无需该设置即可发往任何后端。

Chat Completions 客户端可以用 `store: true` 要求提供方保存补全，而 Messages 后端
不保存补全。`store: false` 不要求这类后端做任何额外的事，因此请求去掉该字段后照常
发送；`store: true` 会以 `unsupported_capability`（`request_storage`）失败，因为后端
无法保存补全。在 `global.router.allow_lossy_translation` 中列出 `store` 后，Router 会
去掉 `store` 发送这类请求，并在 `x-vsr-warnings` 中为 `store` 报告 `dropped` 诊断。
Chat Completions 和 Responses 后端仍按客户端发送的原样收到 `store`。

Responses 客户端仍可以与 Chat Completions 或 Messages 后端一起使用 `previous_response_id`。Router 检索并物化保留的历史，移除 Router 拥有的对象控制，然后按所选后端格式编码无状态请求。存储输入中的 developer 和 system 项按轮次顺序作为指令重放：位于当前请求的顶层 `instructions` 之后，并位于当前请求自身输入中的 developer 和 system 项之前。之前请求的顶层 `instructions` 不会沿用。

Responses custom tools 使用 `type: custom` 和扁平的 `format` 对象。它们的
`custom_tool_call` 和 `custom_tool_call_output` 项在 Chat 或 Responses 后端上保留
自由格式输入、工具结果和调用 ID，同时支持缓冲与流式响应。Messages 后端会以
`unsupported_capability` 拒绝它们。Responses 的 `text.verbosity` 映射到 Chat 的
`verbosity` 字段；Messages 没有对应字段，因此 Router 丢弃这个输出详细程度提示，
并在 `x-vsr-warnings` 中报告 `text.verbosity`。

### Router 拒绝的 Chat 请求字段 {#chat-request-fields-the-router-refuses}

对于无法传给所有后端的 Chat Completions 字段，Router 会失败关闭：请求得到 `400` 和下表中的错误码，不会到达任何后端。
如果某个取值请求的内容，后端在没有该字段时也会照做，Router 就接受它：后端收到的请求不带该字段，`x-vsr-warnings`
会为它报告 `dropped` 诊断。取值为 `null` 视同没有该字段；`logprobs: false` 请求的是默认行为，两者都不会被报告。

| 字段 | 错误码 | 接受的取值 |
| --- | --- | --- |
| `audio` | `unsupported_audio` | 无 |
| `function_call` | `unsupported_function_call` | 无 |
| `functions` | `unsupported_functions` | 无 |
| `logprobs` | `unsupported_logprobs` | `false` |
| `modalities` | `unsupported_modalities` | 恰好为 `["text"]` |
| `moderation` | `unsupported_moderation` | 无 |
| `prediction` | `unsupported_prediction` | 无 |
| `prompt_cache_options` | `unsupported_prompt_cache_options` | 无 |
| `prompt_cache_retention` | `unsupported_prompt_cache_retention` | 无 |
| `safety_identifier` | `unsupported_safety_identifier` | 任意字符串：供提供方滥用监控使用的终端用户 ID，后端不会使用它 |
| `service_tier` | `unsupported_service_tier` | `"auto"` 和 `"default"` |
| `top_logprobs` | `unsupported_top_logprobs` | 无 |
| `web_search_options` | `unsupported_web_search_options` | 无 |
| `messages[].audio`，即回放历史中的输出音频 | `unsupported_output_audio` | 无 |
| `messages[].function_call`，即历史中的旧版函数调用 | `unsupported_legacy_function_call` | 无 |
| `messages[].tool_calls[].function.TokenizedArguments` | `unsupported_messages_tool_calls_function_tokenized_arguments` | 无 |
| 消息角色不能容纳的内容部分，例如 system 消息中的图像 | `unsupported_content` | 无 |
| `type` 不是 `function` 或 `custom` 的工具 | `unsupported_tool` | 无 |
| 类型为 `allowed_tools` 的 `tool_choice` | `unsupported_tool_choice` | 无 |
| `text`、`json_object`、`json_schema` 以外的 `response_format` | `unsupported_output_format` | 无 |

Responses 请求的 `service_tier` 和 `safety_identifier` 遵循同样的规则。它的其他 Router 无法承载的顶层字段，即
`background`、`context_management`、`max_tool_calls`、`moderation`、`prompt`、`prompt_cache_options`、
`prompt_cache_retention` 和 `top_logprobs`，会以 `unsupported_<field>` 失败。

### 后端拒绝的采样控制 {#sampling-controls-a-backend-rejects}

有些后端会拒绝其线路格式定义的采样控制。例如，运行推测解码（speculative
decoding）的 vLLM 后端会以 HTTP 400 拒绝任何携带 `min_p` 或 `logit_bias` 的请求。
在模型的 `unsupported_capabilities` 中列出这类控制：

```yaml
providers:
  models:
    - name: qwen3-mtp
      api_format: chat_completions
      unsupported_capabilities: [sampling_min_p, sampling_logit_bias]
      backend_refs:
        - name: mtp-pool
          endpoint: mtp-server:8000
          provider: vllm
```

- **仅对该模型省略。** 发给该模型的请求，无论它是决策的选择还是回退候选，到达时都不带所列字段。
- **会报告。** `x-vsr-warnings` 会为每个被省略的字段（`min_p` 或 `logit_bias`）报告 `dropped` 诊断。
- **其他模型不受影响。** 同一决策的其他模型仍会收到这些字段。

该列表接受 `sampling_min_p`、`sampling_logit_bias` 和 `developer_role`（[见下文](#developer-messages-a-backend-drops)）；其他名称会在加载时失败。网关会把模型的请求分摊到它的各个
`backend_refs`，因此只要其中任何一个拒绝某个控制，就应列出它。若要为决策路由到的每个模型移除某个控制，请改为在该决策的
[`request_params`](../tutorials/plugin/request-params) `blocked_params` 中列出它。

### 聊天模板拒绝的 system 消息 {#system-messages-a-chat-template-refuses}

一个请求可以携带多条 system 和 developer 消息：Chat Completions 客户端可能为每条指令单独发送一条 system 消息，
Responses 请求有 `instructions` 和 developer 输入项，编码代理还会在对话中间插入 system 消息。许多开源模型的聊天模板
只接受一条位于开头的 system 消息：Qwen3.5 及之后的模板会以 "System message must be at the beginning." 拒绝请求，
gpt-oss 则丢弃这段文本。同一个请求因此会随决策选中的模型而成功或失败。

对使用这类模板的后端，在 `global.router.allow_lossy_translation` 中加入 `messages[].role`（参见下文的“校验和失败行为”）：

```yaml
global:
  router:
    allow_lossy_translation: ["messages[].role"]
```

- **开头只有一条 system 消息。** Chat 后端把对话开头的 system 和 developer 消息合并为一条 system 消息收到，各段文本按原顺序排列，之间以空行分隔：Chat 请求开头的 system 和 developer 消息，或 Responses 请求的 `instructions` 及开头的 developer 和 system 输入项。`cache_control` 边界保留在设置它的文本末尾。只有一条开头 system 消息的请求原样到达后端。
- **后续消息作为用户文本。** 对话中间的 system 或 developer 消息在原位置以用户文本到达 Chat 后端：并入紧邻其前的用户轮次；位于 assistant 轮次或工具结果之后时，则成为单独的用户轮次。Messages 请求 `messages` 中的 system 消息以同样方式到达 Responses 后端。
- **会报告。** `x-vsr-warnings` 会为 `messages[].role` 报告一条 `approximated` 诊断。
- **其他后端不受影响。** Messages 后端按原样收到这些消息；Responses 后端收到的 Chat 或 Responses 客户端的 system 和 developer 消息保持各自的角色。

不启用该选项时，每条 system 和 developer 消息都以其原有角色到达后端。

### 后端丢弃的 developer 消息 {#developer-messages-a-backend-drops}

OpenAI 把 `developer` 消息和 `system` 消息一样当作指令。Chat Completions 客户端用 `developer` 角色发送指令，
Responses 请求的 `instructions` 及其 `developer` 输入项到达 Chat 后端时也是 developer 消息。而 vLLM 遵循的
DeepSeek V4 编码把 developer 消息当作独立的一轮：模型开启思考且请求不带工具时，会丢弃最后一条用户消息之前的所有
developer 消息，模型因此看不到这些指令。请在这类模型的 `unsupported_capabilities` 中列出 `developer_role`：

```yaml
providers:
  models:
    - name: deepseek-v4-flash
      api_format: chat_completions
      unsupported_capabilities: [developer_role]
      backend_refs:
        - name: deepseek-pool
          endpoint: deepseek-server:8000
          provider: vllm
```

- **对该模型以 system 发送。** 发给该模型的请求，无论它是决策的选择还是回退候选，其中所有 developer 指令和消息到达时都是 `system` 角色。
- **会报告。** `x-vsr-warnings` 会为 `messages[].role` 报告 `approximated` 诊断。
- **其他模型不受影响。** 同一决策的其他模型仍会收到 `developer` 角色。

列出 `messages[].role` 后，Chat 后端会在那一条 system 消息中收到开头的 developer 消息（[见上文](#system-messages-a-chat-template-refuses)），请求只报告这一条诊断。

## 配置后端格式

客户端可以使用任何受支持的面向客户端端点；`api_format` 控制所选后端接收的内容：

```yaml
providers:
  models:
    - name: hosted/claude
      provider_model_id: claude-model-id
      api_format: anthropic
      backend_refs:
        - name: anthropic-primary
          endpoint: https://api.anthropic.com
          provider: anthropic
          auth: {api_key_env: ANTHROPIC_API_KEY}
          headers:
            anthropic-version: "2023-06-01"
          weight: 100
```

`api_format` 只选择后端编解码器。它并不暗示 Anthropic、OpenAI 或任何其他运行时 Provider。Router 拥有的监听器要求物理模型声明 `backend_refs[].provider`；仅元数据的 `listeners: []` 配置将传输和凭据留给外部网关。本地 `vllm-sr serve` 工作流通过 standalone 前端或生成的 Envoy 配置管理后端传输，因此不接受没有后端的物理模型；该拓扑应使用外部网关部署配置。

先用其后端原生路径和最小请求直接测试后端。然后使用 Agent Harness 或其他 API 客户端所需的协议，通过 Router 发送相同的语义请求。成功的健康检查并不能校验请求 schema、流式、工具或错误翻译。

## 校验和失败行为

- 即使客户端和后端格式匹配，公共请求也会被解码为中立契约。未知或不支持的请求字段会以 `400` `invalid_request_error`
  失败关闭，并像 OpenAI 那样给出字段：

  | 请求 | 错误码 | 消息 |
  | --- | --- | --- |
  | API 未定义的成员 | `unknown_parameter`，`param` 为该成员的路径 | `Unrecognized request argument supplied: new_field`；位于消息内时为 `messages[].new_field` |
  | 合法 JSON，但值的类型错误 | `invalid_type`，`param` 为该成员的路径 | `Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.` |
  | 不是 JSON 的请求体 | `invalid_json` | `request JSON is invalid` |
  | Router 认识但无法承载的成员 | `unsupported_<field>` | 给出该字段 |

  Anthropic Messages 客户端在 Anthropic 的 `invalid_request_error` 中收到相同的消息，该信封没有 code 字段。
- 后端的 `4xx` 错误以客户端的错误格式到达客户端，使用 Router 的错误码和客户端发送的字段名；见[请求预算错误](../api/router#request-budget-errors)。
- Chat 请求接受但不转发客户端库在回放 assistant 消息时添加的字段：LiteLLM 的 `provider_specific_fields`（存放其消息类型没有的回复字段），以及 openai SDK 自己的解析结果 `parsed` 和 `tool_calls[].function.parsed_arguments`。取值不为 null 的字段报告为 `dropped`。与其他 Responses 输入项一样，reasoning 项可以带有 `status`，openai SDK 的流式辅助方法会设置它。
- 函数工具的 `parameters` 为 `null` 时（OpenAI 的类型化客户端会为不带参数的工具这样发送），Router 将其视为没有 `parameters` 的工具，Responses 的 `tools[]` 和 Chat 的 `tools[].function` 均是如此：每种后端收到的都是 schema `{"type":"object"}`，而不是 `null`。
- 跨协议请求保留共享语义。无法表示的目标特定功能会返回类型化协议错误。对于运维人员可以接受的损失（例如 `tool_result.is_error`），`lossy_translation` 错误会指出接受该损失的 `global.router.allow_lossy_translation` 设置，请求中带签名 thinking 的 `unsupported_capability` 错误也会指出；其他 `unsupported_capability` 错误不会指出。
- 响应保持客户端协议的 JSON 或 SSE 形状。Provider 传输错误和不完整流会与成功的模型响应分开翻译。
- 对于 Anthropic 兼容提供方，缺失的可空 `stop_sequence` 会被解释为 null，并产生有界的兼容性诊断。这适用于缓冲 Messages、`message_start.message` 和 `message_delta.delta`。显式 null 不产生诊断；停止原因为 `stop_sequence` 时，仍必须提供非空的匹配序列，终止增量仍必须包含 `stop_reason`。
- OpenAI 兼容 Chat 提供方通过 `choices[].stop_reason` 返回匹配到的停止字符串时（例如 vLLM），Anthropic Messages 客户端会收到 `stop_sequence` 停止原因及该序列。Chat 和 Responses 客户端不受影响。
- vLLM 的服务元数据无论取值如何都不会导致回复失败：回复或流式块顶层的 `prompt_logprobs`、`prompt_token_ids`、`prompt_text`、`prompt_routed_experts`、`kv_transfer_params`、`ec_transfer_params` 和 `metrics`，以及 choices 中的 `token_ids`、`routed_experts` 和 `weight_versions`。任何客户端都不会收到这些字段，也不会收到服务器的 `system_fingerprint`（其中含有 vLLM 的版本、提交和并行布局）；Router 没有可以替代它的稳定指纹，提供服务的配置由 `x-vsr-config-version` 标明。Chat 后端的 Chat 客户端收到的其余回复内容与后端发送的一致，缓冲与流式均如此。元数据取值不为 null 或空时报告为 `dropped`。在 vLLM-Omni 的流中，`modality` 不是 `text` 的块会以 `unsupported_upstream_audio`（或 `unsupported_upstream_modality`）失败，因为其 `delta.content` 承载的是编码后的媒体，而不是回答文本。
- `tool_choice` 为 `none` 的请求到达 Chat Completions 后端时不带 `tools`、`tool_choice` 和 `parallel_tool_calls`，`x-vsr-warnings` 将 `tools` 报告为 `dropped`。vLLM 等模型服务器无论 `tool_choice` 取值如何都会把声明的工具渲染进提示词，模型因此可能开始一个随后被服务器丢弃的调用，只留下调用前的那句话。Messages 后端会收到这些工具，因为历史中含有 `tool_use` 块时它们必须存在；Responses 后端会收到这些工具以及 `tool_choice: "none"`，由 Responses API 负责执行这一选择。
- Chat 提供方以 `finish_reason: "stop"` 结束强制指定的工具调用。对于以这种方式结束且包含 `tool_use` 块的轮次，Anthropic Messages 客户端会收到 `stop_reason: "tool_use"`。因达到输出上限、命中停止序列或拒绝而结束的轮次保留原停止原因。
- 当输出上限在工具调用中途截断一个非流式回复时，客户端收到带有上限停止原因的回复：Chat 为 `finish_reason: "length"`，Responses 为原因为 `max_output_tokens` 的 incomplete 响应，Anthropic Messages 为 `stop_reason: "max_tokens"`。Chat 和 Responses 客户端收到的调用参数与后端发送的一致。由于 `tool_use.input` 必须是对象，Messages 客户端收到的是模型在截断前已完成的成员，Router 会为 `content.tool_use.input` 报告 `approximated` 诊断。只有回复中的最后一个调用可以被截断。不是 JSON 对象开头的参数，或者被截断的调用之后还有其他输出，仍会使回复失败。
- Chat 流的一个 chunk 可以包含多个 `index` 相同的 `tool_calls` 条目，vLLM 的工具解析器在最后一个 chunk 中补发调用的剩余部分时就是这样。这些条目按顺序续写同一个调用，与 OpenAI SDK 的拼接方式一致。`index` 相同但指向不同调用的条目会以 `duplicate_stream_tool_index` 失败。
- Anthropic 的 `model_context_window_exceeded` 停止以 `finish_reason: "length"` 到达 Chat 客户端，并以原因为 `max_output_tokens` 的 incomplete 响应到达 Responses 客户端，与 vLLM 报告输出达到 `max_model_len` 的方式一致。Router 会为 `response.stop_reason` 报告 `approximated` 诊断。`pause_turn` 停止对这些客户端仍会失败。
- Responses 的同一个 reasoning 项可以在 `response.output_item.done` 与 `response.completed` 或 `response.incomplete` 的对应 output 项中携带不同的非空 `encrypted_content` 字符串。Router 接受这一差异，其他字段、output 数量和顺序仍需一致。非空字符串变为缺失、null 或空字符串时，仍以 `stream_terminal_output_mismatch` 失败。协议转换仍将该字段报告为 `dropped`。
- 适用时，`x-vsr-warnings` 以及 `x-vsr-trace` 的 `client_protocol` 和 `upstream_protocol` 成员会暴露翻译细节。参见 [VSR 路由标头](../api/headers)。流式响应头发送后才发现的诊断会记录在转换告警指标和结构化调试日志中，不能添加到已经发送的响应头。

仓库在编解码器测试、Envoy ExtProc 边界，以及 18 单元格部署矩阵中成对验证全部三种协议：三种客户端格式 × 三种后端格式 × 缓冲或流式模式。完整的验证和扩展契约见[已实现的编解码器设计](../proposals/multi-protocol-adaptor)。

## Codex CLI {#codex-cli}

Codex CLI 使用 Responses 端点并发送若干兼容性字段，Router 按下表处理：

| 字段 | Router 行为 |
| --- | --- |
| `prompt_cache_key` | 转发到 `openai` 和 `responses` 后端。路由到 `anthropic` 后端时，因 Messages 没有对应字段而省略，并报告 `dropped`。决策中的 `request_params.blocked_params` 可以在派发前移除它。 |
| `include: ["reasoning.encrypted_content"]` | 接受但不转发。Router 不中继提供方加密的推理内容，因此推理项没有 `encrypted_content`。其他 `include` 值仍不支持。 |
| `client_metadata` | 接受但不转发，因为它包含 Codex 遥测，而非模型输入。 |
| `text.verbosity` | 转发到 `responses` 后端，映射为 `openai` Chat 后端的 `verbosity`，在 `anthropic` Messages 后端上报告为 `dropped`。仅接受 `low`、`medium` 和 `high`。 |

每个被接受但未转发的字段都会作为 `dropped` 项出现在 `x-vsr-warnings` 中。

在 Responses 输出中，`usage.input_tokens_details.cached_tokens` 始终是 Responses schema 要求的整数：后端未报告缓存读取计数时（例如未启用 `--enable-prompt-tokens-details` 的 vLLM Chat 服务）为 `0`，这也是 vLLM 自身 Responses 服务的输出。后端未报告缓存写入计数时省略 `cache_write_tokens`。已报告的零和正数计数会保留。Codex CLI 遇到 `null` 计数或缺少 `cached_tokens` 的详情对象时会使该轮失败。

Codex 也可以请求推理摘要。Router 会将 `reasoning.summary` 转发到 Responses 后端。
Chat Completions 或 Messages 后端不能请求摘要，因此 Router 接受该轮请求，并在
`x-vsr-warnings` 中报告被丢弃的设置。多 Agent 命名空间工具和托管网页搜索
仍不支持；请在指向 Router 的 Codex `config.toml` 中禁用它们。以下设置已通过
Codex CLI 0.156.1 核验：

```toml
web_search = "disabled"

[features]
multi_agent = false
```
