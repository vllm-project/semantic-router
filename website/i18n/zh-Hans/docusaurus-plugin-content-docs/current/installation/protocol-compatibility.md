---
title: 协议兼容性矩阵
description: 将面向客户端的推理 API 与受支持的后端模型协议匹配，并理解跨协议功能边界。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/installation/protocol-compatibility.md"
  outdated: false
---

# 协议兼容性矩阵

Semantic Router 在数据平面两侧都支持三种推理线格式。客户端请求被解码为协议中立形式，路由策略选择模型，该模型的 `api_format` 选择后端编解码器。响应再被翻译回客户端的原始格式。

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

公共监听器还提供 `GET /v1/models`。完整的方法和路径清单、Responses 对象操作以及请求示例见 [Router API](../api/router)。

## 后端模型协议

在每个 `providers.models[]` 条目上设置 `api_format`。它描述该模型端点实现的线契约，而不是 provider 品牌。

| `api_format` | 后端请求和响应形状 | 默认上游路径 | 说明 |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | 省略 `api_format` 时的默认值。`backend_refs[].chat_path` 可以覆盖 Chat 路径。 |
| `responses` | OpenAI Responses | `/v1/responses` | 后端本身必须实现 Responses 线契约；启用 Router 的 Responses 服务不会为后端添加该 API。 |
| `anthropic` | Anthropic Messages | `/v1/messages` | 在后端 ref 上配置 provider 身份验证和所需的版本标头。 |

这些字段容易混淆：

- 模型 `api_format` 选择请求、响应、错误和流式编解码器；
- 后端 ref 的 `protocol` 选择 HTTP 或 HTTPS 传输；以及
- 后端 ref 的 `provider` 提供 provider 专用的身份验证和路径默认值。它并不能证明端点实现了某种 API 格式。

对于每种后端格式，`base_url` 命名完整的上游 API 根。其路径会被保留，并且协议操作后缀恰好追加一次；仅当 URL 没有路径时，才使用协议的默认 `/v1` 基路径。`chat_path` 仅适用于 Chat Completions。

对于 HTTPS 后端，生成的 Envoy cluster 会同时验证服务器证书链及其 DNS 主机名。HTTPS 副本池必须保持一个主机名，因为受支持的 Envoy 运行时在 cluster 内共享其 TLS 上下文；对不同的 HTTPS 主机使用单独的模型别名。IP 字面量 HTTPS 目标会被拒绝，而不是悄悄削弱主机名验证。与 `vllm-sr serve` 一起使用的自定义 Envoy 镜像必须在 `/etc/ssl/certs/ca-certificates.crt` 提供系统 CA 包；当该信任存储不可用时，启动校验会失败，而不是悄悄禁用验证。

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
| 停止序列 | Supported | Not supported | Supported |
| 原生响应或会话状态字段 | Not supported | Supported | Not supported |

此表描述编解码器表示，而不是模型能力。例如，OpenAI 兼容服务器可以接受 Chat 请求形状，同时对特定模型拒绝图像或工具。在将它们加入路由池之前，先限定实际端点和模型 revision。

对于使用 Chat Completions 或 Responses 后端的 Anthropic Messages 客户端，
`thinking.type: adaptive` 使用后端模型的默认推理行为，`output_config.effort` 会保留。
`thinking.display: omitted` 会从转换后的响应中移除推理内容，包括流式输出。显式设置
`thinking.type: disabled` 需要已配置的推理家族和有效的后端关闭推理控制。不支持的
控制会返回类型化请求错误。`context_management` 中 `clear_thinking_20251015` 的
`keep: all` 编辑没有实际作用，会在这些后端上省略；会改变历史的编辑则被拒绝。
当所选后端使用 Responses 时，会省略它无法表示的 Anthropic `cache_control` 边界。
提示词和工具结果仍会派发，`x-vsr-protocol-warnings` 会为 `cache_control` 报告
`dropped` 诊断。

从 Anthropic 后端转换推理内容时，如果后端附带推理签名，转换会失败；其 thinking
响应默认携带签名。Chat Completions 或 Responses 客户端的缓冲请求会收到类型化的
`unsupported_capability` 错误；流式请求则会在流中途、响应头和可能的早期增量已经
发送后失败，因为签名随推理增量才到达编码器。同格式流量不受影响，包括 Messages
客户端读取 Anthropic 后端的情况。

Responses 客户端仍可以与 Chat Completions 或 Messages 后端一起使用 `previous_response_id`。Router 检索并物化保留的历史，移除 Router 拥有的对象控制，然后按所选后端格式编码无状态请求。

Responses custom tools 使用 `type: custom` 和扁平的 `format` 对象。它们的
`custom_tool_call` 和 `custom_tool_call_output` 项在 Chat 或 Responses 后端上保留
自由格式输入、工具结果和调用 ID，同时支持缓冲与流式响应。Messages 后端会以
`unsupported_capability` 拒绝它们。Responses 的 `text.verbosity` 映射到 Chat 的
`verbosity` 字段；Messages 没有对应字段，因此 Router 丢弃这个输出详细程度提示，
并在 `x-vsr-protocol-warnings` 中报告 `text.verbosity`。

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
          base_url: https://api.anthropic.com
          provider: anthropic
          api_key_env: ANTHROPIC_API_KEY
          extra_headers:
            anthropic-version: "2023-06-01"
          weight: 100
```

`api_format` 只选择后端编解码器。它并不暗示 Anthropic、OpenAI 或任何其他运行时 Provider。Router 拥有的监听器要求物理模型声明 `backend_refs[].provider`；仅元数据的 `listeners: []` 配置将传输和凭据留给外部网关。本地 `vllm-sr serve` 工作流管理 Envoy 传输，因此它不接受没有后端的物理模型；对该拓扑使用外部网关部署配置。

先用其后端原生路径和最小请求直接测试后端。然后使用 Agent Harness 或其他 API 客户端所需的协议，通过 Router 发送相同的语义请求。成功的健康检查并不能校验请求 schema、流式、工具或错误翻译。

## 校验和失败行为

- 即使客户端和后端格式匹配，公共请求也会被解码为中立契约。未知或不支持的请求字段会失败关闭。
- 跨协议请求保留共享语义。无法表示的目标特定功能会返回类型化协议错误。
- 响应保持客户端协议的 JSON 或 SSE 形状。Provider 传输错误和不完整流会与成功的模型响应分开翻译。
- 对于 Anthropic 兼容提供方，缺失的可空 `stop_sequence` 会被解释为 null，并产生有界的兼容性诊断。这适用于缓冲 Messages、`message_start.message` 和 `message_delta.delta`。显式 null 不产生诊断；停止原因为 `stop_sequence` 时，仍必须提供非空的匹配序列，终止增量仍必须包含 `stop_reason`。
- OpenAI 兼容 Chat 提供方通过 `choices[].stop_reason` 返回匹配到的停止字符串时（例如 vLLM），Anthropic Messages 客户端会收到 `stop_sequence` 停止原因及该序列。Chat 和 Responses 客户端不受影响。
- 适用时，`x-vsr-client-protocol`、`x-vsr-upstream-protocol` 和 `x-vsr-protocol-warnings` 会暴露翻译细节。参见 [VSR 路由标头](../troubleshooting/vsr-headers)。流式响应头发送后才发现的诊断会记录在转换告警指标和结构化调试日志中，不能添加到已经发送的响应头。

仓库在编解码器测试、Envoy ExtProc 边界，以及 18 单元格部署矩阵中成对验证全部三种协议：三种客户端格式 × 三种后端格式 × 缓冲或流式模式。完整的验证和扩展契约见[已实现的编解码器设计](../proposals/multi-protocol-adaptor)。

## Codex CLI {#codex-cli}

Codex CLI 使用 Responses 端点并发送若干兼容性字段，Router 按下表处理：

| 字段 | Router 行为 |
| --- | --- |
| `prompt_cache_key` | 转发到 `openai` 和 `responses` 后端。路由到 `anthropic` 后端时，因 Messages 没有对应字段而省略，并报告 `dropped`。决策中的 `request_params.blocked_params` 可以在派发前移除它。 |
| `include: ["reasoning.encrypted_content"]` | 接受但不转发。Router 不中继提供方加密的推理内容，因此推理项没有 `encrypted_content`。其他 `include` 值仍不支持。 |
| `client_metadata` | 接受但不转发，因为它包含 Codex 遥测，而非模型输入。 |
| `text.verbosity` | 转发到 `responses` 后端，映射为 `openai` Chat 后端的 `verbosity`，在 `anthropic` Messages 后端上报告为 `dropped`。仅接受 `low`、`medium` 和 `high`。 |

每个被接受但未转发的字段都会作为 `dropped` 项出现在 `x-vsr-protocol-warnings` 中。

Codex 也可以请求推理摘要。Router 会将 `reasoning.summary` 转发到 Responses 后端。
Chat Completions 或 Messages 后端不能请求摘要，因此 Router 接受该轮请求，并在
`x-vsr-protocol-warnings` 中报告被丢弃的设置。多 Agent 命名空间工具和托管网页搜索
仍不支持；请在指向 Router 的 Codex `config.toml` 中禁用它们。以下设置已通过
Codex CLI 0.156.1 核验：

```toml
web_search = "disabled"

[features]
multi_agent = false
```
