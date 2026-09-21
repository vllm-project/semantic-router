---
title: 协议兼容性矩阵
description: 将面向客户端的推理 API 与受支持的后端模型协议匹配，并理解跨协议功能边界。
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/installation/protocol-compatibility.md"
  outdated: true
---

# 协议兼容性矩阵

Semantic Router 在数据平面两侧都支持三种推理线格式。客户端请求被解码为协议中立形式，路由策略选择模型，该模型的 `api_format` 选择后端编解码器。响应再被翻译回客户端的原始格式。

```text
client endpoint -> client codec -> routing -> backend codec -> model endpoint
```

协议兼容性独立于目标配置和部署支持：

- 使用[后端目标兼容性](backend-target-compatibility)了解 URL、权重、标头、发现和生成方保留；以及
- 使用[部署支持](support-matrix)了解项目维护的栈、集成和硬件配置。

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
| 严格 JSON Schema 输出 | Supported | Supported | Supported |
| 缓冲和流式响应 | Supported | Supported | Supported |
| 推理内容和 effort | Supported | Supported | Supported |
| 没有 schema 的 JSON object 模式 | Supported | Supported | Not supported |
| 音频输入 | Supported | Not supported | Not supported |
| 托管图像生成生命周期 | Not supported | Supported | Not supported |
| 多个响应候选 | Supported | Not supported | Not supported |
| 提示词缓存指令 | Supported | Not supported | Supported |
| 推理 token 预算 | Supported 扩展 | Not supported | Supported |
| Seed 以及频率或存在惩罚 | Supported | Not supported | Not supported |
| `top_k` 采样 | Not supported | Not supported | Supported |
| 停止序列 | Supported | Not supported | Supported |
| 原生响应或会话状态字段 | Not supported | Supported | Not supported |

此表描述编解码器表示，而不是模型能力。例如，OpenAI 兼容服务器可以接受 Chat 请求形状，同时对特定模型拒绝图像或工具。在将它们加入路由池之前，先限定实际端点和模型 revision。

Responses 客户端仍可以与 Chat Completions 或 Messages 后端一起使用 `previous_response_id`。Router 检索并物化保留的历史，移除 Router 拥有的对象控制，然后按所选后端格式编码无状态请求。

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

先用其后端原生路径和最小请求直接测试后端。然后使用应用所需的客户端 API，通过 Router 发送相同的语义请求。成功的健康检查并不能校验请求 schema、流式、工具或错误翻译。

## 校验和失败行为

- 即使客户端和后端格式匹配，公共请求也会被解码为中立契约。未知或不支持的请求字段会失败关闭。
- 跨协议请求保留共享语义。无法表示的目标特定功能会返回类型化协议错误。
- 响应保持客户端协议的 JSON 或 SSE 形状。Provider 传输错误和不完整流会与成功的模型响应分开翻译。
- 适用时，`x-vsr-client-protocol`、`x-vsr-upstream-protocol` 和 `x-vsr-protocol-warnings` 会暴露翻译细节。参见 [VSR 路由标头](../troubleshooting/vsr-headers)。

仓库在编解码器测试、Envoy ExtProc 边界，以及 18 单元格部署矩阵中成对验证全部三种协议：三种客户端格式 × 三种后端格式 × 缓冲或流式模式。完整的验证和扩展契约见[已实现的编解码器设计](../proposals/multi-protocol-adaptor)。
