---
translation:
  source_commit: "0ee41b5f"
  source_file: "docs/api/router.md"
  outdated: false
title: Router API 参考
description: 经 Envoy 暴露的数据面 HTTP 接口与前后端路由行为说明
---

# Router API 参考

Router 是通常通过 Envoy 暴露的数据面 HTTP 接口。

健康检查、配置与发现等控制面端点见 [Router Apiserver API](./apiserver)。

## 入口

| 接口 | 默认端口 | 用途 |
| --- | --- | --- |
| Envoy 公网入口 | `8801` | 面向客户端的路由 HTTP API |
| ExtProc gRPC | `50051` | 内部 Envoy external processing 钩子 |
| Router apiserver | `8080` | 控制与工具类 API，如 `/v1/models`、`/health`、`/config/router` |

## 前端 API

| API 表面 | 公开路径 | 状态 | 说明 |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | 支持 | 主要的路由推理入口 |
| OpenAI Responses API | `POST /v1/responses` | 支持 | 内部转换为 Chat Completions |
| OpenAI Responses API 检索 | `GET /v1/responses/{id}` | 支持 | 需要启用 Response API 服务/存储 |
| OpenAI Responses API 删除 | `DELETE /v1/responses/{id}` | 支持 | 需要启用 Response API 服务/存储 |
| OpenAI Responses API 输入项 | `GET /v1/responses/{id}/input_items` | 支持 | 需要启用 Response API 服务/存储 |
| OpenAI Models API | `GET /v1/models` | 在 apiserver 上支持 | 由 `:8080` 提供；可按需在 Envoy 上再次暴露 |

## 后端模型 API

路由决策之后可指向的上游模型协议。属于面向后端的集成，不一定是公网客户端入口。

| 后端模型 API | 上游路径 | 状态 | 说明 |
| --- | --- | --- | --- |
| OpenAI 兼容 Chat Completions | `/chat/completions` | 支持 | OpenAI 兼容后端的默认族 |
| Anthropic Messages API | `/v1/messages` | 支持 | 转发前将 OpenAI 风格请求转换为 Anthropic 格式 |
| vLLM Omni Chat Completions | `/chat/completions` | 支持 | 用于 omni 与图像生成等后端，例如 `vllm_omni` |

默认使用 OpenAI 兼容 chat-completions 的 provider 族包括 `openai`、`azure-openai`、`bedrock`、`gemini`、`vertex-ai`。

## 前端行为

### OpenAI Chat Completions

- 公开请求路径：`POST /v1/chat/completions`
- 这是路由推理的主入口。
- 可使用显式模型名，或路由自动模型名（如 `MoM` 或 `auto`）。

最小请求示例：

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": "What is the derivative of x^2?"
    }
  ]
}
```

### OpenAI Responses API

- 公开请求路径：
  - `POST /v1/responses`
  - `GET /v1/responses/{id}`
  - `DELETE /v1/responses/{id}`
  - `GET /v1/responses/{id}/input_items`
- Router 在内部将 `POST /v1/responses` 转为 Chat Completions，再将后端响应转回 Responses API 格式。
- 检索与删除路径需要启用 Response API 服务/存储。

最小请求示例：

```json
{
  "model": "auto",
  "input": "Summarize the benefits of retrieval-augmented generation."
}
```

## 后端行为

### Anthropic API

- 当模型配置为 `api_format: anthropic` 时，Router 可路由到 Anthropic 后端。
- Anthropic 支持位于后端模型 API 层。
- 客户端入口仍是 OpenAI 风格的 Chat Completions 或 Responses API，而不是 `POST /v1/messages`。
- Router 将上游请求转为 Anthropic `POST /v1/messages`，再将响应转回 OpenAI 兼容输出。
- 基于 Anthropic 的路由不支持流式。

### vLLM Omni 与多模态/图像生成

- Router 支持通过 omni 模型与图像生成后端进行多模态/图像生成路由。
- `vllm_omni` 是支持的图像生成后端类型之一。
- 当模态决策解析到 omni 模型时：
  - Chat Completions 请求返回原始 omni Chat Completions 响应。
  - Responses API 请求会规范为 Responses API 输出项；若生成了图像，可包含 `image_generation_call` 等项。
- 多模态或图像生成走上述路径，而不是单独的公开协议。

## 配置关联

上游目标与各 provider 行为来自标准 router 配置：

```yaml
providers:
  models:
    - name: claude-sonnet
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 3.0
        completion_per_1m: 15.0
      backend_refs:
        - base_url: https://api.anthropic.com
          provider: anthropic
```

- 上游路由目标配置在 `providers.models[].backend_refs[]`。
- 可选的成本感知策略可使用 `pricing:`。
- Response API 行为在 `global.services.response_api` 下配置。
- 模态与图像生成行为通过路由决策与 `vllm_omni` 等图像生成后端配置。
