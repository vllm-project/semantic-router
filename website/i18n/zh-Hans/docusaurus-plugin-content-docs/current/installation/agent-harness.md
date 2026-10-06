---
title: 接入 Agent Harness
description: 将 Agent Harness 接入稳定的模型 API，并明确路由策略、模型限制与会话连续性。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/installation/agent-harness.md"
  outdated: false
---

# 接入 Agent Harness

An open, programmable **decision layer** for models and compute.

将 Agent Harness 发起的模型调用接入一个公开入口，就可以在保持模型名稳定的同时，
持续调整背后的模型和路由策略。

Agent Harness 负责 Agent 循环、工具执行和任务状态。Router 为每次模型调用评估
策略，选择一个模型，或执行已配置的、有明确边界的多模型工作流。推理运行时和服务
平台执行这些调用并管理相应算力。

如果尚未运行 Router，请先完成[快速开始](/docs/installation)。[使用 Agent 安装](agent)
介绍如何让 Agent 安装和运维 Router；本指南介绍如何把实际使用推理 API 的 Harness
接入 Router。

## 选择推理连接 {#choose-the-inference-connection}

按下表配置 Harness 的模型提供方。具体配置项名称以已安装的 Harness 版本为准。

| 设置 | 使用什么 |
| --- | --- |
| 推理地址 | 公开推理监听器或网关。本地快速开始使用 `http://localhost:8899`；管理 API 和控制面板使用不同地址。 |
| 客户端协议 | 根据 Harness 实际发送的请求，选择 OpenAI Chat Completions、OpenAI Responses 或 Anthropic Messages。 |
| 公开模型 ID | 默认策略使用 `vllm-sr/auto`，也可选择 `GET /v1/models` 返回的已发布入口。 |
| 身份验证 | 推理监听器或网关要求的凭据。后端提供方凭据由运维人员单独配置。 |
| 模型限制与能力 | 所选配方能够支持的上下文、输出、工具、视觉和推理设置。 |

有些客户端要求 base URL 以 `/v1` 结尾，有些会自行添加 `/v1`。应检查最终请求路径，
不要直接在不同 Harness 之间复制 URL：

| 客户端 API | 最终请求路径 | 要求 |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions` | 发送 Chat Completions 请求格式。 |
| Responses | `POST /v1/responses` | Router 的 `global.services.response_api` 及其存储必须可用。 |
| Messages | `POST /v1/messages` | 发送 Messages 请求格式和合适的 `anthropic-version` 请求头。 |

客户端与后端可以使用不同协议，但兼容性仍取决于请求中的功能。例如，自由格式的
custom tools 无法转换到 Messages 后端。启用 Harness 功能前，先阅读
[协议兼容性矩阵](protocol-compatibility)，包括其中的 Codex CLI 限制。将凭据放在
Harness 支持的密钥存储或环境绑定中。

## 验证公开模型 {#verify-the-public-model}

在本地栈中配置至少一个后端后，列出公开模型：

```bash
curl -sS http://localhost:8899/v1/models
```

需要 Router 应用路由策略时，请使用虚拟入口。具体提供方模型名会直接选择该模型，
并绕过配方路由。解析规则见[入口](../tutorials/global/entrypoints)。

发送最小请求并显示响应头：

```bash
curl -sS -i http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

这些命令使用本地快速开始的监听器。其他部署应使用自己的推理地址和所需身份验证。
检查最终助手输出以及 `x-vsr-selected-recipe`、`x-vsr-selected-model` 路由回执；
仅凭成功的 HTTP 状态无法确认回答有效或路由符合预期。[VSR 路由标头](../troubleshooting/vsr-headers)
解释这些回执，[CLI 参考](../api/cli)介绍路由预览和端到端探测。

## 运行任务前设置预算 {#set-budgets-before-running-a-task}

`GET /v1/models` 提供虚拟名称和路由元数据，不公布它们的上下文窗口或输出上限。
请在 Harness 中配置这些值。若要让每个候选模型都可用，应取配方可选择的所有模型
（包括配置的默认模型）的最小上下文窗口和输出上限，并仅启用它们共有的能力。

对于异构模型池，配方级 `candidate_requirements` 可以在评分前检查输出预算以及
声明的工具、推理和结构化输出能力，但它不会配置 Harness 自身的上下文管理。
示例和具体候选资格行为见[虚拟模型：Agent 客户端的限制](../tutorials/global/entrypoints-and-recipes#limits-for-agent-clients)。

## 保持工具与对话连续性 {#preserve-tool-and-conversation-continuity}

在 Harness 中验证一次真实的工具循环：模型请求工具，Harness 执行工具，下一次模型
调用携带与之匹配的工具结果。启用流式输出时，还应确认 Harness 收到完整的工具参数、
终止事件和最终回答。协议转换保留受支持的线格式契约；模型能力和 Harness 行为仍需
端到端验证。

配置 Router Learning 保护时，需要提供显式身份。默认 `scope: conversation` 要求
相关调用同时携带两个请求头：

```http
x-session-id: harness-demo-session
x-conversation-id: harness-demo-conversation
```

这些只是示例值。由 Harness 或受信任网关为实际 session 和 conversation 提供各自
稳定的标识符。`scope: session` 只要求配置中的 session 请求头。缺少必需身份时，
请求仍可路由，但保护不会保持模型。Responses 历史（包括 `previous_response_id`）
不能替代这些请求头。身份优先级和隐私边界见[会话标识](../api/session-identification)。

验证保护行为时，请检查工具后续调用和用户追问实际选择的模型。工具执行权限和任务
状态由 Harness 管理；路由配方不会配置委派 Agent，也不负责外层 Agent 循环。

## 编程入口背后的策略 {#program-the-policy-behind-the-entrypoint}

按照[模型、入口与服务](../tutorials/global/models-entrypoints-serving)连接模型并发布
配方。信号描述请求，决策实施路由策略，算法选择或协调合格模型。稳定入口让 Harness
保持模型 ID 不变，同时持续演进背后的策略。

[Agent Routing 配方](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)
可作为本地、专家和前沿模型路径的起点。使用前请检查模型绑定、分类器要求和回放保留
策略；仓库中的配置会记录请求和响应数据。采用策略变更前，使用
[配方评估工作流](../benchmarking/agent-evaluation-loop)测试代表性任务。
