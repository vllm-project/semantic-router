---
title: 接入 Agent Harness
description: 将 Agent Harness 接入稳定的模型 API，并明确路由策略、模型限制与会话连续性。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/installation/agent-harness.md"
  outdated: false
---

# 接入 Agent Harness

将 Harness 连接到 `vllm-sr/auto` 等 Router 入口。Harness 管理 Agent 循环、工具和
任务状态；Router 按策略选择模型或执行有界多模型工作流，推理后端执行模型调用。

先[安装 Router](/docs/installation)，或[让 Agent 安装](agent)。

## 选择推理连接 {#choose-the-inference-connection}

配置 Harness 的模型提供方；配置项名称以安装版本为准。

| 设置 | 使用什么 |
| --- | --- |
| 推理地址 | 公开推理监听器或网关。本地快速开始使用 `http://localhost:8899`；管理 API 和控制面板使用不同地址。 |
| 客户端协议 | 根据 Harness 实际发送的请求，选择 OpenAI Chat Completions、OpenAI Responses 或 Anthropic Messages。 |
| 公开模型 ID | 默认策略使用 `vllm-sr/auto`，也可选择 `GET /v1/models` 返回的已发布入口。 |
| 身份验证 | 推理监听器或网关要求的凭据。后端提供方凭据由运维人员单独配置。 |
| 模型限制与能力 | 所选配方能够支持的上下文、输出、工具、视觉和推理设置。 |

客户端可能自行添加 `/v1`。请检查最终路径：

| 客户端 API | 最终请求路径 | 要求 |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions` | 发送 Chat Completions 请求格式。 |
| Responses | `POST /v1/responses` | Router 的 `global.services.response_api` 及其存储必须可用。 |
| Messages | `POST /v1/messages` | 发送 Messages 请求格式和合适的 `anthropic-version` 请求头。 |

客户端与后端可在[兼容性矩阵](protocol-compatibility)范围内使用不同协议。自由格式的
custom tools 无法转到 Messages 后端；Codex CLI 的额外限制也列在该页。凭据使用
Harness 支持的密钥存储或环境绑定。

## 验证公开模型 {#verify-the-public-model}

在本地栈中配置至少一个后端后，列出公开模型：

```bash
curl -sS http://localhost:8899/v1/models
```

使用[虚拟入口](../tutorials/global/entrypoints)应用路由策略。具体提供方模型名会绕过
配方路由。

发送最小请求并显示响应头：

```bash
curl -sS -i http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

其他部署请使用对应的推理地址和身份验证。检查回答及 `x-vsr-selected-recipe`、
`x-vsr-selected-model`；HTTP 成功不代表输出有效或路由正确。详见
[路由回执](../troubleshooting/vsr-headers)和 [CLI 预览与探测](../api/cli)。

## 运行任务前设置预算 {#set-budgets-before-running-a-task}

`GET /v1/models` 提供虚拟名称和路由元数据，不公布它们的上下文窗口或输出上限。
请在 Harness 中配置这些值。若要让每个候选模型都可用，应取配方可选择的所有模型
（包括配置的默认模型）的最小上下文窗口和输出上限，并仅启用它们共有的能力。

对于异构模型池，配方级 `candidate_requirements` 可以在评分前检查输出预算以及
声明的工具、推理和结构化输出能力，但它不会配置 Harness 自身的上下文管理。
示例和具体候选资格行为见[虚拟模型：Agent 客户端的限制](../tutorials/global/entrypoints-and-recipes#limits-for-agent-clients)。

## 保持工具与对话连续性 {#preserve-tool-and-conversation-continuity}

运行一次真实工具循环：模型请求工具 → Harness 执行 → 下次调用携带匹配结果。
流式请求还需检查完整工具参数、终止事件和最终回答。协议兼容不代表模型与 Harness
已通过端到端验证。

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

检查工具后续调用和用户追问选中的模型。工具权限、任务状态和外层循环由 Harness
管理；配方不会配置委派 Agent。

## 编程入口背后的策略 {#program-the-policy-behind-the-entrypoint}

在同一入口背后[连接模型并发布配方](../tutorials/global/models-entrypoints-serving)。
信号、决策和算法定义哪些模型可用，以及如何执行。

[Agent Routing 配方](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)
可作为本地、专家和前沿模型路径的起点。使用前请检查模型绑定、分类器要求和回放保留
策略；仓库中的配置会记录请求和响应数据。采用策略变更前，使用
[配方评估工作流](../benchmarking/agent-evaluation-loop)测试代表性任务。
