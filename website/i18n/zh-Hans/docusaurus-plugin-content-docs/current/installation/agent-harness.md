---
is_mtpe: false
title: 接入 Agent Harness
description: 将 Agent Harness 接入稳定的模型 API，并明确路由策略、模型限制与会话连续性。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/installation/agent-harness.md"
  outdated: false
---

# 接入 Agent Harness {#connect-an-agent-harness}

将 harness 指向 `vllm-sr/auto` 等 Router 入口。Harness 负责 agent 循环、工具和任务状态；Router 按你的策略选择模型或运行有界的多模型工作流；推理后端执行模型调用。

先[安装 Router](installation.md)，或[让 Agent 安装它](agent)。

## 选择推理连接 {#choose-the-inference-connection}

配置 harness 的模型提供方。配置项名称因 harness 版本而异。

| 设置 | 使用什么 |
| --- | --- |
| 推理地址 | 公开的推理 listener 或网关。本地快速开始使用 `http://localhost:8899`；管理 API 和 Dashboard 使用不同的地址。 |
| 客户端协议 | 与 harness 实际发送的请求一致：OpenAI Chat Completions、OpenAI Responses 或 Anthropic Messages。 |
| 公开模型 ID | 默认策略使用 `vllm-sr/auto`，或 `GET /v1/models` 返回的已发布入口。 |
| 身份验证 | 推理 listener 或网关要求的凭据。后端提供方的凭据由运维人员另行配置。 |
| 模型限制与能力 | 所选配方能够支持的上下文、输出、工具、视觉和推理设置。 |

客户端可能自行添加 `/v1`。请检查最终路径：

| 客户端 API | 最终请求路径 | 要求 |
| --- | --- | --- |
| Chat Completions | `POST /v1/chat/completions` | 发送 Chat Completions 请求格式。 |
| Responses | `POST /v1/responses` | Router 的 `global.services.response_api` 及其存储必须可用。 |
| Messages | `POST /v1/messages` | 发送 Messages 请求格式和合适的 `anthropic-version` 请求头。 |

客户端与后端可以在[兼容性矩阵](protocol-compatibility)范围内使用不同协议。自由格式的 custom tools 无法到达 Messages 后端；Codex CLI 的额外限制也列在该页。请把凭据放在 harness 支持的密钥存储或环境变量绑定中。

## 验证公开模型 {#verify-the-public-model}

至少配置一个后端，然后列出本地服务的公开模型：

```bash
curl -sS http://localhost:8899/v1/models
```

使用[虚拟入口](../tutorials/global/entrypoints)来应用路由策略。具体的提供方模型名会绕过配方路由。

发送一个最小请求，并带上响应头：

```bash
curl -sS -i http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

其他部署请使用它们的推理地址和身份验证。请同时检查回答以及 `x-vsr-selected-recipe` 和 `x-vsr-selected-model`；仅有 HTTP 成功并不能证明输出或路由正确。见[路由回执](../api/headers)和 [CLI 预览与探针](../api/cli)。

### Harness 自行选择的模型名 {#model-names-the-harness-picks-itself}

Harness 可能请求你没有配置的模型。例如，Claude Code 启动 Explore 子代理时，会请求它自己的 Haiku 模型 ID。Router 对未发布的模型回答“the requested model is not available”，于是该子代理失败。请把 harness 的每一个模型设置都指向某个入口（对 Claude Code 而言是 `ANTHROPIC_MODEL` 和 `ANTHROPIC_DEFAULT_*_MODEL` 变量），或者把它发送的 ID 列入入口的 `model_names`：

```yaml
entrypoints:
  - model_names: [vllm-sr/auto, claude-haiku-5-5]
    recipe: default
```

## 运行任务前设置预算 {#set-budgets-before-running-a-task}

`GET /v1/models` 用于发现虚拟名称和路由元数据，但不公布它们的上下文窗口或输出上限。请在 harness 中配置这些值。为了让每个候选都可用，取配方可选模型（包括配置的默认模型）中最小的上下文窗口和输出上限，并只启用它们共有的能力。

对于异构的模型池，配方级的 `candidate_requirements` 可以在打分前检查输出预算，以及声明的工具、推理和结构化输出支持。它不会配置 harness 自己的上下文管理。示例和具体的资格规则见[虚拟模型：Agent 客户端的限制](../tutorials/global/entrypoints-and-recipes#limits-for-agent-clients)。

listener 的 `timeout`（默认 300 秒）限制整个调用，包括流式响应，到达的输出不会延长它。达到上限的调用会被截断：流被重置，回答的其余部分永远不会到达。

:::warning 长时间推理的调用会触及 listener 超时
对于 agent 模型，请把模型的 `total_timeout` 设为 `0s`，并保留 `idle_timeout` 来限制停滞的流；见[模型配置](model-configuration#tune-timeouts-retries-and-endpoint-health)。在 Claude Code 2.1.304 以 `xhigh` 推理强度在 Qwen3.8-27B-FP8（vLLM 0.32，一块 MI300X）上运行的 Terminal-Bench 2.1 实测中，2.5% 到 4% 的调用超过 300 秒，最长约 17 分钟。
:::

## 保持工具与对话连续性 {#preserve-tool-and-conversation-continuity}

运行一个真实的工具循环：模型请求工具 → harness 执行它 → 下一次调用携带对应的结果。对于流式输出，请检查完整的工具参数、终止事件和最终回答。仅凭协议兼容并不能证明模型和 harness 能配合工作。

配置了 Router Learning 防护时，它需要明确的身份标识。默认的 `scope: conversation` 要求相关调用同时携带两个请求头：

```http
x-session-id: harness-demo-session
x-conversation-id: harness-demo-conversation
```

这些只是示例值：

- **使用真实的标识。** 由 harness 或受信任的网关为你的 session 和 conversation 提供各自不同且稳定的标识。
- **`scope: session`** 只要求配置的 session 请求头。
- **缺少标识**时请求仍可路由，但防护不会保持任何模型。
- **Responses 历史**（包括 `previous_response_id`）不能替代这些请求头。

身份的优先级和隐私边界见[会话标识](../api/session-identification)。

任务进入新阶段时，harness 可以在同一会话中请求一次新的路由决策：在新阶段的第一个请求上发送 `x-vsr-protection-release: 1`，时机是用户轮次，而不是工具循环内部。被拒绝的释放会在 `x-vsr-warnings` 中返回原因。见[显式释放](../tutorials/learning/protection#explicit-release)。

:::warning 只在一个请求上发送释放请求头
不要通过 Claude Code 的 `ANTHROPIC_CUSTOM_HEADERS` 设置它，那会在每个请求上都发送。
:::

请在工具续接和用户追问时检查所选的模型。工具权限、任务状态和外层循环仍由 harness 负责；配方不配置被委派的代理。

Claude Code 发送 `x-claude-code-session-id` 而不是 `x-session-id`，Codex CLI 发送 `session-id` 和 `thread-id`。保持它们的运行及其子代理所用的身份请求头，见[委派子代理的 Agent](../tutorials/learning/protection#agents-that-delegate-to-subagents)。

### 以消息形式发送工具输出的 Agent {#agents-that-send-tool-output-as-messages}

有些 harness（例如 mini-SWE-agent）不使用协议中的工具调用。模型在回复中写出命令，harness 再把命令的输出作为下一条用户消息发回。Router 无法把这种输出与新的用户轮次区分开，因此它永远看不到工具循环，信号也会把输出当作用户写的内容来读取。

在 `scope: conversation` 下，因这类输出而改变的决策会释放当前模型，`min_turns_before_switch` 也无法保持它。请改用 `scope: session`，并为每次 agent 运行使用新的 session ID：

- **仅仅是决策改变**时，运行会保持在原来的模型上。
- **空闲超时、绕过决策、候选被排除或符合条件的[救援](../tutorials/learning/protection)**仍可释放它。
- **要在另一个模型上重新开始任务，**请用新的 session ID 开始新的运行，或直接指定具体模型，从而绕过配方路由。

## 编程入口背后的策略 {#program-the-policy-behind-the-entrypoint}

在同一个入口背后[连接模型并发布配方](../tutorials/global/models-entrypoints-serving)。信号、决策和算法决定哪些模型合格以及它们如何运行。

[Agent Routing 配方](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/agent)可以作为本地、专项和前沿模型通道的起点。使用前请检查它的模型绑定、决策模型问题和回放保留策略；仓库中的配置会记录请求和响应数据。采用策略变更之前，请用[配方评估工作流](../benchmarking/agent-evaluation-loop)测试有代表性的任务。

推理也要在决策中决定。对于有推理家族的模型（例如目录中的模型），由决策的 `modelRefs` 条目决定后端是否思考以及思考的强度，而不是 harness 自己的请求：

| 决策的 `modelRefs` 条目 | Router 发送什么 |
| --- | --- |
| 省略 `use_reasoning` | 关闭推理 |
| `use_reasoning: true` | 决策的 `reasoning_effort`；未设置时为该家族的默认值 |

无论 harness 要求什么推理强度，都是如此。例如，Claude Code 在每次调用中都会按它的强度设置请求自适应思考。请在 agent 入口背后的决策上设置 `use_reasoning: true` 和你想要的强度。见[模型推理](./model-reasoning#select-reasoning-in-a-decision)。

## 让 Agent 调用保持在热的提示缓存上 {#keep-agent-calls-on-a-warm-prompt-cache}

Agent 在每次调用时都会重发整个对话，因此每个提示的大部分是后端已经处理过的前缀。这个前缀是否仍在缓存中，很大程度上决定了调用的延迟和成本，而有三件事会让它失效。

- **换模型。** 提示缓存属于某一个模型，运行中途切换模型会在新模型上重新计算整个上下文。请按上文所述让每次运行保持在一个模型上。[multi_factor 选择器](../tutorials/algorithm/selection/multi-factor#cache-affinity)也会偏向服务过该会话上一轮的模型。
- **换副本。** 每个副本都有自己的前缀缓存，而模型的 `lb_policy`（`round_robin` 或 `least_request`，见[模型配置](model-configuration#tune-timeouts-retries-and-endpoint-health)）不会把会话固定在一个副本上。当多个副本服务 agent 会话时，请把该模型路由到按前缀缓存和负载放置请求的调度器所管理的模型池，例如 [llm-d](k8s/llm-d)（实验性集成），而不是把副本逐个列为 `backend_refs`。
- **驱逐。** 当进行中的会话需要的 KV 缓存超过 GPU 容量时，回来的会话会发现自己的前缀已被驱逐。后端的 CPU 卸载层（例如 vLLM 的 LMCache）可以把这些前缀取回来，而不必重新计算。

Harness 也可能自己让提示从头开始。当它压缩历史（用摘要替换之前的轮次以保持在上下文窗口内）时，后端会在同一模型和副本上重新计算新提示的大部分内容。这既不是 Router 造成的，也不是它能阻止的，因此压缩之后前缀复用率下降并不是路由或放置的问题。

对于 agent 入口，请关闭跨模型 KV 传输（`global.integrations.kv_transfer`）。接收方模型必须在关闭前缀缓存的情况下提供服务，而且[连接器](https://github.com/vllm-project/semantic-router/tree/main/src/kv_connector)拒绝缩放后的 RoPE，因此两个模型都无法服务超出其原生长度的上下文。
