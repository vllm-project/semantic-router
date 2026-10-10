---
title: 虚拟模型
description: 在同一 Semantic Router 部署中，为客户端提供由隔离路由策略支撑的稳定虚拟模型名。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/tutorials/global/entrypoints-and-recipes.md"
  outdated: false
---

# 虚拟模型

## 概览

入口与配方把一个 Semantic Router 部署变成一组面向目标的虚拟模型：

- **入口** 是客户端请求的模型名；
- **配方** 是处理该名称请求的路由策略；
- 提供商、模型端点和共享服务仍可供每个配方使用。

## 解决什么问题？

这种分离让 Agent Harness 可以选择低延迟、高质量或折中目标，而无需知道由哪个后端模型服务该请求。

在规范 YAML 中，`entrypoints` 保存公开名称映射，`recipes` 保存命名的路由策略。

## 各部分如何衔接 {#how-the-pieces-fit}

```text
request model name -> entrypoint -> recipe -> decision -> algorithm -> backend
```

当请求的 `model` 匹配某个 `entrypoints[].model_names` 值时，Router 只评估映射的配方。虚拟模型名随后会被该配方选出的后端替换。

顶层 `routing` 块是 `default` 配方，默认发布为 `vllm-sr/auto`。
显式声明 `recipe: default` 的入口会替换这个内置名称；如果客户端仍需使用
`vllm-sr/auto`，请把它写入该入口的 `model_names`。`auto`、`vllm-sr/flow`
等名称只有显式声明后才可用；算法由所选配方的决策决定。
若所选配方没有匹配的决策，Router 使用 `providers.defaults.model`。

具体后端模型名不同：它们会直接选择该模型并绕过配方路由。当客户端应按目标请求时使用虚拟入口；仅当有意需要那个精确后端时，才使用具体模型名。

## 配置

模型目录是共享的。每个命名配方拥有自己的信号、投影、决策、策略、算法和路由局部插件。

```yaml
routing:
  modelCards:
    - name: fast-model
    - name: accurate-model

entrypoints:
  - model_names: [vllm-sr/mom-v1-flash]
    recipe: flash
  - model_names: [vllm-sr/mom-v1-ultra]
    recipe: ultra

recipes:
  - name: flash
    description: Prefer the lowest-latency eligible backend.
    routing:
      strategy: priority
      decisions:
        - name: fast-path
          description: Serve requests with the fast model.
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: fast-model

  - name: ultra
    description: Prefer the highest-quality eligible backend.
    routing:
      strategy: priority
      decisions:
        - name: quality-path
          description: Serve requests with the accurate model.
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: accurate-model
```

客户端可通过 `/v1/models` 发现入口名称。已路由的响应包含 `x-vsr-selected-recipe`，运维人员可据此确认由哪条策略处理了请求，而无需向客户端暴露后端选择契约。

连接、协议和会话配置见[接入 Agent Harness](../../installation/agent-harness)。

## Agent 客户端的限制 {#limits-for-agent-clients}

`/v1/models` 告诉客户端有哪些虚拟名称以及它们如何解析，但不报告上下文窗口、输出
上限或能力。同一名称背后的模型可能在不同请求间变化。`vllm-sr/auto` 的条目如下：

```json
{
  "id": "vllm-sr/auto",
  "object": "model",
  "created": 1790323030,
  "owned_by": "vllm-semantic-router",
  "description": "Intelligent Router for Mixture-of-Models",
  "routing": {
    "resolution": "virtual",
    "selectable": true,
    "default_route": true,
    "recipe": "default"
  }
}
```

编码 Agent 和其他需要在发送前计算请求大小的客户端，必须在自身配置中设置这些值。
会话的任意一轮都可能到达配方可选择的任意模型，包括 `providers.defaults.model`。
因此，应按这些模型卡的交集配置客户端：

| 客户端设置 | 取值 |
| --- | --- |
| 上下文窗口 | 最小的 `context_window_size` |
| 输出上限 | 最小的 `max_output_tokens` |
| 工具调用 | 仅当所有模型都声明 `tools` 时开启 |
| 图像输入 | 仅当所有模型都声明 `vision` 或 `image_input` 时开启 |
| 推理设置 | 仅当所有模型都声明 `reasoning` 时发送 |

如果配方在以下三个模型中选择，请配置 32,768 token 的上下文窗口、8,192 token 的
输出上限，并启用工具调用。保持图像输入和推理设置关闭。

```yaml
routing:
  modelCards:
    - name: local-coder
      context_window_size: 32768
      max_output_tokens: 8192
      capabilities: [chat, tools]
    - name: reasoner
      context_window_size: 200000
      max_output_tokens: 64000
      capabilities: [chat, tools, reasoning]
    - name: vision-generalist
      context_window_size: 131072
      max_output_tokens: 16384
      capabilities: [chat, tools, vision]
```

默认情况下，Router 会跳过声明的上下文窗口小于估算输入，或声明的能力无法满足图像
等必需输入的候选模型。它不会检查输出上限，因此请求 16,384 个输出 token 时，仍
可能选中 `local-coder`。在配方上配置
[`candidate_requirements`（英文）](https://vllm-sr.ai/docs/installation/configuration#recipe-wide-candidate-and-replay-policies)
后，Router 还会在评分前检查输出上限以及工具、推理和结构化输出声明。只适合部分
候选的请求会发往其中一个；没有合格候选时，在派发前拒绝，详见
[请求预算错误（英文）](https://vllm-sr.ai/docs/api/router#request-budget-errors)。按交集配置客户端，可以让每个
候选模型对每条请求都保持可用。

## 何时使用

当一个部署必须暴露多个路由目标、策略边界或发布轨道时，使用命名入口和配方。当所有客户端应遵循同一策略时，保持单个顶层 `routing` 配置即可；现有 auto-model 流程无需额外配置。

继续阅读：

- [入口](entrypoints)：命名、请求解析、发现和校验规则。
- [配方](recipes)：策略隔离、共享基础设施、生命周期 API 和限制。
- [模型、入口与服务](models-entrypoints-serving)：端到端目录、CLI、后端绑定、服务和运维工作流。
