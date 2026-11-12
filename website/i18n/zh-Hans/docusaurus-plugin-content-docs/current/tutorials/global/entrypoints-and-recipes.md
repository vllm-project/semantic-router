---
is_mtpe: false
title: 虚拟模型
description: 在同一 Semantic Router 部署中，为客户端提供由隔离路由策略支撑的稳定虚拟模型名。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/global/entrypoints-and-recipes.md"
  outdated: false
---

# 虚拟模型 {#virtual-models}

## 概览 {#overview}

入口与配方把一个 Semantic Router 部署变成一组面向目标的虚拟模型：

- **入口** 是客户端请求的模型名；
- **配方** 是处理该名称请求的路由策略；
- 提供方、模型端点和共享服务仍可供每个配方使用。

## 解决什么问题？ {#what-problem-does-it-solve}

这种分离让 agent harness 可以选择低延迟、高质量或折中的目标，而无需知道由哪个后端模型服务该请求。

在规范 YAML 中，`entrypoints` 保存公开名称映射，`recipes` 保存具名的路由策略。

## 各部分如何衔接 {#how-the-pieces-fit}

```text
request model name -> entrypoint -> recipe -> decision -> algorithm -> backend
```

当请求的 `model` 匹配某个 `entrypoints[].model_names` 值时，Router 只评估映射的配方。随后，虚拟模型名会被该配方选出的后端替换。

- **一个配方**会以 `vllm-sr/auto` 发布，不需要入口。
- **多个配方**需要恰好一个标记 `default: true` 的 chat 入口；它的配方就是默认配方。
- **名字是显式的。** 默认配方的显式入口会替换内置名称，因此如果客户端仍需使用 `vllm-sr/auto`，请把它写入该入口的 `model_names`。`auto`、`vllm-sr/flow` 等名称只有在声明后才可用。
- **没有匹配**时，请求交给 `providers.defaults.model`。

算法由所选配方的决策决定。

具体后端模型名不同：它们直接选择该模型，并绕过配方路由。当客户端应按目标请求时使用虚拟入口；只有确实需要那个精确后端时，才使用具体模型名。

## 配置 {#configuration}

模型目录是共享的。每个配方拥有自己的信号、投影、决策、strategy、算法和路由局部插件。

```yaml
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - model_names: [vllm-sr/mom-v1-flash]
    recipe: flash
  - model_names: [vllm-sr/mom-v1-ultra]
    recipe: ultra
recipes:
  - name: default
  - name: flash
    description: Prefer the lowest-latency eligible backend.
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
providers:
  models:
    - name: fast-model
    - name: accurate-model
```

共享的 strategy 和 fallback 默认值位于 `global.router`：

- **没有父配方。** 默认配方不是其他配方的父级。每个 `recipes[]` 条目直接从全局默认值补齐缺失的 strategy 和 fallback 字段，决策还可以覆盖自己的 fallback。
- **部分覆盖。** `fallback: {enabled: false}` 只改变这一个字段，保留共享的超时和重试设置。
- **直接请求**使用全局 fallback 策略。运行时默认的 strategy 是 `priority`。

回放采集默认值是 `global.plugins` 中的 `router_replay` 插件，存储位于 `global.services.router_replay`；配方或决策的 `router_replay` 插件逐字段覆盖它们。

客户端可通过 `/v1/models` 发现入口名称。经过路由的响应包含 `x-vsr-selected-recipe`，运维人员可据此确认由哪条策略处理了请求，而无需向客户端暴露后端选择契约。

连接、协议和会话设置见[接入 agent harness](../../installation/agent-harness)。

## Agent 客户端的限制 {#limits-for-agent-clients}

`/v1/models` 告诉客户端有哪些虚拟名称以及它们如何解析，但不报告上下文窗口、输出上限或能力，而且同一名称背后的模型可能在不同请求间变化。`vllm-sr/auto` 的条目如下：

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

编码 Agent 和其他需要在发送前计算请求大小的客户端，必须在自身配置中设置这些值。会话的任意一轮都可能到达配方可选择的任意模型，包括 `providers.defaults.model`，因此应按这些模型卡的交集配置客户端：

| 客户端设置 | 取值 |
| --- | --- |
| 上下文窗口 | 最小的 `context_window_size` |
| 输出上限 | 最小的 `max_output_tokens` |
| 工具调用 | 仅当所有模型都声明 `tools` 时开启 |
| 图像输入 | 仅当所有模型都声明 `vision` 或 `image_input` 时开启 |
| 推理设置 | 仅当所有模型都声明 `reasoning` 时发送 |

如果配方在以下三个模型中选择，请配置 32,768 token 的上下文窗口、8,192 token 的输出上限，并启用工具调用。保持图像输入和推理设置关闭。

```yaml
recipes:
  - name: default
providers:
  models:
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

默认情况下，Router 会跳过声明的上下文窗口小于估算输入，或声明的能力无法满足图像等必需输入的候选：

- **保守的估算。** 工具调用参数等结构化数据按每字节一个 token 计，图片、音频、视频和文件各有自己的预留。因此，编码 Agent 的长会话可能被估算得超过它实际放得下的窗口。
- **第二次计数。** 当估算超过所有候选的窗口时，Router 会去掉这些预留重新计数，并保留放得下的候选。一个都放不下时，最宽的候选保留：估算从不拒绝请求，真实长度由后端检查。
- **Router 的计数。** 字节数可能超出某个窗口的请求，会用每个模型的[分词器](../../installation/model-configuration#check-prompts-against-the-context-window)计数。计数超出它能到达的每个窗口的区间时（用模型自己的分词器时为 3%，用代理分词器时为 30%），请求会在任何信号运行之前以 `400` `context_length_exceeded` 被拒绝；在区间之内则继续发送，由后端决定。
- **默认不检查输出上限**，因此请求 16,384 个输出 token 时，仍可能选中 `local-coder`。

在配方上配置 [`candidate_requirements`](../../installation/configuration#recipe-candidate-requirements) 后，Router 还会在打分前检查输出上限以及工具、推理和结构化输出声明。一个候选都不适合的请求会在派发前被拒绝；见[请求预算错误](../../api/router#request-budget-errors)。

:::tip
按交集配置客户端，可以让每个候选模型对每条请求都保持可用。
:::

## 何时使用 {#when-to-use}

当一个部署必须暴露多个路由目标、策略边界或发布轨道时，使用入口和多个配方。当所有客户端应遵循同一策略时，保持单个配方即可；它的 `vllm-sr/auto` 入口无需额外配置。

继续阅读：

- [入口](entrypoints)：命名、请求解析、发现和校验规则。
- [配方](recipes)：策略隔离、共享基础设施、生命周期 API 和限制。
- [模型、入口与服务](models-entrypoints-serving)：端到端目录、CLI、后端绑定、服务和运维工作流。
