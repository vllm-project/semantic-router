---
title: 入口
description: 通过受支持推理 API 的 model 字段，暴露用于选择路由配方的稳定虚拟模型名。
translation:
  source_commit: "7f1b814c97035e96a3780a3b8780ea5d3b6a6b24"
  source_file: "docs/tutorials/global/entrypoints.md"
  outdated: false
---

# 入口

## 概览

入口是映射到一个配方的公开虚拟模型名。客户端通过受支持的 Chat Completions、Responses 或 Messages API 中的 `model` 字段选择它，无需用 Router 专用 API 或请求头选择配方。客户端连接和会话配置见[接入 Agent Harness](../../installation/agent-harness)。

## 解决什么问题？

让 Harness 保持 `vllm-sr/mom-v1-flash` 等稳定名称，独立调整背后的模型、阈值和算法。

## 何时使用

在希望实现以下目标时创建入口：

- 发布延迟、质量、成本、安全或团队专用的路由目标；
- 在不暴露后端模型 ID 的情况下，将客户端迁移到不同策略版本；或
- 在一个 Router 部署中运行若干隔离策略。

当每条被路由的请求都应使用默认策略时，使用 `vllm-sr/auto` 或显式声明的默认入口。仅当调用方有意绕过信号、决策、算法和路由局部插件时，才使用具体的提供商模型名。

## 配置

每个入口列出一个或多个别名，以及它们选择的命名配方：

```yaml
entrypoints:
  - model_names:
      - vllm-sr/mom-v1-flash
      - company/fast
    recipe: flash

recipes:
  - name: flash
    description: Low-latency routing for interactive requests.
    routing:
      strategy: priority
      decisions: []
```

两个别名选择同一配方。客户端可以像使用任何其他 chat-completions 模型一样使用任一名称：

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/mom-v1-flash",
    "messages": [{"role": "user", "content": "Summarize this request."}]
  }'
```

入口名称不会到达提供商。配方选出后端后，Router 会把请求改写为该后端的模型名。

## 请求解析 {#request-resolution}

| 请求的模型 | Router 行为 |
| --- | --- |
| `entrypoints[].model_names` 中的值 | 只评估映射的配方。 |
| 未显式声明 `default` 入口时的 `vllm-sr/auto` | 评估来自顶层 `routing` 的 `default` 配方。 |
| 显式声明的 ReMoM、Fusion 或 Flow 入口 | 评估映射的配方，由匹配的决策选择 looper 算法。 |
| 具体的提供商模型或 LoRA 名称 | 直接发送到该后端，不经过配方路由。 |

入口会由 `/v1/models` 列出，并带有路由元数据。成功路由的响应会暴露 `x-vsr-selected-recipe`；路由回放和 Insights 也可以按配方过滤记录。

要改名默认入口，声明 `recipe: default` 并设置 `model_names`。这会替换内置的
`vllm-sr/auto`；如果旧客户端仍需使用它，请显式保留该名称。
裸 `auto` 和 looper 名称都没有隐式行为。

例如，同时发布带命名空间的默认名称和旧客户端使用的 `auto`：

```yaml
entrypoints:
  - model_names: [vllm-sr/auto, auto]
    recipe: default
```

## 命名与校验规则 {#naming-and-validation-rules}

配置加载会在以下情况拒绝入口：

- `model_names` 为空，或 `recipe` 未指向已配置的配方；
- 同一虚拟名被多个入口占用；或
- 虚拟名与提供商模型、LoRA 或其他有效入口（包括内置默认名）冲突。

选择描述稳定客户端契约的名称，而不是当前后端。不要把租户数据或密钥放进名称：入口会出现在模型发现、响应元数据、指标和运维记录中。

入口是策略选择器，不是安全边界。各配方共享 Router 进程和已配置的基础设施；当租户需要更强隔离时，请使用网络、计算和存储隔离。

端到端 CLI 工作流请从[模型、入口与服务](models-entrypoints-serving)开始，或继续阅读 [配方](recipes) 了解入口所拥有的策略。
