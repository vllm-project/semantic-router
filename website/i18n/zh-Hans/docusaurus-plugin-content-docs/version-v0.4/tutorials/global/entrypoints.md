---
title: 入口
description: 通过标准的 OpenAI 兼容 model 字段，暴露用于选择路由配方的稳定虚拟模型名。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/global/entrypoints.md"
  outdated: false
---

# 入口

## 概览

入口是映射到一个配方的公开虚拟模型名。客户端通过普通的 OpenAI 兼容 `model` 字段选择它，因此不需要 Router 专用 API 或请求头。

## 解决什么问题？

入口解决常见的耦合问题：应用可以请求稳定目标，例如 `vllm-sr/mom-v1-flash`，同时运维人员可以更改该目标背后的模型、阈值或算法。

## 何时使用

在希望实现以下目标时创建入口：

- 发布延迟、质量、成本、安全或团队专用的路由目标；
- 在不暴露后端模型 ID 的情况下，将客户端迁移到不同策略版本；或
- 在一个 Router 部署中运行若干隔离策略。

当每条被路由的请求都应使用默认策略时，使用已配置的 auto 别名。仅当调用方有意绕过信号、决策、算法和路由局部插件时，才使用具体的提供商模型名。

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
| `vllm-sr/auto`、`auto` 或其他已配置的 auto 别名 | 评估来自顶层 `routing` 的 `default` 配方。 |
| 已配置的 ReMoM、Fusion 或 Flow 虚拟 slug | 在 `default` 配方中运行该 looper。 |
| 具体的提供商模型或 LoRA 名称 | 直接发送到该后端，不经过配方路由。 |

入口会由 `/v1/models` 列出，并带有路由元数据。成功路由的响应会暴露 `x-vsr-selected-recipe`；路由回放和 Insights 也可以按配方过滤记录。

## 命名与校验规则 {#naming-and-validation-rules}

配置加载会在以下情况拒绝入口：

- `model_names` 为空，或 `recipe` 未指向已配置的配方；
- 同一虚拟名被多个入口占用；或
- 虚拟名与提供商模型、LoRA、auto 别名或 looper slug 冲突。

选择描述稳定客户端契约的名称，而不是当前后端。不要把租户数据或密钥放进名称：入口会出现在模型发现、响应元数据、指标和运维记录中。

入口是策略选择器，不是安全边界。各配方共享 Router 进程和已配置的基础设施；当租户需要更强隔离时，请使用网络、计算和存储隔离。

端到端 CLI 工作流请从[模型、入口与服务](models-entrypoints-serving)开始，或继续阅读 [配方](recipes) 了解入口所拥有的策略。
