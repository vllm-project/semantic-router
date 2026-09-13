---
title: 虚拟模型
description: 在同一 Semantic Router 部署中，为客户端提供由隔离路由策略支撑的稳定虚拟模型名。
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
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

这种分离让应用可以选择低延迟、高质量或折中目标，而无需知道由哪个后端模型服务该请求。

在规范 YAML 中，`entrypoints` 保存公开名称映射，`recipes` 保存命名的路由策略。

## 各部分如何衔接 {#how-the-pieces-fit}

```text
request model name -> entrypoint -> recipe -> decision -> algorithm -> backend
```

当请求的 `model` 匹配某个 `entrypoints[].model_names` 值时，Router 只评估映射的配方。虚拟模型名随后会被该配方选出的后端替换。

顶层 `routing` 块仍是 `default` 配方。请求 `vllm-sr/auto`、`auto` 或其他已配置的 auto 别名时，使用该默认策略。若所选配方没有匹配的决策，Router 使用 `providers.defaults.model`。

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

## 何时使用

当一个部署必须暴露多个路由目标、策略边界或发布轨道时，使用命名入口和配方。当所有客户端应遵循同一策略时，保持单个顶层 `routing` 配置即可；现有 auto-model 流程无需额外配置。

继续阅读：

- [入口](entrypoints)：命名、请求解析、发现和校验规则。
- [配方](recipes)：策略隔离、共享基础设施、生命周期 API 和限制。
- [模型、入口与服务](models-entrypoints-serving)：端到端目录、CLI、后端绑定、服务和运维工作流。
