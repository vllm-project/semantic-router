---
title: 配方
description: 在同一 Semantic Router 部署中定义隔离的路由策略，同时共享提供商模型和平台服务。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/global/recipes.md"
  outdated: false
---

# 配方

## 概览

配方是完整的路由策略边界。它拥有一个目标所用的证据和控制流：信号、投影、决策、路由策略、选择或 looper 算法，以及路由局部插件。

## 解决什么问题？

配方避免两种糟糕的扩展模式：把无关策略混进一张决策图，或为每个路由目标复制整台 Router 部署。它们在复用昂贵模型端点和共享服务的同时，把策略与运行时状态分开。

## 何时使用

当消费者需要不同的延迟、质量、成本、隐私或安全策略时，添加配方。它们也适用于在当前策略旁边预演新策略，然后再把客户端迁移到其入口。

当单一策略服务全部流量时，保持一个顶层 `routing` 块即可。Router 将该块视为 `default` 配方，因此现有配置无需改写。

## 配置

提供商绑定和 model cards 留在顶层。命名配方从其自身的 routing 块引用这些共享模型名：

```yaml
routing:
  modelCards:
    - name: local-model
    - name: general-model

entrypoints:
  - model_names: [vllm-sr/privacy-v1]
    recipe: privacy

recipes:
  - name: privacy
    description: Keep prompts containing sensitive identifiers on the local model.
    routing:
      strategy: priority
      signals:
        pii:
          - name: sensitive-input
            threshold: 0.5
      decisions:
        - name: local-sensitive-route
          description: Keep detected PII on the local backend.
          priority: 200
          rules:
            operator: AND
            conditions:
              - type: pii
                name: sensitive-input
          modelRefs:
            - model: local-model
        - name: general-route
          description: Handle remaining requests with the general backend.
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: general-model
```

名称在所属配方内解析。两个配方都可以定义名为 `sensitive-input` 的信号或决策；任何一方都不能引用对方配方中的定义。

## 隔离什么、共享什么 {#what-is-isolated-and-what-is-shared}

| 配方局部 | 部署共享 |
| --- | --- |
| 信号及其阈值 | 提供商模型和后端端点 |
| 投影和依赖图 | 顶层 `routing.modelCards` |
| 决策、优先级和路由策略 | 共享分类器和嵌入资产 |
| 选择与 looper 策略 | API、身份、可观测性和传输设置 |
| 路由局部插件 | 外部存储和集成服务 |
| 缓存、回放、学习、会话和指标命名空间 | 模型文件和服务连接 |

共享基础设施不会让策略变成全局。例如，共享的 PII 模型可以为多个配方服务，而每个配方定义自己的 PII 信号、阈值和决策行为。

## 安全地更新配方 {#update-recipes-safely}

管理 API 可以校验并更改一个配方，而无需替换文档的其余部分：

| 方法和路径 | 用途 |
| --- | --- |
| `GET /api/v1/config/recipes` | 列出配方及其入口；保留响应中的 `ETag`。 |
| `POST /api/v1/config/recipes/validate` | 校验提议的配方，不写入也不重载。 |
| `PUT /api/v1/config/recipes/{name}` | 创建或替换一个配方及其入口。 |
| `DELETE /api/v1/config/recipes/{name}` | 删除未被引用的命名配方。 |

变更需要在 `If-Match` 中提供当前 `ETag`。缺少前置条件返回 `428`；过期值返回 `412`。被接受的变更会校验完整配置，原子写入并备份，触发 Router 运行时激活，并返回新的 `ETag`。激活可能在 `202` 响应之后继续；在将新策略视为已激活之前，请轮询 `/api/v1/config/hash`。

`default` 配方不能删除。删除其他配方之前，请移除或迁移所有引用它的入口。端点详情见[管理 API 参考](../../api/apiserver)。

## 限制与安全边界 {#limits-and-security-boundaries}

- 命名配方不能声明 `routing.modelCards`；所有配方使用共享的顶层目录。
- 信号、投影和决策引用必须在同一配方内解析。
- 若没有决策匹配，路由回退到部署已配置的默认提供商模型，而不是另一个配方。
- 配方隔离不会隔离 Router 进程、网络、提供商凭据或后台服务。当这些必须成为租户边界时，请使用单独部署。
- 路由插件和共享服务可能持久化提示词、响应和路由元数据。对每个已启用的存储应用保留、访问和加密策略。

浏览[完整配方示例及其 Model Cards](https://github.com/vllm-project/semantic-router/tree/main/config/recipes)。目录、CLI、后端绑定和服务工作流请从[模型、入口与服务](models-entrypoints-serving)开始。
