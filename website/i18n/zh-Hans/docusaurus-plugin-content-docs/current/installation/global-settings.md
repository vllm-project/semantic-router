---
title: 全局设置
sidebar_label: 全局设置
description: global 块保存 Router 范围的设置和每个 recipe 都能使用的共享基础设施。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/installation/global-settings.md"
  outdated: false
---

# 全局设置 {#global-settings}

`global:` 保存 Router 范围的设置和共享基础设施：Router 引擎、其服务、存储、集成、模型目录和默认插件。在 `global:` 下把一项服务或一个存储定义一次，然后通过信号、算法或插件让各条路由按需启用它。

```yaml title="config.yaml"
global:
  services:
    observability:
      metrics:
        enabled: true
  stores:
    response_cache:
      backend_type: memory
  plugins:
    - type: response_cache
      configuration:
        enabled: true
```

## 为什么单独一个块 {#why-a-separate-block}

- **共享基础设施只定义一次**，供所有 recipe 和路由使用，而不必在每个决策中重复连接。
- **让路由本地的策略保持独立**于平台范围的服务：路由匹配、候选模型、算法和插件留在 `routing:` 中。
- **让依赖显式可见。** 模型、存储和外部服务集中在一处，其数据边界和信任边界一目了然。

:::note
全局设置在各 recipe 之间共享。信号、投影和决策属于 recipe 范围，算法和插件属于各个决策。入口和命名 recipe 是顶层对象，而不是全局设置；参见[虚拟模型](../tutorials/global/entrypoints-and-recipes.md)。
:::

## 六个分组 {#the-six-groups}

| 分组 | 拥有 | 指南 |
| --- | --- | --- |
| `global.router` | Router 引擎控制、选择默认值、流式请求体策略、路由输入、学习 | [算法](../tutorials/algorithm/overview.md)、[Router 学习](../tutorials/learning/overview.md)、[路由输入](../tutorials/signal/overview.md#choose-the-user-text-signals-read) |
| `global.services` | API、Response API、可观测性、authz、限流、管理 API、启动状态、replay | [API 与可观测性](../tutorials/global/api-and-observability.md) |
| `global.stores` | 响应缓存、memory、向量存储 | [存储与工具](../tutorials/global/stores-and-tools.md) |
| `global.integrations` | 工具目录和 Looper 运行时/状态 | [存储与工具](../tutorials/global/stores-and-tools.md) |
| `global.model_catalog` | 部署、绑定、准入、嵌入模型和决策模型 | [把 Router 连接到模型运行时](../model-runtime/deploy/connect-the-router.md) |
| `global.plugins` | 每个请求都运行的默认插件，例如 `context`（memory、`compress.signals`）和 `router_replay` | [插件](../tutorials/plugin/overview.md) |

远程文本嵌入见[嵌入](../model-runtime/guides/embeddings.md#use-an-external-embedding-service)。

## 运维边界 {#operational-boundaries}

- **保持覆盖项精简。** 省略的字段继承 Router 默认值。
- **不要把密钥写进 YAML。** 把凭证放在环境变量或 Kubernetes Secret 中，而不是写成字面的 YAML 值。
- **模型推理按模型设置。** 内置推理行为来自 `providers.models[].catalog`。自托管模型可以选择一个内置家族，或在 `providers.models[].reasoning` 中定义内联行为；两者都不属于 `global:`。

:::warning
持久化存储可能包含提示词、回答、嵌入、记忆或 replay 记录。请审慎设置后端认证、传输安全、保留期限以及租户或用户范围。
:::

完整的配置参考见 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。
