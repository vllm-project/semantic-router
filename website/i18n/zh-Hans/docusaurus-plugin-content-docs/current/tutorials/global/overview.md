---
translation:
  source_commit: "bce357c513f391824e8320267d03977794c20f76"
  source_file: "docs/tutorials/global/overview.md"
  outdated: false
---

# 全局配置

## 概览

`global:` 包含路由器级设置和共享基础设施。它是路由局部 `routing:` 树的对应物：在 `global:` 下定义一次服务或存储，然后通过信号、算法或插件让各路由选择加入。

## 主要优势

- 为所有配方和路由定义一次共享基础设施。
- 将路由局部策略与平台级服务分开。
- 明确模型、存储和外部服务依赖。

## 解决什么问题？

嵌入运行时、API、存储后端、可观测性和辅助服务是共享资源。把它们放在决策之外，可以避免重复连接，并使数据和信任边界可见。

## 何时使用

当行为或基础设施由多个配方共享时，使用 `global:`。将路由匹配、候选模型、算法和插件放在 `routing:` 中。全局设置在配方之间共享；信号、投影和决策以配方为范围，算法和插件属于各个决策。

## 配置

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
  stores:
    response_cache:
      enabled: true
      backend_type: memory
```

全局配置有五组：

| 组 | 负责 | 指南 |
|---|---|---|
| `global.router` | Router 引擎控制、选择默认值、流式 body 策略、学习 | [算法](../algorithm/overview)、[路由学习](../learning/overview) |
| `global.services` | API、Response API、可观测性、authz、速率限制、管理 API、启动状态、回放 | [API 与可观测性](./api-and-observability) |
| `global.stores` | 响应缓存、memory、向量存储 | [存储与工具](./stores-and-tools) |
| `global.integrations` | 工具目录和 Looper 端点/状态 | [存储与工具](./stores-and-tools) |
| `global.model_catalog` | 嵌入、系统模型、外部辅助、知识库、能力模块 | [Router 运行时](../../installation/native-backends) |

入口和命名配方是顶层对象，而不是全局设置；见[虚拟模型](./entrypoints-and-recipes)。
远程文本嵌入见[运行时嵌入](../../installation/runtime/embeddings)。

## 运维边界 {#operational-boundaries}

- 覆盖应保持稀疏；省略的字段继承 Router 默认值。
- 将凭据放在环境变量或 Kubernetes Secrets 中，而不是字面 YAML 值。
- 持久存储可能包含提示词、响应、嵌入、记忆或回放记录。请有意设置后端认证、传输安全、保留策略和租户/用户范围。
- 内置推理行为来自 `providers.models[].catalog`。自托管模型可以选择内置家族，或在 `providers.models[].reasoning` 中定义内联行为；两者都不属于 `global:`。
- 完整配置参考见 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。
