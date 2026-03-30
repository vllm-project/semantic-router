---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/global/overview.md"
  outdated: false
---

# Global

## 概览

`global:` 是**路由器级**覆盖层。

与 `signal/`、`decision/`、`algorithm/`、`plugin/` 不同，本节非路由局部：定义共享运行时行为、共享后端服务、内置模型资产与共享能力模块。

## 主要优势

- 为路由器提供统一的共享配置入口。
- 避免在路由间重复共享后端设置。
- 路由匹配留在 `routing:`，全路由器行为留在 `global:`。
- 与路由器内置默认值配合，只覆盖所需项。

## 解决什么问题？

部分配置属于整台路由器，而非单条路由。若状态泄漏到路由局部配置，路由复用变难，共享与局部也难区分。

`global:` 在内置默认之上持有稀疏、全路由器覆盖。

## 何时使用

在以下情况使用 `global:`：

- 设置应对多条路由生效
- 共享存储或运行时服务只需配置一次
- 内置系统模型或运行时策略需要覆盖
- 行为不针对单条已匹配决策

## 配置

规范位置：

```yaml
global:
  router:
    config_source: file
  services:
    observability:
      metrics:
        enabled: true
```

最新 global 文档与主要运行时分组一致：

| Global 区域 | 示例 | 文档 |
| ----------- | ---- | ---- |
| 路由器与服务 | `router.config_source`、`router.model_selection`、`services.api`、`services.response_api`、`services.observability`、`services.router_replay` | [API 与可观测性](./api-and-observability) |
| 存储与集成 | `stores.semantic_cache`、`stores.memory`、`stores.vector_store`、`integrations.tools`、`integrations.looper` | [存储与工具](./stores-and-tools) |
| 模型目录与模块 | `model_catalog.embeddings`、`model_catalog.external`、`model_catalog.system`、`model_catalog.modules.prompt_guard`、`model_catalog.modules.classifier`、`model_catalog.modules.hallucination_mitigation` | [安全、模型与策略](./safety-models-and-policy) |

注意：

- `global:` 保持稀疏；可能时依赖路由器默认
- 除非有意由 Kubernetes CRD 协调驱动运行时配置，否则保持 `global.router.config_source` 为 `file`
- 共享后端服务放在 `global:`
- 路由局部匹配放在 `routing:`
