---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/response-cache.md"
  outdated: false
---

# Response Cache

## 概览

`response_cache` 是路由局部插件：复用语义相近的历史响应。

对应 `config/plugin/response-cache/high-recall.yaml` 与 `config/plugin/response-cache/memory.yaml`。

## 主要优势

- 仅在从缓存受益的路由上复用先前响应。
- 路由局部阈值与全局存储设置分离。
- 不同路由可采用不同缓存策略。

## 解决什么问题？

部分路由强依赖复用，部分需要每次全新生成。`response_cache` 将复用策略留在路由局部，而非默认全局启用缓存。

## 何时使用

- 一条路由应在查询非常相似时偏好缓存响应
- 不同路由需要不同相似度阈值或 TTL
- 路由使用在 `global.stores.response_cache` 配置的共享语义缓存后端

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: response_cache
  configuration:
    enabled: true
    mode: exact_then_semantic
    scope: user
    semantic:
      similarity_threshold: 0.92
    ttl_seconds: 86400
    request_controls:
      enabled: true
      allowed: [no-cache, no-store, bypass, max-age, ttl]
      max_ttl_seconds: 86400
```
