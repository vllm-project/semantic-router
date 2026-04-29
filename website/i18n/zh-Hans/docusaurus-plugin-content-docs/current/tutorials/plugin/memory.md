---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/memory.md"
  outdated: true
---

# Memory

## 概览

`memory` 是路由局部插件：检索与存储对话记忆。

对应 `config/plugin/memory/session-memory.yaml`。

## 主要优势

- 记忆行为局部在受益的路由上。
- 在一个插件中支持检索与自动存储。
- 路由局部记忆策略与共享后端存储配置分离。

## 解决什么问题？

并非每条路由都应承担记忆的复杂度或隐私成本。`memory` 让已匹配路由选用会话感知行为，共享库仍在 `global.stores.memory` 配置。

## 何时使用

- 路由应检索先前对话上下文
- 路由应自动存储有用新轮次
- 记忆设置应仅局部在一类路由上

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: memory
  configuration:
    enabled: true
    retrieval_limit: 5
    similarity_threshold: 0.72
    auto_store: true
```
