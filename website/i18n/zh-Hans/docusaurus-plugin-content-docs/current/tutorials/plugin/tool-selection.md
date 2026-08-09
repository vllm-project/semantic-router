---
translation:
  source_commit: "e0e7df56"
  source_file: "docs/tutorials/plugin/tool-selection.md"
  outdated: false
---

# 工具选择（Tool Selection）

## 概览

`tool_selection` 是一个 decision 插件，用于控制如何为匹配的路由选择工具。
它支持两种模式：

- `add`：从工具数据库中检索工具
- `filter`：筛选传入请求中已有的工具

它对应 `config/fragments/plugin/tool-selection/` 下的片段。

## 主要优势

- 将路由 decision 逻辑与工具检索/筛选行为分离。
- 同时支持由数据库驱动的工具添加和对请求工具的语义筛选。
- 在明确选择行为的同时，保持与路由局部工具策略的兼容性。

## 解决什么问题？

不同路由需要不同的工具选择行为。有些路由应从经过整理的数据库中添加工具，另一些路由则应只保留调用方所提供工具中最相关的部分。`tool_selection` 为这两类场景提供统一的插件契约，并支持 threshold、`top_k` 和保留行为等按路由控制项。

## 何时使用

- decision 应从 `tools_db` 添加最相关的工具
- decision 应对调用方提供的 `tools` 进行语义筛选
- 必须明确配置按路由的工具选择模式

## 配置

在 `routing.decisions[].plugins` 下使用此片段：

```yaml
plugin:
  type: tool_selection
  configuration:
    enabled: true
    mode: filter
    relevance_threshold: 0.55
    preserve_count: 2
```

对于 add 模式：

```yaml
plugin:
  type: tool_selection
  configuration:
    enabled: true
    mode: add
    top_k: 5
```
