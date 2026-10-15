---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/plugin/tool-selection.md"
  outdated: false
---

# 工具选择

## 概览

`tool_selection` 是一个决策插件，控制已匹配路由如何选择工具。
它支持两种模式：

- `add`：从工具数据库检索工具
- `filter`：过滤传入请求中已有的工具

## 主要优势

- 将路由决策逻辑与工具检索/过滤行为分开。
- 同时支持从数据库添加工具，以及对请求工具做语义过滤。
- 在保持与路由局部工具策略兼容的同时，明确声明选择行为。

## 解决什么问题？

不同路由需要不同的工具选择行为。有些路由应从精选数据库添加工具，另一些应只保留调用方提供集合中最相关的工具。`tool_selection` 为这两种情况提供一份插件契约，并带有阈值、`top_k` 和保留行为等按路由控制。

## 何时使用

- 某个决策应从 `tools_db` 添加最相关的工具
- 某个决策应按语义过滤调用方提供的 `tools`
- 按路由的工具选择模式必须明确且可配置

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: filter
      relevance_threshold: 0.25
      preserve_count: 2
```

对于 add 模式：

```yaml
plugins:
  - type: tool_selection
    configuration:
      enabled: true
      mode: add
      tools_db_path: config/tools_db.json
      top_k: 5
      similarity_threshold: 0.35
```

`add` 模式需要已填充的工具数据库；`filter` 模式只考虑调用方已提供的工具。语义相关性不是授权，因此请单独强制工具权限。完整示例见：
[`add-from-database.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/add-from-database.yaml)
和
[`filter-request-tools.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tool-selection/filter-request-tools.yaml)。
