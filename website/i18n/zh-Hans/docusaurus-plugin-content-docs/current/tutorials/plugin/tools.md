---
translation:
  source_commit: "baa07413"
  source_file: "docs/tutorials/plugin/tools.md"
  outdated: false
---

# Tools

## 概览

`tools` 是路由局部插件：工具过滤与语义工具选择。

对应 `config/plugin/tools/semantic-select.yaml`。

## 主要优势

- 工具策略挂在已匹配路由上。
- 一条路由可禁用工具，另一条可过滤或语义选择工具。
- 与全局工具库组合，而非过载 `routing.decisions[]`。

## 解决什么问题？

工具行为是路由策略的一部分。部分路由应完全剥离工具，部分原样传递，部分应约束语义工具候选池。`tools` 插件显式表达该路由局部约定。

## 何时使用

- 路由应禁用所有工具
- 路由应从全局工具库语义选择工具
- 路由应用显式允许/阻止列表限制工具访问

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: tools
  configuration:
    enabled: true
    mode: filtered
    semantic_selection: true
    allow_tools:
      - docs.search
      - tickets.lookup
    block_tools:
      - admin.delete
```
