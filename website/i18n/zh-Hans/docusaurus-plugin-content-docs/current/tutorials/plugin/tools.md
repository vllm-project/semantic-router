---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/plugin/tools.md"
  outdated: false
---

# 工具

## 概览

`tools` 是一个路由局部插件，用于工具过滤和语义工具选择。

## 主要优势

- 将工具策略附着在已匹配的路由上。
- 让一条路由禁用工具，而另一条路由过滤或按语义选择它们。
- 与全局工具数据库组合，而不是让 `routing.decisions[]` 过载。

## 解决什么问题？

工具行为是路由策略的一部分。有些路由应完全剥离工具，有些应原样透传，有些应约束语义工具候选池。`tools` 插件明确声明该路由局部契约。

## 何时使用

- 某条路由应禁用所有工具
- 某条路由应从全局工具数据库按语义选择工具
- 某条路由应使用显式允许/阻止列表限制工具访问
- 隐私路由应让工具历史对路由可见，但把先前的工具/函数调用和结果从所选模型请求中省略

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: tools
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

当所选后端不得接收先前的 assistant 工具/函数调用或工具/函数结果消息时，设置 `mode: none` 并配合 `strip_tool_history: true`。路由器在信号和决策评估之后应用该策略，因此不会更改哪条路由匹配。它只更改绑定提供商的请求正文。校验会拒绝将 `strip_tool_history: true` 与任何其他工具模式一起使用。

工具选择控制到达模型的内容；它不授权工具执行。请在工具服务上强制权限，并将工具 schema 和结果视为绑定提供商的内容。完整示例见：
[`config/fragments/plugin/tools/semantic-select.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/tools/semantic-select.yaml)。
