---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/system-prompt.md"
  outdated: false
---

# System Prompt

## 概览

`system_prompt` 是路由局部插件：在已匹配流量上插入或修改 system prompt。

对应 `config/plugin/system-prompt/expert.yaml`。

## 主要优势

- 指令塑造局部在路由。
- 提示模式显式，而非藏在应用代码中。
- 适合专家、人设或工作流专用路由。

## 解决什么问题？

部分路由需要与路由器默认不同的指令层。`system_prompt` 让这些路由附加额外提示上下文而不影响无关流量。

## 何时使用

- 一条路由需要专家或人设专用指令层
- 提示插入应在决策匹配之后发生
- 提示策略应在路由配置中可见

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: system_prompt
  configuration:
    enabled: true
    mode: insert
    system_prompt: You are a domain expert. Answer precisely, state tradeoffs, and keep the response actionable.
```
