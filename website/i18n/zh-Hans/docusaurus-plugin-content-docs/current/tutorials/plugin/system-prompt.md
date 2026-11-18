---
is_mtpe: false
translation:
  source_commit: "7c874be2"
  source_file: "docs/tutorials/plugin/system-prompt.md"
  outdated: false
---

# 系统提示词

## 概览

`system_prompt` 是一个路由局部插件，用于在匹配的流量中插入或修改系统提示词。

## 主要优势

- 将指令塑形限制在对应路由内。
- 明确声明提示词模式，而不是将其隐藏在应用代码中。
- 适用于专家、角色或工作流专用路由。

## 解决什么问题？

有些路由需要不同于路由器默认值的指令层。`system_prompt` 允许这些路由附加额外的提示词上下文，同时不影响无关流量。

## 何时使用

- 某个路由需要专家或特定角色的指令层
- 提示词插入应在决策匹配后进行
- 提示词策略应在路由配置中保持可见

## 配置

在 `recipes[].decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: system_prompt
    configuration:
      enabled: true
      mode: insert
      system_prompt: You are a domain expert. Answer precisely and state tradeoffs.
```

`mode: insert`（默认）把该提示词放在调用方的系统提示词之前。`mode: replace` 丢弃调用方发送的所有 system 和 developer 指令（包括 Responses 的 `instructions` 字段），并以该提示词取而代之，`context` 插件注入系统提示词的上下文保留在该提示词之后。其他任何模式都会在加载时失败，`vllm-sr config migrate` 会把它写成 `replace`。这类模式原来只替换调用方的第一条 system 消息，因此改写后调用方的其他 system 和 developer 指令也会被丢弃。

插入的文本会发送给选定的模型，并可能改变缓存标识和模型行为。不得在其中放入 secret 或不受信任的调用方文本。完整示例见：
[`config/fragments/plugin/system-prompt/expert.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/system-prompt/expert.yaml)。
