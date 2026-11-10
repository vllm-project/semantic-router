---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/learned/reask.md"
  outdated: false
---

# 再询问信号 {#reask-signal}

## 概览 {#overview}

`reask` 检测当前用户轮次是否在语义上重复同一对话中的近期用户轮次。在 `routing.signals.reasks` 下定义重复提问规则。

该族为学习型：使用 Router 的共享语义嵌入路径，将当前用户轮次与先前用户轮次比较。

## 主要优势 {#key-advantages}

- 捕捉隐式不满，而不要求「this is wrong」这类显式短语。
- 区分一轮重复与多轮持续不满。
- 让决策基于近期对话历史升级，而不是单条消息。
- 复用现有语义相似度栈，而不引入第二套模型表面。

## 解决什么问题？ {#what-problem-does-it-solve}

上一轮回答没用时，用户常常重述同一问题。单轮分类器可能漏掉该模式，因为抱怨是隐式而非显式的。

`reask` 将最近用户轮次与最近若干用户轮次比较，并在连续轮次保持语义相似时浮现可配置的不满信号。

## 何时使用 {#when-to-use}

在以下情况使用 `reask`：

- 重复提问应升级到更强模型
- 希望对一次重复提问与多次重复提问做不同处理
- 显式反馈稀少，但重复用户轮次仍然重要
- 路由决策应依赖同一对话中的用户历史

## 配置 {#configuration}

```yaml
routing:
  signals:
    reasks:
      - name: likely_dissatisfied
        description: Current user turn closely repeats the immediately previous user turn.
        threshold: 0.8
        lookback_turns: 1
      - name: persistently_dissatisfied
        description: Current user turn repeats the last two user turns in a row.
        threshold: 0.8
        lookback_turns: 2
```

每条规则将当前用户轮次与最近 `lookback_turns` 个先前用户轮次比较。仅当该近期连续中的每一轮都高于已配置相似度阈值时，规则才匹配。

## 依赖与限制 {#dependencies-and-limitations}

Reask 使用共享嵌入路径，并在配置了远程嵌入提供方时把近期用户轮次发给它。重复可能是有意的，而不一定是不满，因此把该信号用于升级而非惩罚。完整示例见：
[`config/fragments/signal/reask/dissatisfaction.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/reask/dissatisfaction.yaml)。
