---
is_mtpe: true
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/signal/learned/turn-relation.md"
  outdated: false
---

# 轮次关系信号 {#turn-relation-signal}

## 概览 {#overview}

`turn_relation` 询问决策模型：当前用户轮次与之前的对话是什么关系——延续当前话题（`continue`）、重复之前的提问（`reask`）、转向无关话题（`shift`），或回到更早的话题（`return`）。它取代了 `reasks` 和 `topic_continuity` 信号。

它对应 config/fragments/signal/turn-relation，在 `recipes[].signals.turn_relation` 下声明。

## 主要优势 {#key-advantages}

- 一个问题即可覆盖重复提问、追问和话题切换。
- `input.previous_turns` 限定问题读取多少历史。
- 标签选择规则匹配哪些关系。

## 解决什么问题？ {#what-problem-does-it-solve}

重复提问的用户通常对回答不满意；切换话题的用户不再需要保留的历史。`turn_relation` 让决策可以基于这两种情况路由。

## 何时使用 {#when-to-use}

在以下情况使用 `turn_relation`：

- 重复的请求应交给更强的模型
- 话题切换时应丢弃或重置对话上下文

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      turn_relation:
        - name: likely_dissatisfied
          description: The current user turn repeats the previous one.
          labels: [reask]
          input:
            previous_turns: 1
        - name: topic_boundary
          description: Whether the live turn still depends on retained history.
          labels: [continue, shift]
          input:
            previous_turns: 8
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称 |
| `description` | 规则检测的内容 |
| `labels` | 规则匹配的关系，取自 `continue`、`reask`、`shift` 和 `return` |
| `input.previous_turns` | 问题读取的之前用户轮次数（默认 4） |
| `deployment` | 回答该规则的部署；否则使用 `model_bindings.turn_relation`，再否则使用决策模型 |
| `timeout_ms` | 该问题的时限 |
