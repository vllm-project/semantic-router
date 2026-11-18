---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/turn-relation.md"
  outdated: false
---

# 轮次关系信号 {#turn-relation-signal}

## 概述 {#overview}

`turn_relation` 询问决策模型，当前用户轮次与之前的对话是什么关系。它提出 `turn_relation` [预设](../../../system-one/reference/presets#preset-catalog)问题：该轮次 `continue`（延续当前话题）、`reask`（重新询问之前某轮问过的内容）、`shift`（转向无关话题）或 `return`（回到之前的话题）。它取代了 `reasks` 和 `topic_continuity` 信号。

## 主要优势 {#key-advantages}

- 一个问题就能覆盖重复提问、追问和话题转换。
- `input.previous_turns` 限制问题读取的历史长度。
- 用 labels 选择规则匹配哪些关系。

## 解决什么问题？ {#what-problem-does-it-solve}

重复提问的用户通常对答案不满意，而转换话题的用户已经不需要保留的历史。`turn_relation` 让决策可以按其中任何一种情况路由。

## 何时使用 {#when-to-use}

在以下情况使用 `turn_relation`：

- 重复的请求应交给更强的模型；
- 话题转换时应丢弃或重置对话上下文。

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
            previous_turns: 2
        - name: topic_boundary
          description: Whether the live turn still depends on retained history.
          labels: [continue, shift]
          input:
            previous_turns: 8
    decisions:
      - name: retry-stronger
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: turn_relation, name: likely_dissatisfied}
        modelRefs:
          - model: strong-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: turn_relation, name: ...}` 条件使用 |
| `description` | 规则检测的内容 |
| `labels` | 规则匹配的关系，取自 `continue`、`reask`、`shift` 和 `return`；没有 `labels` 的规则本身不匹配任何内容，由它的条件用 `label` 指定一个 |
| `input.previous_turns` | 问题读取当前轮次之前的多少个轮次（用户和助手；默认 4，最多 32） |
| `deployment` | 回答规则的部署；否则为 `turn_relation` 绑定，再否则为决策模型 |
| `timeout_ms` | 答案的截止时间 |

当模型选中的关系属于规则的 `labels` 时，规则匹配。第一个轮次之前没有轮次，因此不提问，也没有规则匹配。目前还没有任何模型有这个问题的训练形式：所有模型都零样本回答。它会把当前轮次和它读取的之前轮次发给部署。完整示例见 [`config/fragments/signal/turn-relation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/turn-relation)。
