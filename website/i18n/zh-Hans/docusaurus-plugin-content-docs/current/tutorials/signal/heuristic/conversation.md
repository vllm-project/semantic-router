---
translation:
  source_commit: "cf757016"
  source_file: "docs/tutorials/signal/heuristic/conversation.md"
  outdated: false
---

# Conversation 信号

## 概览

`conversation` 检测传入 chat-completion 请求形态的结构事实：用户消息数量、是否存在 developer 消息、定义的工具数量、assistant 工具调用次数、已完成的工具周期，以及当前请求是否仍处于活跃工具循环中。它映射到 `config/signal/conversation/`，并在 `routing.signals.conversation` 下声明。

该信号族为启发式：无需任何模型推理，直接检查请求的 `messages[]` 和 `tools[]` 数组。

## 主要优势

- 无需关键词启发式规则，即可将智能体型（大量使用工具的）请求路由到能力更强的模型。
- 在结构层面区分单轮和多轮会话。
- 零延迟：对已经解析的请求字段进行快速内存扫描即可完成评估。
- 生成命名信号，投影和决策可像处理其他信号族一样使用这些信号。

## 解决什么问题？

现代 LLM 请求的形态差异很大。简单的“2+2 等于多少？”与包含 developer 指令、三个工具定义和多个工具调用周期的智能体编码会话在结构上截然不同。`conversation` 将这些结构差异转换为稳定的命名信号，使决策树能够将每种形态路由到合适的模型层级。

## 何时使用

在以下情况使用 `conversation`：

- 路由取决于会话深度（单轮与多轮）
- 带工具定义的智能体型请求应转到能力更强的模型
- developer 消息是否存在会改变路由策略
- 需要统计工具调用周期以检测复杂的智能体工作流

## 配置

```yaml
routing:
  signals:
    conversation:
      - name: multi_turn_user
        description: At least two user messages.
        feature:
          type: count
          source:
            type: message
            role: user
        predicate:
          gte: 2

      - name: has_developer_message
        description: Request includes a developer message.
        feature:
          type: exists
          source:
            type: message
            role: developer

      - name: tool_heavy
        description: Three or more tool definitions.
        feature:
          type: count
          source:
            type: tool_definition
        predicate:
          gte: 3
```

## 特征类型

| `feature.type` | 描述 | 是否需要 predicate？ |
| --- | --- | --- |
| `count` | 统计匹配项并返回原始整数。 | 是 |
| `exists` | 至少有一个匹配项时返回 1.0，否则返回 0.0。 | 否（隐式布尔值） |

## 来源类型

| `source.type` | 可选 `role` | 描述 |
| --- | --- | --- |
| `message` | `user`、`assistant`、`system`、`developer`、`tool`、`non_user` 或空（全部） | 统计消息，可选择按角色筛选。 |
| `tool_definition` | — | 统计请求级 `tools[]` 数组中的条目。 |
| `assistant_tool_call` | — | 统计所有 assistant 消息中的 `tool_calls`。 |
| `assistant_tool_cycle` | — | 统计 `tool` 角色的消息（已完成的工具结果）。 |
| `active_tool_loop` | — | 最新请求正活跃地继续工具循环时返回 1：最后一条消息是工具结果、最新用户轮次紧接在工具结果之后，或 assistant 工具调用数多于返回的工具结果数。仅有历史上已完成的工具调用不会匹配。 |

## 决策用法

```yaml
routing:
  decisions:
    - name: agentic_routing
      rules:
        operator: AND
        conditions:
          - type: conversation
            name: tool_heavy
          - type: conversation
            name: multi_turn_user
      modelRefs:
        - model: gpt-4o
```
