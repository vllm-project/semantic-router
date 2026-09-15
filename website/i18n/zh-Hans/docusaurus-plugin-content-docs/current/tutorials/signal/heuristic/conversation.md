---
translation:
  source_commit: "9fbd7d85183341c61c25dfc526160eb0599434f8"
  source_file: "docs/tutorials/signal/heuristic/conversation.md"
  outdated: false
---

# 会话信号 {#conversation-signal}

## 概览 {#overview}

`conversation` 按聊天结构与协议事实路由，例如消息数量、开发者指令、可用工具、显式工具使用约束，或正在进行的工具循环。在 `routing.signals.conversation` 下定义这些规则。

该族为启发式：检查请求的 `messages[]` 与 `tools[]` 数组，不做任何模型推理。

## 主要优势 {#key-advantages}

- 无需关键词启发式，即可把智能体式（工具密集）请求路由到有能力的模型。
- 在结构层面区分单轮与多轮对话。
- 对已解析的请求字段做内存扫描；不需要模型推理。
- 产生命名信号，投影与决策可像其他族一样消费。

## 解决什么问题？ {#what-problem-does-it-solve}

现代 LLM 请求的形态差异很大。简单的「2+2 等于几？」与带开发者指令、三个工具定义、多轮工具调用循环的智能体编程会话，结构完全不同。`conversation` 把这些结构差异变成稳定命名信号，使决策树能把每种形态路由到合适的模型档。

## 何时使用 {#when-to-use}

在以下情况使用 `conversation`：

- 路由依赖对话深度（单轮 vs 多轮）
- 带工具定义的智能体请求应走向更强模型
- 开发者消息的存在会改变路由策略
- 需要统计工具调用循环以检测复杂智能体工作流

## 配置 {#configuration}

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

## 特征类型 {#feature-types}

| `feature.type` | 说明 | 是否需要谓词？ |
|---|---|---|
| `count` | 统计匹配项。返回原始整数。 | 是 |
| `exists` | 至少一项匹配则返回 1.0，否则 0.0。 | 否（隐式布尔） |

## 源类型 {#source-types}

| `source.type` | 可选 `role` | 说明 |
|---|---|---|
| `message` | `user`、`assistant`、`system`、`developer`、`tool`、`non_user`，或空（全部） | 统计消息，可按角色过滤。 |
| `tool_definition` | — | 统计请求级 `tools[]` 数组中的条目。 |
| `tool_choice_required` | — | 请求协议要求工具调用（含命名工具选择）时返回 1。 |
| `tool_choice_none` | — | 请求协议显式禁止工具调用时返回 1。 |
| `assistant_tool_call` | — | 统计所有 assistant 消息中的 `tool_calls`。 |
| `assistant_tool_cycle` | — | 统计 `tool` 角色消息（已完成的工具结果）。 |
| `active_tool_loop` | — | 请求尾部仍在继续工具循环时返回 1：最后一条 assistant 消息请求工具、最后一条消息是工具结果，或最近用户轮次紧跟工具结果。更早未匹配或已完成的调用不会让后续轮次保持在工具循环中。 |
| `flow_tool_state` | — | 仅当请求以携带可恢复工作流状态的 Router Flow 工具结果结尾时返回 1。历史 Flow 工具结果不匹配。 |
| `image_content` | — | 独立统计图像内容部分，无论本地嵌入模型能否解码该图像。 |

## 决策用法 {#decision-usage}

```yaml
routing:
  decisions:
    - name: agentic_routing
      description: Send tool-heavy chats to an agent-capable model.
      priority: 100
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

## 依赖与限制 {#dependencies-and-limitations}

该信号检查传入的 `messages`、`tools` 与工具选择控件，但不会持久化它们。`tool_choice` 是执行约束，不是工具可用性：`required`、命名工具或 Anthropic `any` 匹配 `tool_choice_required`，而 `none` 匹配 `tool_choice_none`。现代 `tool_choice` 与旧版 `function_call` 同时存在时，以 `tool_choice` 为准。这些事实描述请求形态，不是工具安全或用户意图。请在工具边界做授权。完整示例见：
[`config/fragments/signal/conversation/agentic-shape.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/conversation/agentic-shape.yaml)。
