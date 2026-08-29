# Conversation Signal

## Overview

`conversation` routes on the structure of a chat, such as message count,
developer instructions, available tools, or an active tool loop. Define these
rules under `routing.signals.conversation`.

This family is heuristic: it inspects the request's `messages[]` and `tools[]` arrays without any model inference.

## Key Advantages

- Routes agentic (tool-heavy) requests to capable models without keyword heuristics.
- Distinguishes single-turn from multi-turn conversations at the structural level.
- Uses an in-memory scan of already-parsed request fields; no model inference is required.
- Produces named signals that projections and decisions can consume like any other family.

## What Problem Does It Solve?

Modern LLM requests vary dramatically in shape. A simple "What is 2+2?" is structurally different from an agentic coding session with developer instructions, three tool definitions, and multiple tool-call cycles. `conversation` turns these structural differences into stable named signals so the decision tree can route each shape to the right model tier.

## When to Use

Use `conversation` when:

- routing depends on conversation depth (single-turn vs multi-turn)
- agentic requests with tool definitions should go to more capable models
- the presence of a developer message changes routing policy
- you need to count tool-call cycles to detect complex agentic workflows

## Configuration

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

## Feature Types

| `feature.type` | Description | Predicate required? |
|---|---|---|
| `count` | Counts matching items. Returns the raw integer. | Yes |
| `exists` | Returns 1.0 if at least one item matches, else 0.0. | No (implicit boolean) |

## Source Types

| `source.type` | Optional `role` | Description |
|---|---|---|
| `message` | `user`, `assistant`, `system`, `developer`, `tool`, `non_user`, or empty (all) | Counts messages, optionally filtered by role. |
| `tool_definition` | — | Counts entries in the request-level `tools[]` array. |
| `tool_choice_required` | — | Returns 1 when the request protocol requires a tool call, including a named tool choice. |
| `tool_choice_none` | — | Returns 1 when the request protocol explicitly forbids tool calls. |
| `assistant_tool_call` | — | Counts `tool_calls` across all assistant messages. |
| `assistant_tool_cycle` | — | Counts `tool` role messages (completed tool results). |
| `active_tool_loop` | — | Returns 1 when the request tail is actively continuing a tool loop: the last assistant message requests a tool, the last message is a tool result, or the latest user turn directly follows a tool result. Older unmatched or completed calls do not keep later turns in the tool loop. |
| `flow_tool_state` | — | Returns 1 only when the request ends with a Router Flow tool result carrying resumable workflow state. Historical Flow tool results do not match. |
| `image_content` | — | Counts image content parts independently of whether the image can be decoded by a local embedding model. |

## Decision Usage

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

## Dependencies and Limitations

The signal inspects the incoming `messages`, `tools`, and tool-choice controls
but does not persist them. `tool_choice` is an execution constraint rather than
tool availability: `required`, a named tool, or Anthropic `any` matches
`tool_choice_required`, while `none` matches `tool_choice_none`. When modern
`tool_choice` and legacy `function_call` are both present, `tool_choice` is
authoritative. These facts describe request shape, not tool safety or user
intent. Apply authorization at the tool boundary. See a complete example:
[`config/fragments/signal/conversation/agentic-shape.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/conversation/agentic-shape.yaml).
