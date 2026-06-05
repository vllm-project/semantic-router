# Conversation Signal

## Overview

`conversation` detects structural facts about the shape of the incoming chat-completion request: how many user messages, whether a developer message is present, how many tools are defined, assistant tool-call counts, and completed tool cycles. It maps to `config/signal/conversation/` and is declared under `routing.signals.conversation`.

This family is heuristic: it inspects the request's `messages[]` and `tools[]` arrays without any model inference.

## Key Advantages

- Routes agentic (tool-heavy) requests to capable models without keyword heuristics.
- Distinguishes single-turn from multi-turn conversations at the structural level.
- Zero latency: evaluation is a fast in-memory scan of already-parsed request fields.
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
| `assistant_tool_call` | — | Counts `tool_calls` across all assistant messages. |
| `assistant_tool_cycle` | — | Counts `tool` role messages (completed tool results). |

## Decision Usage

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
