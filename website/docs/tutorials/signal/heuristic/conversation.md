# Conversation Signal

## Overview

`conversation` routes on chat structure and protocol facts, such as message
count, developer instructions, available tools, explicit tool-use constraints,
or an active tool loop. Define these rules under `recipes[].signals.conversation`.

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
recipes:
  - name: default
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
| `active_tool_loop` | — | Returns 1 when the request tail is actively continuing a tool loop: the last assistant message requests a tool, the last message is a tool result, or the latest user turn directly follows a tool result. A system message inside the conversation, which Claude Code inserts after a tool result or the user's message, is passed over. Older unmatched or completed calls do not keep later turns in the tool loop. |
| `flow_tool_state` | — | Returns 1 only when the request ends with a Router Flow tool result carrying resumable workflow state. Historical Flow tool results do not match. |
| `last_tool_call` | — | Counts the tool calls of the latest assistant turn, the latest run of consecutive assistant messages: every call, the calls named in `tools`, or the calls that match an entry of `calls`. With `match: no_match` it counts the calls that match no entry. |

To route on images in the request, use an [`input_modality`](./input-modality)
rule with `modality: image`.

## Routing on What the Agent Just Did

`last_tool_call` describes the agent's previous turn, so a decision can route each
turn of an agent session by what that turn did. A `calls` entry matches a call
whose name is in `tools` (any name when `tools` is empty) and, when `pattern` is
set, whose `argument` matches it. `argument` names a string field of the call's
JSON arguments, such as Bash's `command` or Codex's `exec_command` `cmd`. When
the arguments are not a JSON object, or that field is missing, null or not a
string, the entry does not match, whatever the pattern. Without an `argument`,
the pattern reads the raw arguments, and an entry without a `pattern` matches on
the name alone. Patterns are RE2 regular expressions. `tools` and `calls` cannot
both be set on one source.

Coding agents read files through shell tools as often as through read tools, so a
read-only turn has to be judged by the command as well as the tool name. The
pair below matches a turn that only read: at least one read call, and no call
that is not a read. The command pattern is a lexical heuristic, not a guarantee
that a command cannot write: it accepts one command from a short list, with no
chaining, pipe, redirection, substitution or line break. It leaves out commands
such as `find`, `sed` and `git diff`, which can write through an option, but some
listed commands can too (`git log --output`, `tree -o`), so it decides which model
serves a turn and is no permission check. In a Responses history each call of a
turn is its own item, and consecutive assistant items count as one turn. Tool
results that follow the calls do not end the turn.

```yaml
recipes:
  - name: default
    signals:
      conversation:
        - name: last_turn_read
          description: The previous assistant turn read files.
          feature:
            type: count
            source:
              type: last_tool_call
              calls:
                - tools: [Read, view_image]
                - tools: [Bash]
                  argument: command
                  pattern: '^[ \t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \t][^;&|<>\n`$]*)?$'
                - tools: [exec_command]
                  argument: cmd
                  pattern: '^[ \t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \t][^;&|<>\n`$]*)?$'
          predicate:
            gte: 1
        - name: last_turn_other
          description: The previous assistant turn made a call that is not a read.
          feature:
            type: count
            source:
              type: last_tool_call
              calls:
                - tools: [Read, view_image]
                - tools: [Bash]
                  argument: command
                  pattern: '^[ \t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \t][^;&|<>\n`$]*)?$'
                - tools: [exec_command]
                  argument: cmd
                  pattern: '^[ \t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \t][^;&|<>\n`$]*)?$'
              match: no_match
          predicate:
            gte: 1
```

A decision whose rule is `last_turn_read AND NOT last_turn_other` then sends the
turn after a read-only turn to a cheaper model. A turn with no call never counts
as a read, and neither does a shell call whose arguments the pattern cannot read.
The source reads only the history the request carries, so after compaction it
judges the latest assistant turn that compaction kept.

## Decision Usage

```yaml
recipes:
  - name: default
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
