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
| `failed_tool_result` | — | Counts the failed tool results of the current user turn since the last two successful results in a row. See [Escalate when tool calls keep failing](#escalate-when-tool-calls-keep-failing). |

To route on images in the request, use an [`input_modality`](./input-modality)
rule with `modality: image`.

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

## Escalate When Tool Calls Keep Failing

An agent whose tool calls keep failing is often on a model too weak for the
step. A `failed_tool_result` rule lets a recipe send those requests to a
stronger model and bring the session back once the tools work again. Nothing
changes until a recipe declares the rule and a decision routes on it.

```yaml
recipes:
  - name: agent
    signals:
      conversation:
        - name: repeated_tool_failures
          description: Two tool results of the current turn failed.
          feature:
            type: count
            source:
              type: failed_tool_result
          predicate:
            gte: 2
    decisions:
      - name: escalate_on_tool_failures
        priority: 200
        rules:
          type: conversation
          name: repeated_tool_failures
        modelRefs:
          - model: premium-ref
      - name: default_route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: qwen-chat
```

**What counts as a failure.** The Router reads only the failure markers a
client puts in the request, never the wording of an error:

- Anthropic Messages: a `tool_result` block with `is_error: true`. Claude Code
  sets it when a command exits nonzero or a tool reports an error. The cause
  is not read, so a marked read of a missing file counts like a failed build.
- OpenAI Responses and Chat Completions: a tool output has no failure field.
  The one marker the Router reads is the exit code Codex writes above a shell
  command's output, `Exit code: N` or, for `exec_command`, `Process exited with
  code N`. Codex writes that header itself and ends it with an `Output:` line
  that the command's own output follows, so only the lines above the first
  `Output:` line are read and nothing the command prints can pass for a
  status. A nonzero code is a failure; a command still running is not.
- A Chat client without that header never escalates, since its tool messages
  carry no marker at all.

**When it starts and when it returns.** The count runs over the tool results of
the current user turn, in order. A failed result adds one, two successful
results in a row reset it to zero, and a user message that is not sent with
tool results starts a new turn at zero. With `gte: 2`, two failures escalate,
the escalation holds through one success, and the request after the second
success in a row matches the normal decision again. A failure between single
successes keeps counting, so alternating results escalate once and stay there
instead of switching models on every request. The count comes from the
history the client resends, so it needs no session state and gives the same
answer on every replica.

**Prompt cache.** A model switch starts the new model's prompt cache cold. The
first escalated request sends the whole conversation uncached, and so does the
first request after the return. The two-success reset keeps that to one switch
each way per run of failures.

**Session protection.** Router Learning protection keeps a session on its
model, also inside a tool loop, but a change of matched decision releases it,
so the escalation and the return both take effect. Protection still holds the
session on its current model while that model is a candidate of the matched
decision, so list the stronger model only in the escalation decision. If the
normal decision listed it as well, the session would stay on it after the
return.

**Seeing why.** The response names the decision in `x-vsr-selected-decision`
and the model in `x-vsr-selected-model`. The replay record lists the rule
under `signals.conversation` and the count under `signal_values`, as
`conversation:repeated_tool_failures`, and `x-vsr-trace` lists
`conversation:repeated_tool_failures` under `signals` for a caller allowed to
see it.

## Dependencies and Limitations

The signal inspects the incoming `messages`, `tools`, and tool-choice controls
but does not persist them. `tool_choice` is an execution constraint rather than
tool availability: `required`, a named tool, or Anthropic `any` matches
`tool_choice_required`, while `none` matches `tool_choice_none`. When modern
`tool_choice` and legacy `function_call` are both present, `tool_choice` is
authoritative. These facts describe request shape, not tool safety or user
intent. Apply authorization at the tool boundary. See a complete example:
[`config/fragments/signal/conversation/agentic-shape.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/conversation/agentic-shape.yaml).
