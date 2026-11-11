# Turn Relation Signal

## Overview

`turn_relation` asks the decision model how the current user turn relates to the conversation before it.
It asks the `turn_relation`
[preset](../../../system-one/reference/presets#preset-catalog): the turn
`continue`s the current topic, `reask`s for what an earlier turn asked,
`shift`s to an unrelated topic, or `return`s to an earlier one. It replaces
the `reasks` and `topic_continuity` signals.

## Key Advantages

- One question covers repeats, follow-ups and topic changes.
- `input.previous_turns` bounds how much history the question reads.
- Labels select which relations a rule matches.

## What Problem Does It Solve?

A user who repeats a question is usually unhappy with the answer, and a user
who changes topic no longer needs the retained history. `turn_relation` lets a
decision route on either.

## When to Use

Use `turn_relation` when:

- a repeated request should go to a stronger model;
- a topic change should drop or reset conversation context.

## Configuration

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

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: turn_relation, name: ...}` conditions use |
| `description` | What the rule detects |
| `labels` | The relations the rule matches, among `continue`, `reask`, `shift` and `return`; a rule without `labels` matches nothing itself, and its conditions name one with `label` |
| `input.previous_turns` | How many turns before the current one the question reads, user and assistant (default 4, at most 32) |
| `deployment` | The deployment that answers the rule; else the `turn_relation` binding, else the decision model |
| `timeout_ms` | The answer's deadline |

A rule matches when the relation the model chooses is one of its `labels`.
A first turn has no turns before it, so it asks no question and no rule
matches. The question has no trained form on any model yet: every model answers it
zero-shot. It sends the current turn and the previous turns it reads to the
deployment. See complete examples under
[`config/fragments/signal/turn-relation/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/signal/turn-relation).
