# Turn Relation Signal

## Overview

`turn_relation` asks the decision model how the current user turn relates to
the conversation before it: it `continue`s the current topic, `reask`s an
earlier turn, `shift`s to an unrelated one, or `return`s to an earlier topic.
It replaces the `reasks` and `topic_continuity` signals.

It maps to config/fragments/signal/turn-relation and is declared under
`routing.signals.turn_relation`.

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

- a repeated request should go to a stronger model
- a topic change should drop or reset conversation context

## Configuration

```yaml
routing:
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

| Field | Meaning |
| --- | --- |
| `name` | The rule's name |
| `description` | What the rule detects |
| `labels` | The relations the rule matches, among `continue`, `reask`, `shift` and `return` |
| `input.previous_turns` | How many earlier user turns the question reads (default 4) |
| `deployment` | The deployment that answers the rule; else `model_bindings.turn_relation`, else the decision model |
| `timeout_ms` | Bound of the question |
