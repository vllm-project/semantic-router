# Reask Signal

## Overview

`reask` detects when the current user turn semantically repeats the most recent user turns in the same conversation. It maps to `config/signal/reask/` and is declared under `routing.signals.reasks`.

This family is learned: it uses the router's shared semantic embedding path to compare the current user turn against prior user turns.

## Key Advantages

- Captures implicit dissatisfaction without requiring explicit phrases like "this is wrong".
- Distinguishes a one-turn repeat from a repeated multi-turn dissatisfaction streak.
- Lets decisions escalate based on recent conversation history instead of a single message.
- Reuses the existing semantic similarity stack instead of introducing a second model surface.

## What Problem Does It Solve?

Users often restate the same question when the previous answer was not useful. A single-turn classifier can miss that pattern because the complaint is implicit rather than explicit.

`reask` solves that by comparing the latest user turn to the most recent user turns and surfacing configurable dissatisfaction signals when the streak stays semantically similar.

## When to Use

Use `reask` when:

- repeated questions should escalate to a stronger model
- you want different handling for one repeated ask vs multiple repeated asks
- explicit feedback is sparse, but repeated user turns still matter
- routing decisions should depend on same-conversation user history

## Configuration

Source fragment family: `config/signal/reask/`

```yaml
routing:
  signals:
    reasks:
      - name: likely_dissatisfied
        description: Current user turn closely repeats the immediately previous user turn.
        threshold: 0.8
        lookback_turns: 1
      - name: persistently_dissatisfied
        description: Current user turn repeats the last two user turns in a row.
        threshold: 0.8
        lookback_turns: 2
```

Each rule compares the current user turn to the latest `lookback_turns` prior user turns. A rule matches only when every turn in that recent streak stays above the configured similarity threshold.
