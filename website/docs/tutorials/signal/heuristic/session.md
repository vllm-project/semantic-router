# Session Signal

## Overview

`session` exposes runtime-derived multi-turn facts as named routing signals under `routing.signals.session`.

It maps to `config/signal/session/` and is declared as part of the normal signal catalog, so decisions and projections can reference session facts without reviving the removed `SESSION_STATE` public surface.

## Key Advantages

- Lets decisions reference multi-turn state through named reusable signals.
- Keeps session-aware routing inside the same config graph as other signals.
- Supports numeric predicates over runtime facts such as turn depth or cache warmth.
- Allows model-specific continuity rules using `previous_model` and `candidate_model`.

## What Problem Does It Solve?

A multi-turn route often depends on facts that are not visible in the raw prompt: whether the request belongs to an existing session, which model served the last turn, and whether switching models is likely to be expensive.

`session` solves that by exposing those runtime-derived facts as normal routing signals, so routes can compose them with domain, complexity, safety, or projection logic.

## When to Use

Use `session` when:

- a route behaves differently on the first turn vs. continuation turns
- you want continuity rules tied to the previous or candidate model
- replay-backed routing logic needs explicit named signal references
- decisions should combine session facts with domain or projection conditions

## Configuration

Source fragment family: `config/signal/session/`

```yaml
routing:
  signals:
    session:
      - name: session_present
        description: Requests that belong to an existing multi-turn conversation.
        fact: session_present
        predicate:
          gte: 1
      - name: warm_cache_continuation
        description: Prefer staying on the warmed model when the same conversation continues.
        fact: cache_warmth
        previous_model: qwen3-8b
        predicate:
          gte: 0.6
      - name: expensive_handoff
        description: Detect costly mid-session upgrades into the premium coding model.
        fact: handoff_penalty
        intent_or_domain: computer science
        previous_model: qwen3-8b
        candidate_model: qwen3-32b
        predicate:
          gte: 0.15
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | yes | Signal name referenced from decisions and projections |
| `fact` | yes | Runtime-derived fact name injected by the router |
| `predicate` | no | Numeric threshold predicate over the fact value |
| `intent_or_domain` | no | Optional domain/task-family guard |
| `previous_model` | no | Only match when the previous turn used this model |
| `candidate_model` | no | Only match when evaluating this candidate model |
| `description` | no | Human-readable explanation |

## Runtime Behavior

The router computes session signal values from request/session context and stores them alongside the normal signal confidence/value map. Decisions can then reference them with `type: session` just like other signal families.

## Known Limitations

- `session` is a routing surface, not a general-purpose persisted DSL state machine.
- Facts are runtime-derived and router-owned; unsupported fact names will fail config validation.
- Session signals help describe continuity conditions, but the actual stay-versus-switch scoring lives in `algorithm.session_aware`.
