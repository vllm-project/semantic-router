# Session metric signal

## Overview

`session_metric` is a unified routing signal family for **session-context numeric inputs**: either **state-backed** values (from `SESSION_STATE` scalars with optional normalization) or **lookup-backed** values (from named tables using keys built from `SESSION_STATE` paths).

It maps to `config/signal/session-metric/` and is declared under `routing.signals.session_metrics`.

## Key Advantages

- One signal type covers cumulative cost style metrics and table-backed penalties (for example handoff cost).
- Keeps session-aware numbers in the same projection and decision pipeline as other signals.
- Clear `kind` field separates pure state reads from resolver-backed lookups.
- Aligns authoring with `SESSION_STATE` as the contract for cross-turn fields.

## What Problem Does It Solve?

Session-scoped scalars and discrete table lookups are not keyword or embedding signals. Without `session_metric`, those numbers would need ad hoc wiring or duplicate signal types in config and the decision engine.

## When to Use

Use `session_metric` when:

- a projection score or decision should weight cumulative cost or similar numeric session fields (`kind: state`)
- a score depends on keys from session state and an external lookup table (`kind: lookup`)

## Configuration

Each rule sets `kind` to `state` or `lookup` (or omit `kind` and set `state` or `table` so the compiler can infer).

Source fragment family: `config/signal/session-metric/`

### State kind

```yaml
routing:
  signals:
    session_metrics:
      - name: session_cost_pressure
        kind: state
        state: session_routing.cumulative_cost_usd
        normalize: minmax
        min: 0
        max: 10
```

### Lookup kind

```yaml
routing:
  signals:
    session_metrics:
      - name: handoff_penalty
        kind: lookup
        table: handoff_penalties
        key:
          - session_routing.current_model
          - session_routing.candidate_model
```

## Decision engine

Decision leaves use the unified type `session_metric` with the rule `name`, for example `session_metric:session_cost_pressure`. For backward compatibility, leaves that still use `session:` or `lookup:` with the same rule names continue to match the same hydrated values.

## Runtime

Lookup rows require a pluggable `LookupResolver` at request time; state rules read numeric snapshots keyed by the same dotted paths as `state` (see `classification.SignalSessionContext.Scalars`).
