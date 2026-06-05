---
sidebar_position: 4
---

# Mappings

## Overview

`routing.projections.mappings` turns a projection score into named routing bands that decisions can consume.

Use mappings when:

- decisions should read semantic labels such as `balance_reasoning` or `verification_required` instead of raw thresholds
- one score should feed several decisions through reusable named outputs
- routing policy review should happen over named bands instead of inline numeric comparisons

## Key Advantages

- Converts numeric scores into readable policy names that decisions consume.
- Centralizes threshold policy in one place instead of duplicating it across routes.
- Lets one score feed many decisions through reusable named outputs.
- Supports optional confidence calibration via `sigmoid_distance`.

## What Problem Does It Solve?

Scores are useful internal signals, but decision rules should not depend on everyone remembering that "0.82 means reasoning tier" or "0.35 means verification required."

Mappings solve that by turning numeric thresholds into reusable policy names.

That gives you two benefits:

- one score can feed many decisions
- threshold policy lives in one place instead of being repeated across routes

This is also the point where a projection becomes decision-visible. In the current implementation, decisions can only reference `mapping.outputs[*].name`, not score names or partition names.

## How Mappings Behave at Runtime

The implementation supports two mapping methods:

- `threshold_bands` (default, also used when `method` is unset) — emits the **first** matching output band.
- `multi_emit` — emits **every** matching output band, so one mapping can set several orthogonal policy tags from the same score. Requires at least two outputs.

Each output declares one or more bounds using:

- `lt`
- `lte`
- `gt`
- `gte`

Important runtime details:

- outputs are checked in declared order
- with `threshold_bands`, the first matching output wins
- with `multi_emit`, every matching output is emitted (in declared order)
- if no output matches, the mapping emits nothing
- optional `calibration` computes a confidence for each emitted projection output

The supported calibration method today is `sigmoid_distance`, which derives confidence from how far the score sits from the nearest threshold boundary.

## Canonical YAML

```yaml
routing:
  projections:
    mappings:
      - name: difficulty_band
        source: difficulty_score
        method: threshold_bands
        calibration:
          method: sigmoid_distance
          slope: 10.0
        outputs:
          - name: balance_simple
            lt: 0.18
          - name: balance_medium
            gte: 0.18
            lt: 0.48
          - name: balance_complex
            gte: 0.48
            lt: 0.82
          - name: balance_reasoning
            gte: 0.82

  decisions:
    - name: reasoning_deep
      priority: 250
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
          - type: projection
            name: balance_reasoning
```

## DSL

```dsl
PROJECTION mapping difficulty_band {
  source: "difficulty_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [
    { name: "balance_simple", lt: 0.18 },
    { name: "balance_medium", gte: 0.18, lt: 0.48 },
    { name: "balance_complex", gte: 0.48, lt: 0.82 },
    { name: "balance_reasoning", gte: 0.82 }
  ]
}

ROUTE reasoning_deep {
  PRIORITY 250
  WHEN domain("math") AND projection("balance_reasoning")
  MODEL "google/gemini-3.1-pro"
}
```

## Config Fields

| Field | Meaning |
|-------|---------|
| `name` | mapping identifier |
| `source` | score name to read from |
| `method` | `threshold_bands` (default) or `multi_emit` |
| `calibration` | optional confidence model for the matched output |
| `outputs[].name` | decision-visible projection name |
| `outputs[].lt/lte/gt/gte` | threshold bounds for that output |

## Dashboard

- `Config -> Projections` edits mappings in canonical config form
- `Config -> Decisions` can reference mapping outputs with condition type `projection`

## Configuration

Mappings are configured under `routing.projections.mappings`. Each mapping requires a `name`, a `source` score, a `method` (`threshold_bands` by default, or `multi_emit`), and a list of `outputs` with threshold bounds. See the [Canonical YAML](#canonical-yaml) and [Config Fields](#config-fields) sections above for full field reference.

## When to Use

Use mappings when:

- several routes should share the same tier names
- you want readable decision rules such as `projection("verification_required")`
- threshold policy should be centralized and auditable

## When Not to Use

Do not use mappings when:

- the decision should reference a raw signal directly
- the score is only diagnostic and not part of routing policy
- you have not first defined the score that this mapping should read from

## Design Notes

- Keep output names policy-oriented so decisions read like routing intent, not threshold math.
- Keep threshold bands monotonic, ordered, and easy to audit.
- Let decisions consume named outputs with `type: projection` rather than repeating numeric thresholds in multiple places.
- With `threshold_bands`, avoid overlapping bands unless you intentionally want order-dependent behavior; the runtime returns the first matching output. Use `multi_emit` when overlapping bands should all fire (e.g. orthogonal policy tags).

## Next Steps

- Read [Scores](./scores) when you need to build the numeric source for a mapping.
- Read [Overview](./overview) for the full `routing.projections` workflow and signal/decision relationship.
