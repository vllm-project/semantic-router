---
sidebar_position: 4
---

# Mappings

## Overview

`routing.projections.mappings` turns a projection score into named routing bands that decisions can consume.

## What Problem Does It Solve?

Scores are useful internal signals, but decision rules should not depend on everyone remembering that "0.82 means reasoning tier" or "0.35 means verification required."

Mappings solve that by turning numeric thresholds into reusable policy names.

This is also the point where a projection becomes decision-visible. Decisions
reference `mapping.outputs[*].name`, not score names or partition names.

## How Mappings Behave at Runtime

Two mapping methods are supported:

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

## Configuration

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
      description: Use the reasoning model for the highest difficulty band.
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

Mappings make no model or storage calls; they transform scores already computed
for the request. Thresholds still inherit the uncertainty and calibration of
their input signals. See a complete end-to-end example in the
[`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml).
