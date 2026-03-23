---
sidebar_position: 3
---

# Scores

## Overview

`routing.projections.scores` combines matched signal evidence into one continuous numeric value.

Use scores when:

- one route depends on several weak signals rather than one decisive detector
- learned and heuristic evidence should contribute to the same routing outcome
- you want numeric aggregation to stay outside the decision layer

## What Problem Does It Solve?

Decisions are built for readable boolean logic. They are not a good place to express "take a little evidence from context length, some from reasoning markers, subtract some weight for very simple requests, and then decide which tier this belongs to."

Scores solve that by giving you one explicit numeric layer between signals and decision policy.

In the `balance` recipe, for example:

- `difficulty_score` blends simplicity, context length, structure, reasoning markers, embeddings, and complexity signals
- `verification_pressure` blends fact-check needs, reference requests, high-stakes domains, correction feedback, and context length

That keeps the weighting story in one place instead of scattering it across many decisions.

## How Scores Behave at Runtime

The current implementation supports `method: weighted_sum` only.

Each input contributes:

`weight * input_value`

How `input_value` is computed depends on `value_source`:

- omitted or `binary`: use `match` when the signal matched and `miss` when it did not
- `confidence`: use the matched signal confidence, or `0` when the signal did not match

Current defaults:

- `match` defaults to `1.0`
- `miss` defaults to `0.0`

The validator requires every input to reference a declared signal under `routing.signals`.

Supported input types currently include:

- `keyword`
- `embedding`
- `domain`
- `fact_check`
- `user_feedback`
- `preference`
- `language`
- `context`
- `structure`
- `complexity`
- `modality`
- `authz`
- `jailbreak`
- `pii`

Scores are internal projection state. Decisions do not reference score names directly; mappings consume them next.

## Canonical YAML

```yaml
routing:
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: simple_request_markers
            weight: -0.28
          - type: context
            name: long_context
            weight: 0.18
          - type: keyword
            name: reasoning_request_markers
            weight: 0.22
            value_source: confidence
          - type: embedding
            name: agentic_workflows
            weight: 0.18
            value_source: confidence
          - type: complexity
            name: general_reasoning:hard
            weight: 0.22
```

## DSL

```dsl
PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [
    { type: "keyword", name: "simple_request_markers", weight: -0.28 },
    { type: "context", name: "long_context", weight: 0.18 },
    { type: "keyword", name: "reasoning_request_markers", weight: 0.22, value_source: "confidence" },
    { type: "embedding", name: "agentic_workflows", weight: 0.18, value_source: "confidence" },
    { type: "complexity", name: "general_reasoning:hard", weight: 0.22 }
  ]
}
```

## Config Fields

| Field | Meaning |
|-------|---------|
| `name` | score identifier |
| `method` | currently `weighted_sum` |
| `inputs[].type` | signal family to read from |
| `inputs[].name` | declared signal name |
| `inputs[].weight` | contribution multiplier; negative weights lower the score |
| `inputs[].value_source` | `binary` or `confidence` behavior |
| `inputs[].match` / `inputs[].miss` | explicit values for binary mode |

## When to Use

Use scores when:

- several weak indicators should combine into one difficulty or escalation signal
- the same weighted story should be reused by more than one route
- you want one central place to tune routing sensitivity

## When Not to Use

Do not use scores when:

- one raw signal already decides the route cleanly
- the rule can stay readable as ordinary boolean logic
- you need a decision-visible output name immediately; scores still need a mapping

## Design Notes

- Keep score names stable because `routing.projections.mappings[*].source` depends on them.
- Document why each weight exists, especially when mixing confidence-bearing learned signals with heuristic signals.
- Prefer scores for numeric aggregation and keep `routing.decisions` focused on readable boolean composition.
- Use negative weights when a matched signal should actively lower the tier, as `balance` does for obviously simple requests.

## Next Steps

- Read [Mappings](./mappings) to turn a score into named routing bands for decisions.
- Read [Overview](./overview) for the full projection workflow and signal/decision relationship.
