---
sidebar_position: 3
---

# Scores

## Overview

`routing.projections.scores` combines matched signal evidence into one continuous numeric value.

## What Problem Does It Solve?

Decisions are built for readable boolean logic. They are not a good place to express "take a little evidence from context length, some from reasoning markers, subtract some weight for very simple requests, and then decide which tier this belongs to."

Scores solve that by giving you one explicit numeric layer between signals and decision policy.

## How Scores Behave at Runtime

Only `method: weighted_sum` is supported.

Each input contributes:

`weight * input_value`

How `input_value` is computed depends on `value_source`:

- omitted or `binary`: use `match` when the signal matched and `miss` when it did not
- `confidence`: use the matched signal confidence, or `0` when the signal did not match
- `raw`: use the raw numeric value from `SignalValues` (e.g., a count or measurement), or `0` when absent

Defaults:

- `match` defaults to `1.0`
- `miss` defaults to `0.0`

Most inputs reference a declared signal under `routing.signals`. The
`kb_metric` and `projection` types instead reference derived runtime state as
described below.

Supported input types include:

- `keyword`
- `embedding`
- `domain`
- `fact_check`
- `user_feedback`
- `reask`
- `preference`
- `language`
- `context`
- `structure`
- `complexity`
- `modality`
- `authz`
- `jailbreak`
- `pii`
- `kb`
- `conversation`
- `event`
- `kb_metric`
- `projection`

For `kb_metric`, `kb` identifies a configured knowledge base, `metric` selects
`best_score`, `best_matched_score`, or a metric declared by that knowledge
base, and `value_source` is `score`. For `projection`, `name` identifies an
earlier score or mapping output.

Scores are internal projection state. Decisions do not reference score names directly; mappings consume them next.

## Configuration

```yaml
document:
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

### Raw value source

When a signal family exposes numeric measurements (counts, distances, token totals) through `SignalValues`, use `value_source: raw` to feed them directly into the weighted sum instead of reducing them to binary or confidence scalars.

```yaml
document:
  projections:
    scores:
      - name: workload_pressure
        method: weighted_sum
        inputs:
          - type: structure
            name: many_questions
            weight: 0.2
            value_source: raw
          - type: structure
            name: nested_depth
            weight: 0.4
            value_source: raw
```

Raw values can differ in scale across signal families. Choose weights carefully or use threshold bands that account for the expected numeric range.

## Config Fields

| Field | Meaning |
|-------|---------|
| `name` | score identifier |
| `method` | currently `weighted_sum` |
| `inputs[].type` | supported signal family, `kb_metric`, or `projection` |
| `inputs[].name` | declared signal name, or an earlier score/mapping output for `projection` |
| `inputs[].kb` / `inputs[].metric` | knowledge-base name and numeric metric for `kb_metric` |
| `inputs[].weight` | contribution multiplier; negative weights lower the score |
| `inputs[].value_source` | `binary`, `confidence`, `raw`, or `score` (for projection inputs); `confidence` on a `projection` input reads a mapping output's calibrated confidence |
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

## Hierarchical Composition

Scores can reference earlier projection scores or mapping output confidences using `type: projection`. This enables layered routing constructs where one score builds on another.

### Score-to-Score Reference

Use `value_source: score` (or omit `value_source`) to read a previously computed score value:

```yaml
document:
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_request_markers
            weight: 0.6
            value_source: confidence

      - name: verification_pressure
        method: weighted_sum
        inputs:
          - type: projection
            name: difficulty_score
            value_source: score
            weight: 0.8
          - type: fact_check
            name: needs_fact_check
            weight: 0.4

    mappings:
      - name: verification_band
        source: verification_pressure
        method: threshold_bands
        outputs:
          - name: needs_deep_verify
            gte: 0.7
          - name: standard_verify
            lt: 0.7
```

### Confidence Reference

Use `value_source: confidence` to read the calibrated confidence from a mapping output band:

```yaml
- type: projection
  name: needs_deep_verify
  value_source: confidence
  weight: 0.5
```

### Dependency Ordering

Scores can be declared in any order. The runtime evaluates them in topological order so that dependencies are always resolved before dependents. Cycles are rejected at config validation time.

Scores make no additional model calls and do not persist content by themselves;
they combine signal results already present for the request. Weights do not
calibrate heterogeneous inputs automatically, so evaluate the complete score
and mapping on labeled traffic. See a complete example in the
[`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml).
