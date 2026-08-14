---
sidebar_position: 1
---

# Projections

## Overview

Projections sit between signal extraction and decision matching. They resolve
competition among signals, combine several signals into a score, and turn that
score into named outputs a decision can reference.

The route pipeline is:

1. `routing.signals` extracts facts.
2. `routing.projections` coordinates or derives facts.
3. `routing.decisions` matches policy and chooses candidate models.

## Key Advantages

- Reuses one coordination or threshold policy across several decisions.
- Keeps numeric aggregation out of boolean decision trees.
- Preserves named, explainable outputs for replay and debugging.

## What Problem Does It Solve?

Individual signals are intentionally narrow. Real routing policy often needs
one winner from a competing intent group or one reusable difficulty score from
several weak indicators. Without projections, that logic is duplicated across
decisions and numeric thresholds become hard to audit.

## When to Use

Use projections when:

- only one member of a domain or embedding group should remain active
- several signals should contribute to one continuous score
- several decisions should share the same named threshold bands

Skip projections when one raw signal expresses the route condition clearly or
when multiple matches should remain independently visible.

## Projection Types

| Type | Goal | Decision-visible? | Guide |
|---|---|---|---|
| `partitions` | Keep one winner from competing domain or embedding signals | No; decisions still reference the winning raw signal | [Partitions](./partitions) |
| `scores` | Combine signal values with `weighted_sum` | No; scores feed mappings or other scores | [Scores](./scores) |
| `mappings` | Convert a score into named threshold outputs | Yes, through `type: projection` | [Mappings](./mappings) |
| trace | Explain partition, score, and mapping results in Router Replay | Operational only | [Projection Traces](./traces) |

Current methods are:

- partition semantics: `exclusive` and `softmax_exclusive`
- score method: `weighted_sum`
- mapping methods: `threshold_bands` (first matching output) and `multi_emit`
  (every matching output; requires at least two outputs)
- optional mapping calibration: `sigmoid_distance`

## Configuration

```yaml
routing:
  signals:
    embeddings:
      - name: technical-support
        threshold: 0.75
        candidates: [installation help, troubleshooting]
      - name: account-management
        threshold: 0.72
        candidates: [billing issue, subscription change]
    context:
      - name: long-context
        min_tokens: 4K
        max_tokens: 200K

  projections:
    partitions:
      - name: support-intents
        semantics: exclusive
        members: [technical-support, account-management]
        default: technical-support
    scores:
      - name: request-difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical-support
            value_source: confidence
            weight: 0.5
          - type: context
            name: long-context
            weight: 0.5
    mappings:
      - name: difficulty-band
        source: request-difficulty
        method: threshold_bands
        outputs:
          - name: support-fast
            lt: 0.5
          - name: support-escalated
            gte: 0.5

  decisions:
    - name: escalated-support
      description: Route difficult support requests to the larger model.
      priority: 150
      rules:
        operator: AND
        conditions:
          - type: projection
            name: support-escalated
      modelRefs:
        - model: support-large
```

Only mapping output names are referenced with `type: projection`. Decisions do
not reference partition or score names directly.

The DSL exposes the same three concepts through `PROJECTION partition`,
`PROJECTION score`, and `PROJECTION mapping` blocks. The Dashboard exposes them
under **Config > Projections**.

## Dependencies and Limitations

- Projections make no additional model or storage calls; they consume signal
  results already computed for the request.
- A partition default is a fallback, not evidence that its member matched.
- Weighted sums do not automatically calibrate inputs from different signal
  families. Evaluate weights and mapping bands together on labeled traffic.
- Cycles between derived scores are rejected during validation.
- The schema is defined in
  [`projection_config.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/projection_config.go),
  and the maintained `balance` recipe provides an end-to-end example in
  [`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml).
