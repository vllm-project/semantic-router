---
sidebar_position: 1
---

# Projections

## Overview

Projections sit between signal extraction and decision matching. They resolve
competition among signals, combine several signals into a score, and turn that
score into named outputs a decision can reference.

The route pipeline is:

1. `recipes[].signals` extracts facts.
2. `recipes[].projections` coordinates or derives facts.
3. `recipes[].decisions` matches policy and chooses candidate models.

## Key Advantages

- Reuses one coordination or threshold policy across several decisions.
- Keeps numeric aggregation out of boolean decision trees.
- Preserves named, explainable outputs for replay and debugging.

## What Problem Does It Solve?

Individual signals are intentionally narrow. Real routing policy often needs
one winner from a competing domain group or one reusable difficulty score from
several weak indicators. Without projections, that logic is duplicated across
decisions and numeric thresholds become hard to audit.

## When to Use

Use projections when:

- only one member of a domain group should remain active
- several signals should contribute to one continuous score
- several decisions should share the same named threshold bands

Skip projections when one raw signal expresses the route condition clearly or
when multiple matches should remain independently visible.

## Projection Types

| Type | Goal | Decision-visible? | Guide |
|---|---|---|---|
| `partitions` | Keep one winner from competing domain signals | No; decisions still reference the winning raw signal | [Partitions](./partitions) |
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
recipes:
  - name: default
    signals:
      domains:
        - name: computing
          labels: [computer science]
        - name: business
      context:
        - name: long-context
          min_tokens: 4K
          max_tokens: 200K
    projections:
      partitions:
        - name: subject-areas
          semantics: exclusive
          members: [computing, business]
          default: computing
      scores:
        - name: request-difficulty
          method: weighted_sum
          inputs:
            - type: domain
              name: computing
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
            - name: difficulty-low
              lt: 0.5
            - name: difficulty-high
              gte: 0.5
    decisions:
      - name: escalated-computing
        description: Route difficult computing requests to the larger model.
        priority: 150
        rules:
          operator: AND
          conditions:
            - type: projection
              name: difficulty-high
        modelRefs:
          - model: large-model
```

Only mapping output names are referenced with `type: projection`. Decisions do
not reference partition or score names directly.

The DSL exposes the same three concepts through `PROJECTION partition`,
`PROJECTION score`, and `PROJECTION mapping` blocks. The Dashboard exposes them
under **Routing → Projections**.

## Route by image quality and layout

Opt in by referencing `image_features` in a recipe's projection score with
`value_source: raw`. Both indicators range from 0 to 1:

- `blur`: higher means lower Laplacian variance, which can indicate blur or
  simply a smooth image.
- `edge_density`: higher means larger average grayscale changes between
  adjacent pixels, which can indicate dense text or chart structure.

These are heuristics, not calibrated quality scores. The router takes the
maximum of each indicator independently across readable images. It considers
the first four top-level image parts from user messages in the current turn.
System and developer messages do not end the turn; tool results and earlier
turns do not contribute images. It accepts only inline base64 PNG or JPEG and
never fetches URLs. After base64 decoding, each image is limited to 2 MiB of
image data and 4,194,304 pixels, with dimensions of at least 3 × 3. The
analysis reduces the long side to at most 384 pixels.
Unreadable images do not contribute. If none can be scored, the projection is
unknown; a lower-priority image-presence decision can provide the fallback.

For example, put this in a recipe whose strategy is `priority`, using model
aliases that accept images. Tune the illustrative thresholds to your traffic.

```yaml
signals:
  input_modality:
    - {name: has-image, modality: image}
projections:
  scores:
    - name: blur-score
      method: weighted_sum
      inputs:
        - {type: image_features, name: blur, value_source: raw, weight: 1}
    - name: layout-score
      method: weighted_sum
      inputs:
        - {type: image_features, name: edge_density, value_source: raw, weight: 1}
  mappings:
    - name: blur-band
      source: blur-score
      outputs:
        - {name: blurry, gte: 0.5}
    - name: layout-band
      source: layout-score
      outputs:
        - {name: dense, gte: 0.5}
        - {name: simple, lt: 0.5}
decisions:
  - name: difficult-image
    priority: 30
    rules:
      operator: OR
      conditions:
        - {type: projection, name: blurry, on_unknown: no_match}
        - {type: projection, name: dense, on_unknown: no_match}
    modelRefs: [{model: strong-vision}]
  - name: simple-image
    priority: 20
    rules: {type: projection, name: simple, on_unknown: no_match}
    modelRefs: [{model: standard-vision}]
  - name: image-fallback
    priority: 10
    rules: {type: input_modality, name: has-image}
    modelRefs: [{model: fallback-vision}]
```

## Dependencies and Limitations

- Projections make no additional model or storage calls; they consume signal
  results already computed for the request.
- A partition default is a fallback, not evidence that its member matched.
- Weighted sums do not automatically calibrate inputs from different signal
  families. Evaluate weights and mapping bands together on labeled traffic.
- Cycles between derived scores are rejected during validation.
- See the
  [`feedback` recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/feedback/config.yaml)
  for an end-to-end example of scores and mappings, and
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  for a partition.
