# Router DC

## Overview

`router_dc` is a selection algorithm for semantic query-to-model matching.

It aligns to `config/algorithm/selection/router-dc.yaml`.

## Key Advantages

- Uses semantic similarity instead of only explicit ranking rules.
- Keeps learned selector thresholds visible in config.
- Useful when prompt semantics matter more than static priority.

## What Problem Does It Solve?

Some routes need the chosen model to match the prompt style or task semantics closely. `router_dc` makes that learned semantic selector part of the decision instead of an external hidden step.

## When to Use

- the best candidate depends on semantic similarity between prompt and model profile
- you want a learned selector without full online exploration
- one route should route by semantic fit rather than only cost or latency

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: router_dc
  router_dc:
    temperature: 0.2
    dimension_size: 384
    min_similarity: 0.7
    use_query_contrastive: true
```
