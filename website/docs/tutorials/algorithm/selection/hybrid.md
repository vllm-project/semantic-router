# Hybrid

## Overview

`hybrid` is a selection algorithm that combines several ranking signals into one weighted score.

It aligns to `config/algorithm/selection/hybrid.yaml`.

## Key Advantages

- Blends multiple selectors instead of committing to only one.
- Makes weighting explicit and easy to audit.
- Supports gradual migration between ranking policies.

## What Problem Does It Solve?

No single selector is always enough. `hybrid` lets one route combine feedback, semantic matching, and cost-aware routing without duplicating combination logic elsewhere.

## When to Use

- one route should combine several ranking signals
- you want a weighted transition between older and newer selectors
- the final choice should reflect both quality and operational cost

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: hybrid
  hybrid:
    elo_weight: 0.4
    router_dc_weight: 0.4
    automix_weight: 0.2
    cost_weight: 0.1
```
