# RL Driven

## Overview

`rl_driven` is a selection algorithm for online exploration and personalization.

It aligns to `config/algorithm/selection/rl-driven.yaml`.

## Key Advantages

- Supports exploration instead of always exploiting the current best model.
- Can personalize routing as more interactions accumulate.
- Keeps online-learning behavior local to one decision.

## What Problem Does It Solve?

If the router should keep learning instead of freezing the current winner, a static selector becomes a bottleneck. `rl_driven` exposes an exploration-based policy for those cases.

## When to Use

- the route should keep exploring candidate models online
- personalization should adapt over time
- the route can tolerate some exploration cost in exchange for better long-term selection

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: rl_driven
  rl_driven:
    exploration_rate: 0.15
    use_thompson_sampling: true
    enable_personalization: true
```
