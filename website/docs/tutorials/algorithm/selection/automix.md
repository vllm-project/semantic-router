# Automix

## Overview

`automix` is a selection algorithm for routes that should trade off verification quality, escalation depth, and cost.

It aligns to `config/algorithm/selection/automix.yaml`.

## Key Advantages

- Encodes a cost-versus-quality policy directly in the route.
- Supports bounded escalation instead of unlimited retries.
- Keeps verification behavior local to the decision that needs it.

## What Problem Does It Solve?

Some routes should prefer a cheaper candidate first, but still escalate when verification confidence is too low. `automix` makes that policy explicit instead of hard-coding it in application logic.

## When to Use

- one route has several candidate models with different cost and quality profiles
- escalation should stop after a small number of retries
- the route should stay cost-aware instead of always choosing the strongest model

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
```
