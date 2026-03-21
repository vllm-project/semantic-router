# Elo

## Overview

`elo` is a selection algorithm that ranks candidate models with an Elo-style feedback score.

It aligns to `config/algorithm/selection/elo.yaml`.

## Key Advantages

- Reuses historical feedback instead of only current-request heuristics.
- Makes ranking behavior easy to tune with a small parameter set.
- Supports category-aware weighting for routes with distinct workloads.

## What Problem Does It Solve?

If model quality changes over time, a fixed winner is too rigid. `elo` lets the route prefer candidates that have consistently performed well on similar traffic.

## When to Use

- you collect route-level feedback or quality comparisons
- ranking should improve over time as more comparisons arrive
- one route sees repeatable workloads where a rating system is useful

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: elo
  elo:
    initial_rating: 1200
    k_factor: 32
    category_weighted: true
    min_comparisons: 10
```
