# Latency Aware

## Overview

`latency_aware` is a selection algorithm that prefers the fastest acceptable candidate according to latency percentiles.

It aligns to `config/algorithm/selection/latency-aware.yaml`.

## Key Advantages

- Keeps latency SLOs visible at the route level.
- Balances TTFT and TPOT instead of relying on one metric.
- Useful for routes where responsiveness matters more than absolute quality.

## What Problem Does It Solve?

Some routes need to stay within latency budgets even when several candidate models could answer. `latency_aware` lets the route prefer models that satisfy those budgets.

## When to Use

- the route has multiple viable candidates but strict response-time goals
- TTFT and TPOT should both influence the winner
- latency should be the main tie-breaker after the route matches

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: latency_aware
  latency_aware:
    tpot_percentile: 90
    ttft_percentile: 95
```
