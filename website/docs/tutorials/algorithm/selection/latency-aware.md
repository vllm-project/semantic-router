# Latency-Aware Selection

## Overview

To choose the fastest eligible candidate, weight Multi Factor on latency
alone. [Multi Factor](./multi-factor) then ranks candidates by their observed
latency at the configured percentile and selects the fastest. This replaces the removed
`latency_aware` selector.

## Key Advantages

- Compares candidates using the latency percentile that matters to the route.
- No model state to manage: purely data-driven from runtime metrics.
- Adds SLO ceilings, cost or load later without changing the algorithm type.

## What Problem Does It Solve?

Some routes care more about responsiveness than absolute model quality. A
latency-only `multi_factor` compares already eligible candidates using observed
latency statistics instead of static assumptions.

## When to Use

- The route has multiple viable candidates and latency should determine the
  winner.
- You want fast responses without hard-coding one model.

## Configuration

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights:
      latency: 1
    latency_percentile: 90
```

`latency_percentile` picks the percentile of the observed latency each
candidate is compared at. By default `multi_factor` compares time per output
token (TPOT), and a candidate's time to first token (TTFT) when it has no TPOT
samples; set `latency_metric: ttft` or `tpot` to compare one measurement.
Latency is observed on successful responses only, so a model that fails fast
does not look fast. Latency observations are local to a Router process; they
are not a cluster-wide scheduler. See [Multi Factor](./multi-factor) for SLO
ceilings and the other factors.

### Migrating from latency_aware

`vllm-sr config migrate` rewrites `latency_aware` as this `multi_factor`
weighting. A block that set one percentile keeps its metric
(`latency_metric`) and percentile. A block that set both `tpot_percentile` and
`ttft_percentile` compares TPOT at the TPOT percentile, with a note to review:
`multi_factor` compares one measurement where `latency_aware` averaged both.

Reusable fragment: [`config/fragments/algorithm/selection/latency-aware.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/latency-aware.yaml).
