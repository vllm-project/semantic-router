# Multi Factor

## Overview

`multi_factor` ranks candidates by a configurable combination of quality,
latency, cost, and load, then rejects any candidate that violates a hard limit.

The configuration belongs to the decision that declares it. Each matched
decision uses its own weights, limits, percentile, and no-candidate policy.

## Key Advantages

- Single-decision SLO-aware routing without orchestrating multiple selectors.
- Each factor has an explicit source: quality from `quality_score`, latency from
  observed percentiles, cost from pricing, and load from in-flight requests.
- Min-max normalization makes the configured weights comparable across the
  candidate set.
- No model state to train. No external service required.
- Hard SLO ceilings (TPOT, TTFT, cost, in-flight) prune unsafe candidates before scoring.

## What Problem Does It Solve?

Real routes often contain a faster, cheaper model and a slower, stronger one.
The right choice depends on current observations and policy, not one fixed
metric. `multi_factor` expresses that trade-off in one decision and can remove
candidates that exceed configured limits.

## When to Use

- A decision has two or more candidates that differ in quality, latency, cost,
  or load.
- You want to enforce a latency ceiling when observations are available,
  without writing a separate decision.
- Quality, latency, cost, and load all matter and no single one dominates.

## Sibling Algorithms

- `latency_aware` is a special case of this — latency-only scoring. Use it when the other dimensions truly do not matter.
- `hybrid` combines Elo, Router-DC, and AutoMix selector scores before applying
  a cost adjustment. `multi_factor` scores configured quality and pricing plus
  observed latency and load directly.

## Algorithm Principle

For each candidate model $m$ in the candidate set, after SLO filtering:

$$\text{score}(m) = w_Q \cdot \hat{Q}(m) + w_L \cdot (1 - \hat{T}(m)) + w_C \cdot (1 - \hat{C}(m)) + w_{\text{load}} \cdot (1 - \hat{N}(m))$$

Where:

- $\hat{Q}(m)$, $\hat{T}(m)$, $\hat{C}(m)$, $\hat{N}(m)$ are quality / latency / cost / load values **min-max normalized to [0, 1] across the surviving candidate set**.
- Latency, cost, and load are inverted (`1 - ...`) because lower-is-better.
- Quality is direct because higher-is-better.
- Weights are normalized to sum to 1 (negative weights clamped to zero). Equal weights are the recoverable default.

## SLO Filtering

Before scoring, any candidate that exceeds a non-zero ceiling is removed:

- `max_tpot_ms` — p95 (or configured) TPOT observed via `pkg/latency`
- `max_ttft_ms` — p95 (or configured) TTFT observed via `pkg/latency`
- `max_cost_per_1m` — configured prompt pricing
- `max_inflight` — current in-flight request count from `pkg/inflight`

If all candidates are filtered out, behavior is controlled by `on_no_candidates`:

| Value | Behavior |
|---|---|
| `cheapest` (default) | Return the candidate with the lowest configured `providers.models[].pricing.input_cost_per_million_tokens` |
| `first` | Return the first candidate as listed |
| `fail` | Return an error to the caller |

## Configuration

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
    slo:
      max_tpot_ms: 200       # optional, omit for no ceiling
      max_ttft_ms: 800       # optional
      max_cost_per_1m: 5.0   # optional, USD per 1M prompt tokens
      max_inflight: 50       # optional
    latency_percentile: 95   # which percentile to read (default 95)
    on_no_candidates: cheapest
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights.quality` | float | `0.25` | Weight for `quality_score` configured per model |
| `weights.latency` | float | `0.25` | Weight for percentile latency (lower-is-better, inverted) |
| `weights.cost` | float | `0.25` | Weight for prompt pricing (lower-is-better, inverted) |
| `weights.load` | float | `0.25` | Weight for in-flight request count (lower-is-better, inverted) |
| `slo.max_tpot_ms` | float | `0` (off) | Hard ceiling for p95 TPOT in milliseconds |
| `slo.max_ttft_ms` | float | `0` (off) | Hard ceiling for p95 TTFT in milliseconds |
| `slo.max_cost_per_1m` | float | `0` (off) | Hard ceiling for prompt cost per 1M tokens |
| `slo.max_inflight` | int | `0` (off) | Hard ceiling for concurrent in-flight requests |
| `latency_percentile` | int | `95` | Percentile read from `pkg/latency` (1-100) |
| `on_no_candidates` | string | `cheapest` | Fallback policy when SLO filters everything: `cheapest`, `first`, `fail` |

## Known Limitations

- Quality scoring depends on `quality_score` being configured per model. Models without it contribute zero to the quality signal.
- Min-max normalization is **per-request across the candidate set**, so absolute scale of any signal does not matter — but if all candidates have the same value on a dimension, that dimension contributes 0.5 (neutral).
- Load and latency are observed per Router process, not across the whole
  cluster. Replicas can therefore choose different candidates.
- A missing quality, latency, or cost observation contributes `0` for that
  scoring dimension. It does not trigger an SLO exclusion, because absence is
  not evidence that the configured ceiling was exceeded.

See a complete example:
[`config/fragments/algorithm/selection/multi-factor.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/multi-factor.yaml).
