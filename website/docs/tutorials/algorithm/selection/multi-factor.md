# Multi Factor

## Overview

`multi_factor` is a selection algorithm that composes four raw runtime signals — **quality**, **latency**, **cost**, and **load** — into a single weighted score per candidate, with optional SLO hard ceilings that prune candidates before scoring.

It aligns to `config/algorithm/selection/multi-factor.yaml` and addresses issue [#37](https://github.com/vllm-project/semantic-router/issues/37).

## Key Advantages

- Single-decision SLO-aware routing without orchestrating multiple selectors.
- Each signal is a live source: quality from `quality_score` config, latency from `pkg/latency` percentiles, cost from pricing, load from `pkg/inflight`.
- Min-max normalization across the candidate set means weights have intuitive meaning regardless of absolute signal scales.
- No model state to train. No external service required.
- Hard SLO ceilings (TPOT, TTFT, cost, in-flight) prune unsafe candidates before scoring.

## What Problem Does It Solve?

Real routes care about more than one dimension at once: a faster cheaper model and a slower better model both exist in the same candidate pool, and the "right" answer depends on current load and SLO targets, not just the static config. Existing single-signal selectors (`latency_aware`, cost-only routing, quality-only routing) force a hard choice. `multi_factor` lets one decision rule express a smooth tradeoff across all four dimensions, with optional hard SLO ceilings to fence off unsafe candidates.

## When to Use

- A decision has 2+ candidate models that differ along multiple dimensions (e.g. a faster cheaper model and a slower better model) and you want a smooth tradeoff knob.
- You want SLO enforcement (e.g. "never route to a model with p95 TPOT > 200ms") without writing a separate decision rule.
- Quality, latency, cost, and load all matter and no single one dominates.

## Sibling Algorithms

- `latency_aware` is a special case of this — latency-only scoring. Use it when the other dimensions truly do not matter.
- `hybrid` composes request-time selectors and read-only learning evidence into
  one score. `multi_factor` composes raw runtime signals directly. Both are
  useful and complementary.

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
| `cheapest` (default) | Return the candidate with the lowest configured `prompt_per_1m` |
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
- Load uses an in-process tracker (`pkg/inflight`), so in multi-replica deployments each replica sees only its own load, not cluster-wide. Acceptable for the typical sidecar deployment; an external state store could be wired later for true cluster-wide load awareness.
- The in-flight tracker self-heals via TTL eviction (default 10 minutes) to recover from missed `End` calls, but cannot detect actively-running long requests beyond that window — they will appear "free" to the selector. Tune via `pkg/inflight.SetMaxAge` if your workloads routinely exceed 10 minutes per request.
