# Multi Factor

## Overview

`multi_factor` chooses one candidate from quality, latency, cost, and load. Hard
eligibility rules run first; the surviving candidates are then compared with a
weighted or lexicographic objective.

| Factor | Source | Direction |
| --- | --- | --- |
| Quality | Versioned Overall, capability, or operator index | Higher is better |
| Latency | Observed TPOT, falling back to TTFT, at the selected percentile | Lower is better |
| Cost | Input/output pricing applied to this request's token budget | Lower is better |
| Load | Current in-flight requests in this Router process | Lower is better |

Quality is resolved for the candidate's exact reasoning effort. A score from a
different effort is never borrowed. Coverage is not part of the objective; when
both the final objective value and intelligence score tie, higher coverage is
the deterministic tie-breaker.

## What Problem Does It Solve?

A model pool often contains a stronger model, a cheaper model, and a faster
model. This selector keeps eligibility explicit and makes that trade-off one
auditable decision policy.

## When to Use

Use `multi_factor` when a decision has at least two candidate models and must
optimize a quality/latency/cost/load trade-off. Use `static` when order alone is
the policy, or `latency_aware` when latency is the only selection signal.

## Configuration

### Balanced

The default `weighted` strategy min-max normalizes each available factor over
the surviving pool and computes:

$$
S(m)=w_Q\hat Q(m)+w_T(1-\hat T(m))+w_C(1-\hat C(m))+w_L(1-\hat L(m))
$$

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    objective:
      strategy: weighted
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
      min_coverage: 1.0
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

Weights are normalized to sum to one. If every configured weight is zero, the
selector recovers to equal weights.

### Accuracy-first

`lexicographic` applies priorities in order. Each stage keeps candidates within
the declared relative tolerance of the best observed value, then passes that
band to the next stage.

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    objective:
      strategy: lexicographic
      priorities:
        - {factor: quality, tolerance: 0.03}
        - {factor: cost, tolerance: 0.05}
        - {factor: latency, tolerance: 0.05}
    quality:
      index: vllm-sr/intelligence@1.0.0
      on_missing: exclude
      min_coverage: 1.0
```

This keeps models within 3% of the best quality score, then chooses the cheaper
and faster candidate inside that band.

### Cost-first with a quality floor

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/intelligence@1.0.0
      on_missing: exclude
      min_coverage: 1.0
      min_score: 65
    objective:
      strategy: lexicographic
      priorities:
        - {factor: cost, tolerance: 0.05}
        - {factor: quality, tolerance: 0.03}
        - {factor: latency, tolerance: 0.05}
```

The quality floor excludes models below 65 before optimization. The objective
then keeps the models within 5% of the cheapest estimate and chooses the best
quality inside that cost band.

`weights` cannot be combined with a lexicographic objective. A priority factor
may appear only once.

## Quality eligibility and missing data

```yaml
quality:
  index: acme/clinical-quality@1.0.0
  on_missing: exclude
  min_coverage: 0.5
  min_score: 60
```

- `index` selects a versioned built-in or operator index.
- `min_coverage` applies after the index's own missing-data policy. It can make a
  route stricter than the index definition.
- `min_score` is a hard floor on the index's declared scale and requires
  `on_missing: exclude`.
- `exclude` removes a candidate without qualifying exact-effort evidence.
- `disable_quality` keeps the pool, but one missing candidate disables quality
  for the entire comparison. It never changes weights for only one model.
- Selection diagnostics report the chosen intelligence score and coverage, or
  mark intelligence evidence unavailable; missing evidence is not shown as zero.

See [Open Intelligence Index](../../../benchmarking/open-intelligence-index) for
the built-in hierarchy and [Custom evaluations](../../../benchmarking/custom-evaluations)
for operator-defined evidence.

## Cost and SLOs

Input cost is calculated from the request's pre-routing token estimate. Output
cost uses `max_output_tokens` when the caller supplies it. If neither is known,
the selector falls back to the configured per-million-token rates. This same
request mix is used for the cost factor, `max_cost_per_1m`, and the `cheapest`
fallback.

```yaml
multi_factor:
  slo:
    max_tpot_ms: 200
    max_ttft_ms: 800
    max_cost_per_1m: 5.0
    max_inflight: 50
  latency_percentile: 95
  on_no_candidates: cheapest
```

SLO ceilings are enforced only when the corresponding observation is available.
If every candidate is excluded, `on_no_candidates` chooses `cheapest`, `first`,
or `fail`. `fail` is a strict policy: the Router returns HTTP 503 and never
substitutes the first configured candidate. Eval dry-runs report the selection
as unavailable for the same request.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `objective.strategy` | `weighted` | `weighted` or `lexicographic` |
| `objective.priorities[].factor` | — | `quality`, `latency`, `cost`, or `load` |
| `objective.priorities[].tolerance` | `0` | Relative band from the best value, from 0 to 1 |
| `quality.index` | Catalog default | Versioned quality index |
| `quality.on_missing` | `exclude` when explicit | `exclude` or pool-wide `disable_quality` |
| `quality.min_coverage` | Index policy | Minimum admitted evidence coverage, from 0 to 1 |
| `quality.min_score` | Off | Hard quality floor on the index scale |
| `weights.*` | `0.25` | Balanced quality, latency, cost, and load weights |
| `latency_percentile` | `95` | Observed latency percentile, from 1 to 100 |
| `on_no_candidates` | `cheapest` | `cheapest`, `first`, or `fail` |

Latency and load observations are local to each Router process. Replicas may
therefore make different choices. Use shared telemetry or an infrastructure
router when fleet-global capacity is required.

See the complete fragment:
[`config/fragments/algorithm/selection/multi-factor.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/multi-factor.yaml).
