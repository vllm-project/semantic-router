# data/

Pre-processed workload CDF files used by the examples and CLI.

Each file is a JSON array of `[token_length, cumulative_fraction]` pairs
representing the empirical CDF of total token counts (input + output) for
that trace.

| File | Source | Description |
|---|---|---|
| `azure_cdf.json` | Azure LLM Inference Trace 2023 | 28K prod requests; p90=4.2K tokens |
| `lmsys_cdf.json` | LMSYS-Chat-1M | Single-turn conversations |
| `lmsys_multiturn_cdf.json` | LMSYS-Chat-1M (multi-turn) | Accumulated context per turn |
| `agent_heavy_cdf.json` | Synthetic agent-heavy | SWE-bench 40% + BFCL 25% + RAG 35% |

## Workload mixture scenarios

FleetSim also includes versioned workload-archetype mixture fixtures. They use
privacy-safe aggregate CDF references by default and are intended for simulation
and evaluation evidence only; they do not change production routing behavior.

| File | Scenario | Description |
|---|---|---|
| `workload_mixture_nominal.json` | Fixed mixture | Interactive chat, multi-turn chat, and agent-heavy demand at nominal weights |
| `workload_mixture_drift.json` | Composition drift | Two windows that shift from chat-dominated to agent-heavy demand |
| `workload_mixture_burst.json` | Burst | Three windows with a higher-arrival agent-heavy burst |

Mixture scenario files use schema
`fleet-sim.workload-mixture/v1alpha1`. Each archetype defines an `id`,
`version`, `source`, `arrival_process`, `slo_class`, `model_eligibility`,
`residency`, and nominal `weight`. Optional `composition_schedule` windows can
override weights and arrival-rate multipliers over time.

CDF-only archetypes validate marginal token-length distributions. They do not
preserve unknown cross-feature correlations. Trace archetypes can preserve
correlations that are explicitly present in the trace rows.

Use `vllm-sr-sim mixture-optimize --scenario data/workload_mixture_burst.json`
to evaluate these fixtures with the repository's `FleetOptimizer`, including
aggregate-CDF baselines, individual archetype stress cases, nominal mixtures,
composition drift/burst windows, robust recommendations, sensitivity, and
explicit infeasibility diagnostics.

## Workload archetype forecast aggregates

Forecast aggregate fixtures use schema
`fleet-sim.workload-archetype-forecast/v1alpha1` and taxonomy
`fleet-sim.workload-archetype-taxonomy/v1alpha1`. Each aggregate window records
only content-free, low-cardinality buckets: request count, total-token summary,
latency summary, model class, SLO class, region, archetype weights, and
uncertainty. The privacy policy rejects caller/session/user/request ids,
arbitrary domains/hosts/URLs, and raw prompt/response fields. Windows below the
configured minimum request count fail validation unless explicitly redacted.

| File | Scenario | Description |
|---|---|---|
| `workload_forecast_seasonal.json` | Seasonal pattern | Repeating three-window demand pattern where seasonal-naive should beat simple controls |
| `workload_forecast_drift.json` | Taxonomy drift | Aggregate demand shifts from chat-heavy to agent-heavy windows |
| `workload_forecast_burst.json` | Burst and rollback | Agent-heavy demand spike followed by recovery; can also exercise stale-data rollback by passing `--now-s` beyond `max_staleness_s` |

Use `vllm-sr-sim forecast-backtest --scenario data/workload_forecast_seasonal.json`
to compare static mean, reactive last-window, moving-window, seasonal-naive, and
linear-trend forecasts. The command converts each forecast into a FleetSim
mixture scenario from the source mixture archetypes, reports backtest error,
uncertainty coverage, burst/drift/oscillation diagnostics, and states when
forecasting does not beat simpler controls.

Forecast backtests are advise-only. The generated recommendation is recorded
separately from downstream actuation; FleetSim does not apply cooldowns, quotas,
load-balancer changes, or rollback actions on behalf of the operator. Missing or
stale forecast inputs fall back to ordinary reactive controls.

## Bring your own CDF

A CDF file is a JSON array of `[token_length, cumulative_fraction]` pairs,
sorted by token length, with cumulative fractions in `[0, 1]`.

```python
import json, numpy as np

# lengths = list of (input_tokens + output_tokens) per request
lengths = sorted(lengths)
n = len(lengths)
cdf = [[int(lengths[i]), (i + 1) / n] for i in range(0, n, max(1, n // 200))]
json.dump(cdf, open("data/my_workload_cdf.json", "w"))
```

Then pass `--cdf data/my_workload_cdf.json` to any CLI command.
