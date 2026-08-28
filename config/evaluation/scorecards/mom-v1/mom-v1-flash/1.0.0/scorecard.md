### Launch scorecard

- Evaluation contract: `vllm-sr/mom-evaluation/v1`
- Core suite version: `1.0.0`
- Entrypoint: `vllm-sr/mom-v1-flash`
- Recipe version: `1.0.0`
- Run mode: `smoke`
- Generated: `2026-08-28T06:56:17+00:00`

| Metric | Value | Layer | Classification |
| --- | ---: | --- | --- |
| `avg_total_tokens` | 980.0 | operational | diagnostic |
| `failure_rate` | 0.02 | operational | diagnostic |
| `gpqa_d` | 82.5 | general_quality | blocking |
| `ifeval` | 68.0 | instruction_following | blocking |
| `latency/v1:p99_latency_ms` | 950.0 | - | blocking |
| `latency/v1:tail_latency_ratio` | 2.1 | - | diagnostic |
| `latency/v1:time_to_first_token_ms` | 180.0 | - | blocking |
| `mmlu_pro` | 71.0 | general_quality | blocking |
| `p50_latency_ms` | 420.0 | operational | diagnostic |
| `p99_latency_ms` | 1800.0 | operational | diagnostic |
| `robustness_matrix` | 88.0 | robustness | diagnostic |
| `safety_baseline` | 92.0 | safety_baseline | blocking |

### Baseline comparison

- `local/step-3.7-flash` (latency_standalone): {"gpqa_d": {"value": 81.0}, "mmlu_pro": {"value": 70.0}}

### Known limitations

- Smoke or diagnostic runs are not publishable launch scores.
- Full formal scores require all seven provider backends to be reachable.

Full result bundle: [`mom_eval_result.json`](config/evaluation/scorecards/mom-v1/mom-v1-flash/1.0.0/mom_eval_result.json)

