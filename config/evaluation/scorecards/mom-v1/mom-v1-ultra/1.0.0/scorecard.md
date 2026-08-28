### Launch scorecard

- Evaluation contract: `vllm-sr/mom-evaluation/v1`
- Core suite version: `1.0.0`
- Entrypoint: `vllm-sr/mom-v1-ultra`
- Recipe version: `1.0.0`
- Run mode: `smoke`
- Generated: `2026-08-28T06:56:17+00:00`

| Metric | Value | Layer | Classification |
| --- | ---: | --- | --- |
| `avg_total_tokens` | 980.0 | operational | diagnostic |
| `failure_rate` | 0.02 | operational | diagnostic |
| `gpqa_d` | 82.5 | general_quality | blocking |
| `ifeval` | 68.0 | instruction_following | blocking |
| `mmlu_pro` | 71.0 | general_quality | blocking |
| `orchestration/v1:avg_provider_calls` | 2.3 | - | diagnostic |
| `orchestration/v1:bounded_resource_adherence` | 0.98 | - | blocking |
| `orchestration/v1:orchestration_quality_delta` | 1.5 | - | blocking |
| `p50_latency_ms` | 420.0 | operational | diagnostic |
| `p99_latency_ms` | 1800.0 | operational | diagnostic |
| `robustness_matrix` | 88.0 | robustness | diagnostic |
| `safety_baseline` | 92.0 | safety_baseline | blocking |

### Baseline comparison

- `local/glm-5.2` (quality_standalone): {"gpqa_d": {"value": 81.0}, "mmlu_pro": {"value": 70.0}}

### Known limitations

- Smoke or diagnostic runs are not publishable launch scores.
- Full formal scores require all seven provider backends to be reachable.

Full result bundle: [`mom_eval_result.json`](config/evaluation/scorecards/mom-v1/mom-v1-ultra/1.0.0/mom_eval_result.json)

