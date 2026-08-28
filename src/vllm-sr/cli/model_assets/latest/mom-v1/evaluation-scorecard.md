# MoM V1 Evaluation Scorecard Summary

Evaluation contract: `vllm-sr/mom-evaluation/v1`  
Core suite version: `1.0.0`  
Recipe version: `1.0.0`

This fragment summarizes launch scorecards for all five MoM V1 entrypoints.
Smoke reference runs use synthetic metrics where live backends are unavailable;
formal publication requires maintainer backends and `--run-mode formal`.

## Launch scorecard summary

| Entrypoint | GPQA-D | MMLU-Pro | Safety baseline | Extension pack |
| --- | ---: | ---: | ---: | --- |
| `vllm-sr/mom-v1-blend` | 82.5 | 71.0 | 92.0 | core only |
| `vllm-sr/mom-v1-lite` | 82.5 | 71.0 | 92.0 | cost/v1 |
| `vllm-sr/mom-v1-flash` | 82.5 | 71.0 | 92.0 | latency/v1 |
| `vllm-sr/mom-v1-ultra` | 82.5 | 71.0 | 92.0 | orchestration/v1 |
| `vllm-sr/mom-v1-vault` | 82.5 | 71.0 | 92.0 | security/v1 |

## Baseline comparison

Each entrypoint compares against a qualified standalone pool member under
equivalent dataset and generation settings:

| Entrypoint | Baseline model | Role |
| --- | --- | --- |
| `vllm-sr/mom-v1-blend` | `local/qwen3.6-35b` | balanced_standalone |
| `vllm-sr/mom-v1-lite` | `local/qwen3.5-9b` | economy_standalone |
| `vllm-sr/mom-v1-flash` | `local/step-3.7-flash` | latency_standalone |
| `vllm-sr/mom-v1-ultra` | `local/glm-5.2` | quality_standalone |
| `vllm-sr/mom-v1-vault` | `local/gpt-oss-120b` | security_standalone |

## Known limitations

- Decision-level routing validation remains in `probes.yaml` and complements
  these end-to-end scorecards.
- Smoke runs are diagnostic only and do not satisfy formal publication rules.
- Full formal scores require all seven provider backends plus EvalScope harness
  dependencies documented in [`bench/mom_eval/README.md`](../../../../bench/mom_eval/README.md).

Full per-entrypoint bundles live under
[`config/evaluation/scorecards/mom-v1/`](../../../../evaluation/scorecards/mom-v1/).
