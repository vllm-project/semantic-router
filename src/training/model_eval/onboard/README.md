# Onboarding Evaluation

This module provides a unified onboarding workflow for model evaluation,
threshold policy reporting, and optional routing model updates.

## CLI

The CLI is exposed via:

```bash
python src/training/model_eval/onboard_eval.py --help
```

## Run System Eval + Threshold Report

```bash
python src/training/model_eval/onboard_eval.py \
  --config onboarding_config.json \
  --test-name system_eval \
  --datasets mmlu-pro-en mmlu-prox-zh fact-check-en feedback-en \
  --max-samples 50 \
  --report-out system_eval_report.json \
  --thresholds-out onboarding_thresholds.json \
  --min-accuracy 0.7 \
  --max-latency-ms 2000
```

## Write Thresholds Back Into Config

```bash
python src/training/model_eval/onboard_eval.py \
  --config onboarding_config.json \
  --test-name system_eval \
  --datasets mmlu-pro-en fact-check-en \
  --thresholds-out onboarding_thresholds.json \
  --update-config
```

This writes an `onboarding_thresholds` object into the JSON config file (or into
`--config-out` if provided) so the evaluation policy is stored alongside the
model onboarding config.

## Optional: Update Routing Models

```bash
python src/training/model_eval/onboard_eval.py \
  --config onboarding_config.json \
  --test-name system_eval \
  --datasets mmlu-pro-en \
  --ml-benchmark-queries queries.jsonl \
  --ml-benchmark-model-config models.yaml \
  --ml-benchmark-output benchmark_output.jsonl \
  --ml-train-output ./ml-models
```
