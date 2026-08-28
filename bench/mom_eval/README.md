# MoM Evaluation Runner

Run first-class Mixture-of-Models (MoM) evaluation under the versioned
`vllm-sr/mom-evaluation/v1` contract. This complements decision-level routing
evaluation in `probes.yaml` by measuring end-to-end quality, operational
behavior, baseline comparison, and objective-specific extension packs.

## Prerequisites

1. Serve the MoM entrypoint under test:

```bash
vllm-sr serve vllm-sr/mom-v1-blend
```

2. Ensure provider backends listed in the recipe `config.yaml` are reachable.
3. For EvalScope-backed benchmarks, install optional dependencies:

```bash
python3 -m venv .venv-mom-eval
. .venv-mom-eval/bin/activate
python -m pip install -e 'bench[real_eval]'
python -m pip install -r tools/agent/requirements.txt
```

## Commands

Validate contracts:

```bash
make mom-eval-validate
```

Smoke evaluation (CI-friendly, synthetic metrics when backends are unavailable):

```bash
make mom-eval-smoke MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-blend
```

Release-candidate evaluation with regression gate:

```bash
make mom-eval-rc MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-ultra
```

Publish scorecard artifacts:

```bash
make mom-eval-publish MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-vault
```

Generate all MoM V1 reference scorecards:

```bash
make mom-eval-reference
```

Direct CLI:

```bash
python bench/mom_eval/run_mom_eval.py \
  --manifest config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml \
  --entrypoint vllm-sr/mom-v1-blend \
  --run-mode smoke \
  --synthesize \
  --output-dir config/evaluation/scorecards/mom-v1/mom-v1-blend/1.0.0
```

## Artifact layout

Published scorecards are stored at:

```
config/evaluation/scorecards/<recipe>/<entrypoint>/<version>/
  mom_eval_result.json
  scorecard.json
  scorecard.md
  regression_report.json
  provenance.yaml
```

The index file `config/evaluation/scorecards/index.yaml` tracks historical
scorecards for regression comparison.

## Related docs

- [MoM evaluation guide](../../../website/docs/benchmarking/mom-evaluation.md)
- [Core suite manifest](../../../config/evaluation/mom-core-suite/v1/manifest.yaml)
- [Extension pack registry](../../../config/evaluation/packs/registry.yaml)
- [PL-0038 execution plan](../../../tools/agent/docs/plans/pl-0038-mom-evaluation-epic.md)
