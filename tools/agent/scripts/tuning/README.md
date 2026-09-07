# DSL Tuning Framework

This internal tool traces failed routing probes to configurable DSL parameters,
proposes a bounded change, and rejects it when protected probes regress. It is
an analytical tuning loop, not gradient training or unrestricted config search.

## Run a Scenario

From the repository root:

```bash
PYTHONPATH=tools/agent/scripts \
python -m tuning.cli SCENARIO \
  --endpoint http://localhost:8080 \
  --config path/to/config.yaml \
  --probes path/to/probes.yaml \
  --router-pid ROUTER_PID \
  --max-iter 10
```

The live scenarios use the router evaluation and config-management APIs to
inspect traces, update the configured YAML, confirm the active config hash, and
roll back rejected changes. Run against a disposable configuration and review
every generated mutation before promoting it.

## Built-in Scenarios

| Scenario | Mode | Purpose |
|---|---|---|
| `privacy` | live | adjust privacy-routing thresholds and identify missing signal coverage |
| `calibration` | live | remove non-beneficial category escalations while protecting higher-severity probes |
| `confidence` | offline | derive per-category confidence strategies from collected observations |

Offline analysis does not need a running router. See
`scenarios/confidence.py` and
`verify_results/run_confidence_verification.py` for the expected input shape.

## Versioned confidence calibration

The confidence scenario can also build a reproducible, offline artifact from a
manifest. The manifest binds the `avg_logprob` transform, dataset and model
identity, three disjoint result splits, selection constraints, and the declared
fallback policy. Each split points to paired small/large result arrays:

```json
{
  "schema_version": "confidence-calibration/v1",
  "name": "mmlu-pro-confidence-v1",
  "method": "avg_logprob",
  "score_domain": {"min": 0.0, "max": 1.0},
  "normalization": {
    "type": "linear_clamped",
    "min_logprob": -3.0,
    "max_logprob": 0.0
  },
  "dataset": {"name": "mmlu-pro", "version": "v1", "digest": "sha256:..."},
  "population": "Fixed MMLU-Pro question set with deterministic answer labels",
  "outcome": "Whether the model answer matches the answer key",
  "expected_impact": "Improve escalation quality within the declared budget",
  "models": {
    "small": {"id": "small-model", "version": "v1"},
    "large": {"id": "large-model", "version": "v1"}
  },
  "splits": {
    "train": {"small_results": "data/train-small.json", "large_results": "data/train-large.json"},
    "calibration": {"small_results": "data/cal-small.json", "large_results": "data/cal-large.json"},
    "held_out": {"small_results": "data/held-small.json", "large_results": "data/held-large.json"}
  },
  "objective": {
    "primary_metric": "accuracy",
    "max_escalation_rate": 0.85,
    "min_net_uplift": 0
  },
  "fallback": {"on_no_safe_threshold": "retain_current"},
  "policy": {"current_threshold": 0.72, "rollback_identity": "threshold-0.72"}
}
```

`train` is retained as part of the manifest and integrity record and is
reported descriptively; threshold selection uses only `calibration`, and the
final estimate is reported on `held_out`. The artifact also includes the
current-policy baseline and collection settings so reviewers can distinguish a
candidate proposal from an approved production change. A candidate is never
auto-promoted; a held-out result with no benefit remains a review finding.

Threshold candidates are selected from the policy regions between adjacent
observed confidence scores. The sweep does not use a fixed epsilon, so close
scores remain distinguishable and narrow operating points cannot be skipped.

Run it without a live router:

```bash
PYTHONPATH=tools/agent/scripts \
python tools/agent/scripts/tuning/verify_results/run_confidence_verification.py \
  --manifest path/to/confidence-manifest.json \
  --output confidence-calibration-artifact.json
```

The command only reads the recorded results and writes the candidate artifact;
it never calls a model or changes active request policy. `held_out` is used
only after selection, so its metrics do not influence the chosen threshold.
The hermetic CI fixture lives under
`tests/fixtures/confidence_calibration_v1/`; it verifies the same manifest and
artifact contract without requiring the real dataset or an API.

### Collect a real MMLU-Pro artifact

The repository does not check in the original calibration dataset. For a new
experiment, prepare a deterministic question file and collect paired results
with an OpenAI-compatible endpoint. The collector defaults to serial requests,
sets `enable_thinking=false`, and validates the answer and logprob response
contract. The small model uses a shorter output budget for confidence scores;
the large model has a separate budget so it can finish longer answers:

```bash
python -m pip install datasets

PYTHONPATH=tools/agent/scripts \
python tools/agent/scripts/tuning/prepare_mmlu_pro_dataset.py \
  --output /tmp/mmlu-pro-confidence-questions.json \
  --samples-per-category 25 \
  --seed 42

PYTHONPATH=tools/agent/scripts \
python tools/agent/scripts/tuning/collect_confidence_results.py \
  --dataset /tmp/mmlu-pro-confidence-questions.json \
  --output-dir /tmp/mmlu-pro-confidence-results \
  --small-model qwen3-8b \
  --large-model qwen3-32b \
  --max-concurrency 1
```

The collector reads the API key from `DASHSCOPE_API_KEY`, writes paired split
results and `manifest.json`, and then builds
`confidence-calibration-artifact.json`. A small pilot is useful for validating
the pipeline; its uncertainty should be reported before treating the result as
a production threshold recommendation.

## Probe Shape

```yaml
decisions:
  - id: standard_route
    expected_decision: standard_route
    variants:
      - id: capital_france
        query: What is the capital of France?
        tags: [baseline]
```

Keep regression probes representative of behavior that must not change. A
tuning result is only as useful as its labels, severity weights, and protected
coverage.

## Add a Scenario

Implement `Scenario` in a module under `tuning/scenarios/`, then register the
class in `BUILTIN_SCENARIOS` in `tuning/cli.py`. A scenario may override result
adaptation, severity, iteration display, or final output construction. Use
`OfflineAnalyzer` directly when no live config mutation is needed.

Keep scenario-specific policy and parsing in the scenario module; keep generic
trace analysis, fix selection, regression checks, and config mutation in the
shared engine modules.

## Validate

```bash
PYTHONPATH=tools/agent/scripts \
python -m pytest tools/agent/scripts/tuning/tests/test_framework.py
```

The tests are hermetic and do not require a live router. Live scenario results
belong beside their probe set and config snapshot, not in this README.
