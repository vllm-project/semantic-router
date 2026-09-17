# DSL Tuning Framework

This internal tool traces failed routing probes to configurable DSL parameters,
proposes a bounded change, and rejects it when protected probes regress. It is
an analytical tuning loop, not gradient training or unrestricted config search.

## Run a Scenario

From the repository root:

```bash
PYTHONPATH=tools/calibration \
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

Offline analysis does not need a running router. The confidence verifier reads
paired `small_results.json` and `large_results.json` files from an explicit input
directory and writes its report to the requested destination:

```bash
PYTHONPATH=tools/calibration \
python -m tuning.verify_confidence \
  --data-dir path/to/paired-results \
  --output .cache/calibration/confidence-verification.json
```

Both input files contain JSON arrays whose records include `question_id`,
`predicted`, and `correct_answer`. Small-model records also contain `category`
and `avg_logprob`. Only matching question IDs are evaluated. See
`scenarios/confidence.py` for confidence normalization and severity policy.
Keep generated reports with the evaluated inputs and config snapshots outside
the tool source directory.

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
PYTHONPATH=tools/calibration \
python -m pytest tools/calibration/tuning/tests/test_framework.py
```

The tests are hermetic and do not require a live router. Live scenario results
belong beside their probe set and config snapshot, not in this README.
