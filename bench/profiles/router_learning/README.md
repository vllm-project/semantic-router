# Router Learning Evaluation Profiles

These profiles turn the deterministic Router Learning architecture evaluation
into a pass/fail check. Each profile defines minimum or maximum bounds for the
metrics produced by `bench/agentic_routing_experiment.py
--learning-architecture`.

## Run the evaluation

Use the default `pr` profile for the shorter gate:

```bash
python3 bench/agentic_routing_experiment.py \
  --learning-architecture \
  --profile pr \
  --output-dir /tmp/router-learning-eval
```

The Make target writes results to `.agent-harness/router-learning-eval`:

```bash
make bench-router-learning
make bench-router-learning PROFILE=release
```

The output directory contains:

- `learning_architecture_summary.json`: measured routing metrics;
- `learning_architecture_verdict.json`: every threshold check and the overall
  `passed` value.

The process exits with `0` when every threshold passes, `1` when a threshold is
breached, and `2` when `--profile` is used without
`--learning-architecture`.

## Included profiles

| Profile | Intended use | Notable bounds |
| --- | --- | --- |
| `pr` | Fast deterministic regression check | 100% correctness, explainability, and bypass floors; switch rate at most 20%; cost savings at least 0%; p95 overhead at most 50 ms. |
| `release` | Stricter pre-release check | The same correctness floors; switch rate at most 15%; cost savings at least 5%; p95 overhead at most 25 ms. |

Both profiles are calibrated to the deterministic fixtures in the architecture
evaluation. They are regression thresholds, not service-level objectives for a
deployed workload.

## Custom profiles

`--profile` also accepts a JSON file path. A profile maps each metric to a
minimum or maximum bound:

```json
{
  "profile": "custom",
  "thresholds": {
    "routing_correctness_pct": {"min": 100},
    "p95_overhead_ms": {"max": 40}
  }
}
```

Missing or null metrics fail the check. This prevents an incomplete evaluation
from satisfying a profile silently.
