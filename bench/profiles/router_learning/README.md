# Router Learning architecture eval — threshold profiles

Pass/fail gate for the deterministic Router Learning architecture eval
(`bench/agentic_routing_experiment.py --learning-architecture`). Implements the
"named reusable profiles + pass/fail thresholds" slice of #2244.

## What it does

The `--learning-architecture` run produces a summary of routing-quality metrics
over deterministic fixtures (session-aware / bandit / Elo / personalization
adaptations). A **profile** declares min/max bounds per metric; the eval is
gated against it and **exits non-zero on any breach**, so a PR or release
pipeline fails when Router Learning routing quality, cost, cache behavior, or
latency regresses.

## Usage

```bash
# Run + gate against the per-PR profile (default)
python bench/agentic_routing_experiment.py --learning-architecture --profile pr --output-dir OUT
# or:
make bench-router-learning              # PROFILE=pr
make bench-router-learning PROFILE=release
```

Outputs in `OUT/`:
- `learning_architecture_summary.json` — the metric summary (unchanged).
- `learning_architecture_verdict.json` — per-metric checks + overall `passed`.
- a human-readable PASS/FAIL verdict on stderr (for CI logs / PR comments).

Exit codes: `0` pass · `1` threshold breach · `2` misuse (`--profile` without
`--learning-architecture`).

## Profiles

| Profile | Intent | Notable bounds |
|---------|--------|----------------|
| `pr` | per-PR gate | correctness/explainability/bypass = 100%; switch-rate ≤ 20%; cost savings ≥ 0%; p95 overhead ≤ 50 ms (CI-host headroom) |
| `release` | release sign-off | same correctness ceiling; switch-rate ≤ 15%; cost savings ≥ 5%; p95 overhead ≤ 25 ms |

Profiles are JSON: `{"profile": NAME, "thresholds": {METRIC: {"min": x} | {"max": y}}}`.
`--profile` also accepts an explicit path to a custom profile JSON.

## Design notes

- **Missing/null metrics fail closed.** A metric absent from the summary (or
  `None`, e.g. `bypass_correctness_pct` when there are no bypass cases) is a
  failure, so a profile can never be silently satisfied by missing data.
- **Thresholds are calibrated to the deterministic fixtures**, which are
  hand-built to route perfectly — hence the 100% correctness floors. The
  operational bounds (switch-rate, cost, cache, latency) carry headroom tuned
  to the current fixture values (see `git blame` for the baseline run).
- **Scope (first slice of #2244):** thresholds + named profiles + CI gate on the
  existing harness. Deferred follow-ups: live Router Replay mode, and the
  session-scope / tool-loop / idle-reset / privacy fixture expansion.
