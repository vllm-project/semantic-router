# Decision paired performance release gate

The protected ROCm job must generate `report.json` and its `raw/` directory in
the same run that builds the candidate image. Qualify them with:

```bash
python3 tools/ci/decision_perf_release_gate.py validate \
  --report "$REPORT_DIR/report.json" --source-sha "$GITHUB_SHA"
```

The gate expects exactly six Decision model IDs, each measured at 32 questions
and 1 state, 8 questions and 8 states, and 32 questions and 32 states. Every
shape needs concurrency 1, 8, and 32. Each concurrency cell needs three
alternating old/new throughput rounds with at least 32 successful workflows
per arm per round. The raw benchmark receipt, untimed parity result, and
workflow log for every shape travel with the report. Their SHA256 hashes and
path confinement are checked again after the CI artifact handoff.

Both arms must use the same model revision and hardware/network scope. The
new runtime image source, benchmark source, and checked out source must match
the current commit. The gate verifies zero failed workflows, no parity
mismatches, matching decision counts, recomputed throughput and latency
percentiles, correct alternating wave timing, and positive physical batch
counter deltas. High-load cells must show actual batching above one row per
physical batch and no throughput regression worse than 20%.

For each model, concurrency 32 must show at least a 20% throughput gain on
either multi-state workload, or a 10% p95 workflow latency gain on the wide
single-state workload. The threshold is applied to the measured results; a
model that misses it blocks qualification. The measurement compares synthetic
HTTP workflows, not task accuracy. Multi-state results compare the old
single-state fanout with the new batch protocol, so their wire requests are
different. The report labels that scope rather than attributing every gain to
the GPU kernel.

The code does not pin model-file versions. Each measurement records the
old/new model revision and artifact identity used for that run and checks
they agree. A new weight revision can be qualified without changing the gate.

The protected job must have a trusted old serving baseline and an immutable
new candidate image available on the same ROCm host. If either is unavailable,
the report cannot be produced and release stays closed. Publishing a previous
run's JSON cannot qualify a different source commit.
