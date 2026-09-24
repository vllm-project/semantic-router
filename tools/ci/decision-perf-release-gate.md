# Decision Runtime performance release gate

This is the maintained release rule for the six Decision 1.0 models, not a
benchmark result. The executable contract lives in
`decision_perf_release_producer.py`, `decision_perf_release_gate.py`, and their
tests. Raw requests, responses, timings, and private service configuration stay
in protected runner artifacts, never in this repository or a public PR.

## Before measuring

Use an exact clean source checkout, a candidate ROCm image pinned by registry
digest, and a **current-run** six-model ROCm qualification receipt. A protected
provisioner must start six historical/candidate service pairs on the same
reserved GPU, using the same selected model revision and artifact bytes for
each pair. The historical arm needs a reviewed, live-process-attested adapter;
an image label, mounted source tree, or matching response alone does not prove
which code and weights the process loaded. The candidate arm must expose its
launch, artifact, scheduler, and physical-batch identity through `/api/status`
and `/metrics`.

The provisioner supplies one private `decision-paired-baseline-v3` JSON file.
Keep endpoints, container IDs, mounts, credentials, and device details there.
The producer verifies the live containers, read-only mounts, source and artifact
hashes, old-process challenge responses, and candidate launch before and after
measurement. A missing or unverifiable baseline fails closed.

## Produce and validate evidence

Run the producer from the same protected job as the qualification receipt:

```bash
python3 tools/ci/decision_perf_release_producer.py \
  --baseline-config "$PROTECTED_BASELINE_CONFIG" \
  --qualification-receipt "$QUALIFICATION_DIR/qualification.json" \
  --candidate-ref "$CANDIDATE_REF" --owner "$GITHUB_REPOSITORY_OWNER" \
  --source-sha "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID" \
  --run-attempt "$GITHUB_RUN_ATTEMPT" --output-dir "$REPORT_DIR"

python3 tools/ci/decision_perf_release_gate.py validate \
  --report "$REPORT_DIR/report.json" --source-sha "$GITHUB_SHA" \
  --qualification-receipt "$QUALIFICATION_DIR/qualification.json" \
  --candidate-ref "$CANDIDATE_REF" --owner "$GITHUB_REPOSITORY_OWNER" \
  --run-id "$GITHUB_RUN_ID" --run-attempt "$GITHUB_RUN_ATTEMPT"
```

The protected workflow sets these variables and transfers the entire `raw/`
tree with `report.json`. The gate reopens and hashes the raw evidence; copying
an old report, changing its summary, or omitting a required sample cannot
qualify a new commit or run attempt.

## Pass criteria

- All six models pass untimed semantic parity on their selected snapshots,
  including mixed question types and multi-state requests.
- Each model runs the same fixed 3 shapes (32 questions × 1 state, 8 × 8,
  32 × 32) at client concurrency 1, 8, and 32: **54 cells** in total. Each
  cell has three alternating historical/candidate rounds and at least 32
  successful logical workflows per arm per round.
- Every cell has zero HTTP/workflow failures and reconciled request, response,
  decision, timing, and physical-batch counts. The two 32-concurrency
  multi-state cells must demonstrate actual physical batching.
- Candidate decisions per second divided by historical decisions per second
  must be **at least 1.00 in every cell for every model**. A faster model or
  shape cannot compensate for another model's regression. Latency and
  aggregate throughput are published diagnostics, not alternate pass routes.
- The report, ROCm qualification, image digest, model identities, source SHA,
  run ID, and attempt must all join to the same protected build.

Both arms carry equal logical work, but historical multi-state fanout and the
candidate batch API use different wire requests. Do not present this result as
a GPU-kernel-only speedup or as model-quality evaluation. Three rounds reduce
order effects but do not establish statistical noninferiority; retain every
attempt and investigate a failing or borderline cell rather than selecting
favorable rounds, models, or reruns.

The current qualified scheduler policy is 8 running requests, 32 waiting
requests, 4,096 active decision rows, and a model-specific physical batch size
at most 8. Larger limits or a new runtime variant require separate numerical,
memory, and paired-performance qualification before the protected gate changes.
Without a trusted baseline provisioner and current-run evidence, leave the
Decision release switch disabled; distribution must not fall back to an
unqualified image or wheel.
