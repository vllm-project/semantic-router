# Two-case sr-bench smoke

This synthetic import has two public multiple-choice questions. It exercises
dataset provenance and bounded run setup; its answers say nothing about
real-world quality. The source is
[two-case-mcq.jsonl](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-benchmark/fixtures/two-case-mcq.jsonl).

## Freeze data without model calls

From a repository checkout, prepare it in a local, disposable store:

```bash
vllm-sr benchmark --store ./data/sr-bench-skill-smoke dataset prepare --local \
  --benchmark mmlu-pro --profile smoke \
  --source-path tools/agent/skills/vllm-sr-benchmark/fixtures/two-case-mcq.jsonl \
  --revision synthetic-v1 --limit 2 > dataset.json
```

Check `case_count: 2`, `custom_subset: true`, `profile: smoke`, `split: dev`,
the source file SHA and the frozen dataset SHA. Save the full manifest and
case IDs; do not publish a local path as though another host can read it.
The import uses `mmlu-pro`'s adapter format solely to exercise the pipeline;
it is not an official MMLU-Pro sample.

## Preview on a maintained local stack

First verify the Router and sr-bench service are running, the public
entrypoint is listed, and `GET /api/v1/config/hash` reports the active hash.
The sr-bench worker reaches the Router at its container DNS name, not the
host's loopback address. Keep the prepared dataset on the worker, or use the
shared-service preparation path instead of the local import. Freeze a
`mode: preview` manifest with that worker-visible dataset path/SHA, a MoM
target bound to the active hash, one case per source row, and tight limits.
The exact fields and command are in
[Plan and run](https://vllm-sr.ai/docs/benchmarking/sr-bench/plan-and-run).

```bash
vllm-sr benchmark plan --manifest preview.yaml --output preview-frozen.json
vllm-sr benchmark preview --manifest preview-frozen.json --detach --idempotency-key two-case-preview-1
vllm-sr benchmark show RUN_ID
vllm-sr benchmark report RUN_ID --output preview-report.json
```

Require two planned cells for one target, reconcile both outcomes, and record
selected decisions/models, missing traces, signal errors and config identity.
Preview gives no answer-quality score. If the worker cannot read the imported
dataset path or the Router refuses the hash, fix that boundary and freeze a
new plan rather than editing the frozen one.

## Optional bounded live step

Only after the user authorizes the concrete live plan, freeze the same two
cases and one reachable target with `mode: live`, `cost_policy:
capability_only`, concurrency 1, and a bounded output and elapsed-time
budget. The plan should show two subject model calls; a benchmark's grading
may add judge or simulator calls, so inspect the frozen plan before starting.

```bash
vllm-sr benchmark plan --manifest live.yaml --output live-frozen.json
vllm-sr benchmark run --manifest live-frozen.json --detach --idempotency-key two-case-live-1
vllm-sr benchmark show RUN_ID --results
vllm-sr benchmark report RUN_ID --output live-report.json
```

Reconcile both case IDs, final answers, grading status, failures, token usage,
latency and any cost that was actually recorded. This run is a mechanics
smoke, not a measured performance advantage. A replay needs a compatible
saved single-model baseline and preview; discover it with
`vllm-sr benchmark replay-options` rather than manufacturing an estimate.
