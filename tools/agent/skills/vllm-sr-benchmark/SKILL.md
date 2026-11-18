---
name: vllm-sr-benchmark
description: Plan, run and report a bounded vLLM Semantic Router evaluation with sr-bench. Use when a user needs reproducible preview, replay or live evidence for a recipe or model, including case coverage, quality, latency and cost limits.
---

# Run a reproducible Router benchmark

Use the existing benchmark CLI (`vllm-sr benchmark --help`) and
[sr-bench guide](https://vllm-sr.ai/docs/benchmarking/sr-bench).
The [operations evaluation reference](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md)
has the current target, recovery and comparison contracts. A smoke slice
checks the workflow; it is not a full sr-bench score or an unseen holdout.

## 1. Choose what can be measured

| Mode | Produces | Cannot establish |
| --- | --- | --- |
| Preview | Router selection, signal/decision distribution and failures on frozen inputs | Answer quality, generation cost or backend latency |
| Replay | An eligible route estimate from compatible saved single-model answers | New live latency, new generations or a final production verdict |
| Live | Actual model answers, grading, usage, latency and failures | Savings when prices or usage are unknown; broad quality from a tiny slice |

Use `benchmark replay-options` before attempting replay; compatibility is
determined by sr-bench's saved evidence, not by a similar model name. A
bounded page may have more results; follow its cursor. For a new recipe, use
preview first, then live only when needed and authorized. Do not rewrite the
workload, seed, target or request settings merely to make a replay eligible.

## 2. Freeze the plan

Choose the benchmark tasks and profile that match the objective. Record exact
case IDs, source revision and dataset digest; the config hash and recipe;
target model/artifact revisions and reachable worker-side URLs; effective
sampling, reasoning, tool protocol and output limits; grader/judge versions;
and four-bucket prices or `capability_only`. Keep a development set separate
from a reserved holdout. A changed case, judge, price or request profile makes
a different experiment.

Prepare or select an existing dataset, then create a manifest and freeze it:

```bash
vllm-sr benchmark dataset options
vllm-sr benchmark dataset prepare --benchmark mmlu-pro --profile smoke --limit 2 > dataset.json
vllm-sr benchmark plan --manifest candidate.yaml --output frozen.json
```

Dataset preparation can download files but makes no model calls. Inspect the
frozen plan rather than assuming manifest defaults: count planned cells and
expected model, judge and simulator calls; bound concurrency, output tokens,
timeouts and spending. A MoM target uses the Router's current
`active_runtime_hash`; the sr-bench worker must reach its listener from inside
its container and have authorized debug trace access for identity receipts.
`--minimal` omits the sr-bench service. See
[Plan and run](https://vllm-sr.ai/docs/benchmarking/sr-bench/plan-and-run).

Before live dispatch, show the planned call count, cost basis or why spend is
unknown, and the execution limits. Start live calls when the user has
authorized that run; otherwise request authorization for the concrete frozen
plan. Preview and replay also use durable runs and must not be submitted twice
with a new key after a lost acknowledgement.

## 3. Execute and follow the durable run

Use a stable idempotency key and preserve the returned run ID. `--detach`
returns before completion; inspect that run instead of submitting another.
Without `--detach`, interrupting CLI wait leaves the worker running. Use
`benchmark cancel RUN_ID` to stop actual work when asked.

```bash
vllm-sr benchmark preview --manifest frozen.json --detach --idempotency-key smoke-preview-1
vllm-sr benchmark runs
vllm-sr benchmark show RUN_ID
vllm-sr benchmark report RUN_ID --output report.json
```

For a live frozen plan use `benchmark run` in place of `benchmark preview` and
a distinct stable key. Use `benchmark show RUN_ID --results` and `--calls`
when aggregate results need explanation. Inspect failures before invoking the
existing recovery commands; a recovery child does not erase its parent's
failed cells. The [run reliability procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md#run-reliability)
describes retry and cancellation boundaries.

## 4. Reconcile and report

Report planned, completed, failed, skipped and ungraded cells, with case IDs
and reasons for every gap. Keep run ID, frozen manifest/dataset digests,
source/grader revisions, config hash, target definitions and actual request
profile alongside the report. Include route distribution and unknown/fallback
counts; final-channel quality with its denominator; latency and elapsed time;
four token buckets, subject versus judge/simulator cost, reliability and any
guard outcomes the service records. Unknown usage is not zero. An incomplete
or ungraded run is insufficient evidence for a quality/cost verdict.

State what mode and profile actually measured. `capability_only` supports
quality analysis, not a savings claim. A full sr-bench score needs its full
benchmark coverage; a two-case smoke report must be labeled as such. For a
paired comparison, use the separate
[comparison workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md#preserve-comparability)
and sr-bench's compatibility checks rather than subtracting two unrelated
aggregate reports.

For a small no-generation preparation fixture and a bounded preview/live
walkthrough, read [Two-case smoke](references/two-case-smoke.md). It uses
synthetic public questions to verify the mechanics, not benchmark quality.
