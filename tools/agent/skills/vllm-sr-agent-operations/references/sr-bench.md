# sr-bench 1.0 evaluation loop

Use this reference for measured single-model/MoM comparison, route optimization,
or reusable evaluation data. Use [route verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) for
individual API/tool/context-boundary checks. Discover installed commands with
`vllm-sr benchmark --help`; the [product guide](https://vllm-sr.ai/docs/benchmarking/sr-bench)
contains manifest examples and the nine-adapter scope.

## Select work that answers the objective

State the candidate, strongest single-model baseline, dataset identity, quality
margin, cost basis and time/call/output budget before launching paid work.
Quick/dev is for tuning; standard is a disjoint holdout. Smoke demonstrates the
protocol, not statistical capability. Use a capability slice for small routing
changes, then expand only when the evidence warrants it. A slice is never a full
upstream benchmark result or a complete sr-bench score.

The defaults per target are smoke 36, quick 740, standard 3,183 whole tasks across
MMLU-Pro, GPQA Diamond, HLE text, LiveCodeBench v6, SciCode, Terminal-Bench 2.1,
SimpleQA Verified, ARC-AGI-2 and τ³ text. MMLU-Pro quick is 500 subject-stratified
questions, not 12,032. Agent trajectories, code subtasks and grading can require
multiple calls. Keep whole tasks together and report simulator/judge spend.

## Establish one execution owner

Use the service already selected by the CLI and Dashboard. `vllm-sr serve` owns
an independent core worker at `<state-root>/.sr-bench/<stack>/store`; normal
Dashboard/config reloads preserve it. `SR_BENCH_URL` selects an externally
prepared worker without creating another container. For remote URLs, prepare
data and register targets on that worker's host/shared store; local paths are
not uploads. Confirm the service/store identity before dispatch.

```bash
vllm-sr benchmark catalog
vllm-sr benchmark setup --benchmark all
vllm-sr benchmark --no-autostart runs
vllm-sr benchmark target list
```

`setup` inspects prerequisites by default. `--install` explicitly prepares pinned
optional source/interpreter environments and downloads SHA256-verified SciCode
test data. `--build-sandbox` builds the offline code grader and records image and
base-image digests with pinned dependencies. These steps make no model calls.
Terminal task images, source access and fixed judge/simulator targets remain
prerequisites; the core container is not an all-adapter runtime. Read
setup/preflight errors before requesting paid work.

Service tokens stay server-side. Targets name environment references rather than
secret values. Dashboard selects an operator-owned registry and cannot redirect
credentials or choose executables. Distinguish service authentication, Router
management credentials, subject-model credentials and judge/simulator targets.

## Freeze, inspect, execute

Prepare data with `benchmark dataset prepare --benchmark ID --profile PROFILE`;
combine compatible prepared manifests with `dataset combine`. Sources, revision,
case IDs, stratification, seed and content digest accompany each dataset. Never
edit bound task files. Changing a source or selection creates a new identity.
Do not route on benchmark names, expected answers or split labels. Previously
seen GPQA labels require a retest disclosure; public tasks are not guaranteed
uncontaminated.

A run manifest binds targets and runtime hashes, messages/tools, source/grader
versions, sampling, prices and limits. Register Dashboard targets with
`benchmark target register --file targets.json` on the worker host. MoM targets
must use their actual routed endpoint. Price all four exclusive token buckets
for every billed model; unsupported compound usage cannot be priced from only
the selected model. The direct MoM adapter requires complete single-call
accounting. Choose `capability_only` explicitly if prices are unavailable and
make no savings claim.

```bash
vllm-sr benchmark plan --manifest candidate.json --output frozen.json
vllm-sr benchmark run --manifest frozen.json --detach --idempotency-key loop-1
vllm-sr benchmark show RUN_ID
vllm-sr benchmark show RUN_ID --events
vllm-sr benchmark show RUN_ID --calls
vllm-sr benchmark report RUN_ID --output report.json
```

An idempotency key must remain attached to the same plan after a lost submission
acknowledgement. Inspect the existing run before another submission. Do not use
a fresh key, retry a generation, resume an interrupted run or restart a stopped
worker as automatic recovery. `Ctrl-C` stops the CLI wait; `benchmark cancel
RUN_ID` stops actual work and retains partial evidence.

Explicit recovery is a new child attempt, not mutation of a failed parent. Use
`benchmark recover-plan RUN_ID --mode undispatched --output recovery.json` to
inspect never-dispatched cells. `--mode failed` additionally excludes unknown
billing, ambiguous or completed generations and requires deliberate new-attempt
authorization. Review `eligible_cells` and `excluded`; use `selected_cells` for an
explicit subset, then `benchmark recover RUN_ID --plan recovery.json
--idempotency-key KEY`. Failed-case recovery also requires
`--acknowledge-new-attempt`. Preserve this exact request/key after lost responses.
The child reports only its execution cells; inherited parent progress and spend
stay separate. A completed child does not qualify an incomplete parent.

For frozen recipe artifacts, register MoM targets with `capture_recipe: true` and
the expected config hash/canonical preview URL. The server captures a redacted
projection only when generated/active runtime hashes match the target before and
after capture, with an unchanged source config ETag. Preserve this snapshot
and the separate runtime-call hash acknowledgements; never substitute the latest
configuration for missing historical evidence.

Absolute/idle deadlines, output/repetition guards and task/run call/time limits
bound work. Spend reservations and reported actual-cost stopping are not a
provider-enforced universal hard USD cap. Unknown usage remains unknown; do not
silently substitute zero or continue a cost-qualified claim through it.

## Perform the optimization loop

1. Save live single-model and MoM baselines on identical dev tasks. Inspect
   routing decisions, errors, final answers, usage and time before changing config.
2. Make one coherent policy change using the existing config schema, validate,
   plan and apply flow. Wait for the expected active runtime hash. Restart-required
   changes use the authorized `serve --replace-active-config` path.
3. Bind a new manifest to that hash and run `benchmark preview`. Preview checks
   actual Router decisions and selections but has no capability score. Keep
   Learning enabled: its preview selects against a read-only snapshot of the
   active learning state without updating it. Inspect `selection_provenance`
   for the config/state hashes, capture time and sampling seed. A sampled
   snapshot choice is not a promise of the later live choice. Preserve unresolved
   `execution_required` selections with their reasons.
4. For eligible direct/static routing, use `benchmark replay --baseline BASELINE_ID
   --preview PREVIEW_ID`. It reuses saved matrix cells and makes no model calls.
   State-dependent Learning previews, plugin, agent, changed-prompt and compound
   paths require live evaluation.
5. Run the candidate live on the same dev cases and use `benchmark compare
   BASELINE_ID CANDIDATE_ID`. Expand to the untouched standard split only for
   the prespecified acceptance decision; never tune against its failures.

Offline `benchmark regrade RUN_ID --output PATH` currently regrades saved MCQ/grid
final answers with a separate artifact and zero model calls. `benchmark export
DEV_RUN_ID --output PATH` creates an explicit dev training matrix and rejects
holdout/unknown splits. Neither command trains a model or changes original results.

## Review evidence and hand off

Validate actual planned/completed/scored/failed counts, final-channel grading,
model/config acknowledgements, all four token buckets, subject cost versus
simulator/judge overhead, TTFT/latency, actual wall time and accounting coverage.
Partial or failed runs cannot be presented as qualified. Replay estimates remain
separate from measured live metrics. Regrading does not erase the prior grader.

The full score requires all nine complete benchmarks with fixed versioned
weights. Show per-benchmark denominators and uncertainty with every aggregate.
The baseline is the best observed single by the stated aggregate over identical
cases, not a per-question oracle. Exact weighted-quality ties choose the lowest
complete known subject cost, then stable target ID. Show all tied-best IDs and
suppress savings when any tied-best single has incomplete cost. Never choose an
expensive quality tie to inflate savings. Savings use complete compatible subject costs:
`100 * (1 - candidate_cost / baseline_cost)`. Small quick results show direction;
quality equivalence requires a prespecified margin and holdout interval. Token
prices for self-hosted inference are not GPU invoice savings.

For requested Dashboard acceptance, verify the same service/run IDs, launch a
bounded run, inspect metrics and case artifacts, compare, cancel and reload the
page. Verify task filters, data-set identity, read reconnection, explicit recovery
scope/lineage, and same-key reconciliation after a lost recovery response. Compare
current Balance plus both optimization revisions against a compatible single-model
baseline using the URL-persisted selections. Inspect/download the captured recipe
and verify cost/quality uncertainty and full denominators. Review narrow-screen
layout and avoid clipping controls or claiming gains from incompatible datasets.
A page load or mocked browser test alone is not live acceptance.

Periodic follow-ups observe durable status and deadlines. Notify on meaningful
stage completion, a failure or required action; ordinary counter changes need no
message. A stale heartbeat is not proof that generation is healthy. Reconcile
saved dispatch/call receipts before action, preserve failed evidence, and stop
repeat notifications for an unchanged acknowledged blocker. Deliver artifact
links and limitations; disable a completion-only follow-up after delivery.
