---
title: sr-bench 1.0
description: Compare real MoM and single-model capability, cost, latency and token usage on reusable frozen tasks.
---

# sr-bench 1.0

sr-bench compares a Mixture of Models (MoM) entrypoint with single models on the
same frozen tasks. The CLI and **Dashboard → Evaluation** share one durable
service, run IDs, results and reports. Use a small development slice to tune
routing, then evaluate a disjoint holdout before making a quality or savings
claim.

## Choose a scope

Each number below is a whole task count **per target**, not a model-call count.
Coding and agent tasks can make several calls; judges and user simulators add
separately reported costs.

| Benchmark ID | Capability | Smoke | Quick/dev | Standard/holdout |
| --- | --- | ---: | ---: | ---: |
| `mmlu-pro` | Knowledge across 14 subjects | 14 | 500 | 2,000 |
| `gpqa-diamond` | Scientific reasoning | 4 | 40 | 158 |
| `hle` | HLE text-only reasoning | 4 | 40 | 200 |
| `livecodebench` | Programming, cumulative v6 tasks | 2 | 30 | 150 |
| `scicode` | Scientific programming, whole main problems | 1 | 3 | 20 |
| `terminal-bench-2.1` | Sandboxed terminal tasks | 1 | 3 | 15 |
| `simpleqa-verified` | Factual correctness | 5 | 100 | 500 |
| `arc-agi-2` | Public evaluation puzzles, exact output grids | 2 | 12 | 80 |
| `tau3` | τ³ text interaction, three domains | 3 | 12 | 60 |
| **Total** | | **36** | **740** | **3,183** |

Run `vllm-sr benchmark catalog` for the installed adapter identities. Select a
capability slice when that answers the current question. A 500-question MMLU-Pro
slice is a development comparison, not the complete 12,032-question upstream
benchmark. A run without all nine benchmarks has no complete sr-bench score.

Preparation uses pinned sources, stable task IDs, content hashes, a recorded
seed and proportional stratification. Smoke is a subset of quick; standard is
disjoint from quick. Related units such as SciCode subproblems stay together.
Public tasks are not contamination-free. Previously seen GPQA labels require a
retest disclosure even when the local split is called holdout.

## Connect the shared worker

`vllm-sr serve` starts an independent core worker alongside Dashboard. Its store
is `<state-root>/.sr-bench/<stack>/store` and its host API is loopback port
`8090 + stack port offset`. The CLI discovers that store and its private token
from the same workspace. Dashboard reloads and configuration replacement preserve
the worker. `vllm-sr stop` stops it without deleting saved evidence.

The core container has no Docker socket or GPU passthrough and does not include
all upstream harness dependencies. For the code and agent adapters, prepare a
dedicated host worker with pinned interpreters, source checkouts and sandbox
images. Inspect prerequisites with `vllm-sr benchmark setup --benchmark all`;
add `--install` to explicitly install the pinned optional environments and fetch
SciCode test data with a verified SHA256. Add `--build-sandbox` to build the
offline code-grading image; its receipt records the image and base-image digests
and pinned dependencies. Neither setup operation calls a model. The default
cache is `~/.cache/vllm-sr/sr-bench-1.0`, overridable with `SR_BENCH_HOME`.
SciCode data defaults to `assets/scicode/test_data.h5` inside that cache;
`SR_BENCH_SCICODE_TEST_DATA` can select another prepared file. Terminal task
images, source access, judges and simulators still need their declared
preparation. Configure `SR_BENCH_URL` to use that worker instead of creating the local
container. The URL must be reachable from each client; a host-local address and
a Dashboard-container address can differ while referring to the same service.

For standalone development:

```bash
# Provision SR_BENCH_TOKEN privately in both service and client environments.
vllm-sr benchmark --store ./data/sr-bench serve --host 127.0.0.1 --port 8090
```

In another terminal:

```bash
export SR_BENCH_URL=http://127.0.0.1:8090
vllm-sr benchmark --no-autostart runs
```

Non-loopback service binding requires `SR_BENCH_TOKEN`. Use authenticated,
private deployment wiring; browser users access the Dashboard proxy, not the
worker directly. `SR_BENCH_TOKEN_ENV` can name a custom service credential.
Model credentials use separate `api_key_env` references. Keep their values out
of manifests, command arguments and public artifacts.

## Prepare reusable tasks and targets

Install `vllm-sr[bench]` on the preparation host for Parquet sources. Prepare data
on the worker host or its shared store; a local client path is not uploaded to a
remote worker.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --profile quick > mmlu-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark gpqa-diamond --profile quick > gpqa-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset combine \
  mmlu-quick.json gpqa-quick.json > quick-dataset.json
```

Gated sources require the appropriate source access and environment credential.
Local task imports require `--source-path` and `--revision`; their actual bytes
are hashed. `--limit` creates a labeled custom subset. Never edit a prepared
file in place. New questions, selection rules or source bytes create a new
identity.

Register operator-owned targets for Dashboard:

```bash
vllm-sr benchmark --store ./data/sr-bench target register --file targets.json
vllm-sr benchmark target list
```

`targets.json` is an array of target objects. Each has `id`, `kind` (`single` or
`mom`), `base_url`, `model` and, when needed, `api_key_env`. Priced runs supply
`prices` keyed by actual returned model identity. Every price entry has `input`,
`cached_input`, `cache_write` and `output` rates in USD per million tokens.
Record the pricing basis and use the exact same rates in paired runs.

A MoM target also binds the expected runtime `config_hash`; preview uses its
`preview_url`. Priced MoM runs declare `max_inference_calls`. The current direct
MoM adapter requires one fully accounted inference call; unsupported compound
usage cannot be priced from the final selected model alone. Dashboard can
select registered targets but cannot edit their destinations or credentials.

An operator can freeze native generation settings in a target's `request_params`.
These settings override the run's `sampling` defaults, including temperature,
reasoning options and output length when present. Inspect the effective profile
before comparing targets. The run's output cap must accommodate a target's fixed
`max_tokens`; lowering the cap does not rewrite that profile. Choose another
operator-registered profile when different native settings are required.

Judged benchmarks require a fixed single-model judge and
`grader_version: sr-bench-reference-judge-v1`. τ³ also needs a fixed simulator
and release `1.0.1`. Operators supply these in `benchmark-options.json` in the
store. External adapters discover the pinned environments installed by
`benchmark setup`. Set `SR_BENCH_{LCB,SCICODE,TERMINAL,TAU3}_PYTHON` and the
corresponding `_ROOT` variables to override those locations. Exact source
revisions and, for code or terminal tasks, digest-pinned sandbox images remain
required. Preflight reports missing prerequisites before dispatch.

## Plan and run

Create a JSON or YAML manifest with the prepared dataset's `path` and `sha256`,
its matching profile/seed, and the targets to compare. A minimal shape is:

```yaml
version: sr-bench-1.0
name: Development comparison
mode: live
profile: quick
seed: 20260918
dataset:
  path: /absolute/path/from/prepared/manifest/cases.jsonl
  sha256: REPLACE_WITH_PREPARED_DATASET_SHA256
targets:
  - id: single-a
    kind: single
    base_url: http://model-gateway.example:8000/v1
    model: served-model-name
    api_key_env: MODEL_API_KEY
cost_policy: capability_only
limits:
  concurrency: 1
  total_timeout_s: 180
  idle_timeout_s: 30
  max_output_tokens: 4096
  max_run_seconds: 1800
  max_cost_usd: 5
sampling:
  temperature: 0
  top_p: 1
  max_tokens: 4096
```

This example explicitly permits unknown prices and therefore **cannot prove
cost savings**. For cost comparison, supply complete four-bucket target prices
and set `cost_policy: require_priced` (the default). Other limits receive their
recorded defaults during plan creation; inspect the frozen result.

```bash
vllm-sr benchmark plan --manifest candidate.yaml --output frozen.json
vllm-sr benchmark run --manifest frozen.json --detach --idempotency-key iteration-1
vllm-sr benchmark runs
vllm-sr benchmark show RUN_ID
vllm-sr benchmark report RUN_ID --output report.json
```

A repeated idempotency key binds to the same frozen plan. It does not resend
model requests. Without `--detach`, the CLI waits for completion; interrupting
that wait leaves the worker running. Stop actual work with
`vllm-sr benchmark cancel RUN_ID`.

Each call has absolute and idle timeouts, output and repetition limits. Run and
task limits bound elapsed time and call counts. Priced runs reserve estimated
spend before dispatch and stop on reported actual spend. These are not a
provider-enforced universal hard USD cap: missing usage, inaccurate prices or
unsupported backend accounting can leave cost unknown. Unknown usage never
becomes zero or evidence of savings.

## Iterate with preview, replay and live evaluation

1. Run single models and the current MoM on identical quick/dev tasks. Inspect
   wrong answers, failures, decision/model distributions and measured costs.
2. Make one coherent routing change. Use `config validate`, `config plan` and
   `config apply`; wait for the expected active revision. For restart-required
   changes, use the supported `serve --replace-active-config` flow.
3. Bind a new manifest to that revision. Preview MoM targets to check actual
   query decisions and model selections without generating answers.
4. For supported direct routes, replay the preview against saved single-model
   answers. Treat quality, cost and latency as diagnostic estimates.
5. Run the candidate live on the same dev tasks, compare paired results, then
   use the untouched standard split for the prespecified release decision.

```bash
vllm-sr benchmark preview --manifest preview.json --detach
vllm-sr benchmark replay --baseline BASELINE_ID --preview PREVIEW_ID
vllm-sr benchmark compare BASELINE_ID CANDIDATE_ID
vllm-sr benchmark regrade RUN_ID --output regrade.json
vllm-sr benchmark export DEV_RUN_ID --output training-matrix.json
```

Preview has no answer-quality score. With Learning enabled, model selection runs
against a read-only snapshot of active learning state. It returns a concrete
model when selection is resolvable, plus selection status/reason and
`selection_provenance`: config/state hashes, capture time, whether local sampling
occurred and its seed. The snapshot does not update learning state, and a later
live request can differ as state or sampling changes. Keep Learning enabled
during this check; disabling it would test a different policy. The Dashboard
shows these fields beside each preview case and explains unresolved selections.

Replay rejects state-dependent Learning snapshots, unsupported plugin, agent
or compound execution and missing matrix cells instead of calling a model.
Only live runs support measured capability and savings claims. Offline regrade
currently supports saved multiple-choice/grid final answers, preserves the
original results and makes zero model calls. Export is limited to explicitly
marked development rows; holdout and unknown splits are rejected. Export does
not start training.

## Read the results

Reports show full planned denominators, scored/correct/failed counts, per-benchmark
accuracy and uncertainty; four exclusive token buckets; subject and judge/simulator
costs; TTFT and latency percentiles; request-time sum and actual wall time. Missing
metrics remain null. Failed and partial runs remain visible and do not qualify as
completed evaluations.

For a terminal live run, `vllm-sr benchmark reconcile-usage RUN_ID` verifies retained
SSE usage and appends an idempotent, versioned accounting correction without model
requests. Reports, comparisons and subsequent exports use this derived accounting;
the original call/result detail receipts and frozen manifest remain unchanged.
The report exposes the correction hash, time, verified/changed counts and old/new
known spend. Missing or conflicting evidence stays unknown. Reconciliation supports
single-model calls and proven direct MoM calls; a multi-call MoM cannot be repriced
from its final response stream alone.

Reports also provide `cache_neutral_cost_usd`: a counterfactual token-equivalent
subject cost that prices all prompt tokens at the frozen fresh-input rate, plus
output. Comparisons show its saving percentage against the same selected baseline,
alongside observed four-bucket costs. Use both when sequential runs warm caches;
the cache-neutral figure is not billed spend or a measured cache-free execution.

The full sr-bench score uses fixed benchmark weights: MMLU-Pro 10%, SimpleQA 10%,
GPQA 15%, HLE 15%, ARC 10%, LiveCodeBench 10%, SciCode 10%, Terminal-Bench 10% and
τ³ 10%. It requires all nine complete benchmarks. A subset macro result retains
its subset label and is not the full score.

Paired comparison selects the strongest observed single model by the same
aggregate over identical cases and records that selection.
Exact weighted-quality ties prefer the single with the lowest complete known
subject cost, then a stable target ID. The report lists every tied-best single.
If any tied-best single has incomplete cost, savings remain unknown. Savings are
`100 × (1 − candidate subject cost / baseline subject cost)` with complete,
compatible accounting. A small dev sample shows direction; a quality
non-inferiority claim needs a prespecified margin and a holdout confidence
interval. Token-equivalent self-hosted prices do not establish GPU invoice savings.

The default paired quality interval is a conservative weighted Hoeffding bound
for independent case differences. It stays nonzero when every observed pair ties,
including all-wrong samples. The stratified bootstrap interval is retained as a
diagnostic; a degenerate `[0, 0]` bootstrap from a small tied sample does not prove
equivalence. Neither interval includes selection of the strongest observed
baseline, tuning selection or dataset contamination uncertainty.

Dashboard opens on **Runs**, with filters for name/model, status and mode. Each
row shows the completed denominator, failures, persisted update time and target
kind. Read-only polling reconnects after a temporary network failure and discovers
CLI-created runs. Closing or refreshing the page does not restart a run.

In **Create evaluation**, first choose **smoke**, **quick** or **standard**, then
select one or more prepared benchmarks, or **Select all benchmarks**. These are
the actual run profiles; standard uses the holdout split. Available sources must
share a profile, seed and split. **Review plan** composes whole benchmark groups
from those frozen sources without downloading data, resampling questions or
calling a model. Selecting all benchmarks from one source reuses its original
identity; a subset or multi-source composition creates a reusable frozen dataset.
Conflicting selections are rejected rather than silently merged.

**Datasets** provides search, profile/benchmark filters and pagination. Open a
dataset to browse its questions, benchmark coverage and subject groups. Questions
load in pages of 25 with benchmark/category filters and text search; opening one
shows the task instructions and choices, including complete code/agent task
inputs when the pinned source is available. Reference answers, hidden tests and
tool credentials are excluded. Source details and hashes are behind disclosure
controls. **Evaluate dataset** reuses the chosen data. Browsing public questions
does not establish that they are unseen; never use standard tasks for tuning.

Run details separate **Results**, **Questions**, **Calls**, **Evidence** and
**Recipe**. Start with the aggregate results, then inspect individual responses,
accounting and frozen configuration as needed.

**Compare iterations** selects a single-model baseline, then uses checkboxes for
any number of completed candidate runs. Candidates are ordered by creation
time, and selections are preserved in the URL. Quality/cost and iteration charts
accompany paired confidence intervals, savings, tokens, latency and wall time;
CSV and JSON export the comparison. The interface is not limited to two tuning
iterations. Incompatible or incomplete scopes cannot become a comparison row.
A positive estimate with an interval spanning zero is not proof of a gain.

For a MoM target, the operator can register `capture_recipe: true` with its
`config_hash` and canonical `preview_url`. The worker captures a redacted recipe
projection from the Router config API only when the generated and active runtime
hashes match the frozen target before and after capture, and the source config
ETag stays unchanged. **Frozen
recipes** displays and downloads that server-observed snapshot, its capture time
and projection hash. Runtime calls separately acknowledge the active config hash.
Deployment wiring and secrets are omitted; the download is a recipe artifact,
not a complete deployable configuration. Existing runs without a snapshot show
that it is unavailable rather than borrowing a later recipe.

Calls and results load in pages of 100; full call bodies load on inspection.
Metrics and routing distributions come from the complete report independently of
loaded detail pages. Events show the latest 100 entries with access to older
pages. Regrade and training export reuse saved evidence without new model calls.

### Recover unfinished work explicitly

A stopped worker or lost acknowledgement is not permission to retry generation.
Inspect saved dispatches and calls first. On a terminal run, **Review recovery
plan** separates two scopes:

- **Continue undispatched cases** selects cells that have never dispatched a
  model call.
- **Retry known failed cases as new attempts** requires explicit cell selection
  and acknowledgement of additional attempts and spend. Ambiguous calls, unknown
  billing, completed answers and already-claimed cells are excluded.

Both create a separate child run and preserve the parent. Progress, denominator
and costs belong to the child scope; parent spend remains separate in the lineage
panel. A recovery child does not turn a partial parent into a completed benchmark.
The Dashboard saves the exact pending recovery submission in the current tab and
reuses its idempotency key after a lost response. There is no automatic generation
retry or worker restart.

```bash
vllm-sr benchmark recover-plan RUN_ID --mode undispatched --output recovery.json
# Review eligible/excluded cells. Optionally set selected_cells to a reviewed subset.
vllm-sr benchmark recover RUN_ID --plan recovery.json --idempotency-key recovery-1
```

For `--mode failed`, the final command additionally requires
`--acknowledge-new-attempt`. Reuse the same plan, cells and idempotency key when
reconciling an uncertain submission.
