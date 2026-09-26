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

To avoid a host port conflict, set `VLLM_SR_BENCH_PORT` to an absolute port from
1 through 65535 for both `vllm-sr serve` and `vllm-sr benchmark`. This overrides
only the worker's loopback host port; its container port remains 8090 and the
stack port offset is not added to the override.

When only the selected Dashboard image changes, `serve` upgrades its managed
worker after verifying the same launch settings, store and credentials. The CLI
briefly pauses the worker to check its durable journal before replacing it.
Active runs and dataset preparations block the upgrade and resume unchanged.
Finish or cancel runs and wait for preparations to finish before retrying.
Saved results remain in the same store. A stopped worker, changed
credentials or changed launch settings still require explicit reconciliation.
The container runtime must support pausing for this image upgrade.

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

Open **Dashboard → Evaluation → Create evaluation** and choose benchmarks,
an evaluation size, models or recipes, and limits. Benchmarks do not need to be
downloaded first. **Review plan** automatically reuses verified data and prepares
missing datasets and their supported data dependencies. The creation page shows
progress through installation, download and freezing, then presents the frozen
plan. **Start evaluation** begins model requests only after that review.

For independent data management, **Datasets → Prepare dataset** uses the same
shared worker. Accepted preparation jobs continue if the page closes. Completed
datasets become available to both Dashboard and CLI.

The CLI uses the same service operation by default. It waits for completion and
writes the frozen manifest to standard output, so it can still be redirected to
a file. `--url` selects a remote worker; downloads and frozen dataset files remain
on that worker, not on the CLI host.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --benchmark gpqa-diamond \
  --profile quick > quick-dataset.json
```

Repeating `--benchmark` creates one service-owned collection job. It pins eligible
existing sources, prepares only missing benchmarks with the same seed, and
validates the final composition. Source or seed conflicts require explicit
resolution; they are not treated as missing data. A failed job can be retried
explicitly and can reuse verified completed items. Single-benchmark preparation
and explicit `dataset combine` remain available.

Use `--no-wait` to return the preparation job immediately. Closing the page or
interrupting the CLI wait does not cancel the worker's preparation. Both clients
can inspect the same jobs and prepared datasets:

```bash
vllm-sr benchmark --url http://127.0.0.1:8090 dataset options
vllm-sr benchmark --url http://127.0.0.1:8090 dataset prepare \
  --benchmark simpleqa-verified --profile smoke --no-wait
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations PREPARATION_ID
vllm-sr benchmark --url http://127.0.0.1:8090 dataset show
```

Preparation requires Evaluation write permission in Dashboard; reading options,
progress and datasets requires read permission. Read-only Dashboard mode disables
preparation, while benchmark and profile selection remain available for browsing.
If access could not be checked, use **Refresh access** to retry the settings and
account checks. Preparing data does not require model generation permission,
start an evaluation, or make model requests. Automatic dependency installation is limited to the
allowlisted data preparation packages. It does not install execution harnesses,
build sandbox images, or provision model servers. Those remain explicit worker
setup operations. Gated sources require access approval and the appropriate
Hugging Face credential in the **worker environment**; browser or local CLI
credentials are not uploaded. Installation and download failures remain visible
on the preparation job and can be retried explicitly after the cause is fixed.

Local file imports and advanced history options use the explicit `--local` mode.
For this mode, install `vllm-sr[bench]` on the preparation host for Parquet sources.
A local file is never implicitly uploaded to a remote worker, and `--local` cannot
be combined with `--url` or `SR_BENCH_URL`:

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile smoke \
  --source-path ./tasks.parquet --revision imported-v1
```

Local task imports require `--source-path` and `--revision`; their actual bytes
are hashed. `--limit` creates a labeled custom subset in either mode. Never edit
a prepared file in place. New questions, selection rules or source bytes create
a new identity.

### Reserve named evaluation history

For repeated evaluations, prepare native sources with `--source-partition` to
record their upstream partition and canonical task identity. This partition is
the source's `test`, `dev`, or other upstream task collection; it is independent
of sr-bench's evaluation split and seed. Use the same partition and exact source
provenance throughout a history comparison.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile quick --source-partition test > quick.json
# Use the dataset ID returned above; --dataset and --run may be repeated.
vllm-sr benchmark --store ./data/sr-bench dataset exclusions \
  --dataset DATASET_ID --run RUN_ID --output history.json
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile standard --source-partition test \
  --exclusion-snapshot history.json --evaluation-role holdout > standard.json
```

Snapshot compilation reads only explicitly named prepared datasets and frozen
run manifests from the selected local store. It reserves **all memberships**,
including planned or failed cases; membership does not establish that a model
generated a response or a person read it. The immutable snapshot contains task
identity hashes, reference digests, and source provenance, without question or
answer bodies. Named-reference reads are bounded; oversized inputs fail without
publishing a selection.

Standard keeps the existing deterministic ordering and excludes the union of
its original Quick membership and the frozen history **once**. Preparation
either produces the exact requested count or fails before publishing a dataset.
It never fills a shortfall with excluded tasks or changes the profile count.
The snapshot and per-family counts become part of the new dataset identity;
combining datasets and freezing plans preserve that provenance. Existing
artifacts are unchanged. Preparation without the new options retains its
original behavior and makes no additional history qualification.

This first identity policy requires exact source bytes, revision, normalizer and
upstream partition, with native task IDs (including the domain for τ³). GPQA
uses the full hash of its native, unformatted question within that source. Older
prepared artifacts without this identity, normalized imports, missing native
IDs, and cross-source or cross-revision mappings fail explicitly; aliases and
message hashes do not establish equivalence. They need separate provenance
reconciliation before they can support an exclusion claim.

Freeze `--evaluation-role retest` explicitly for a family that is being retested.
It can be used without a history snapshot and does not claim disjointness. A
snapshot, if supplied, still excludes its memberships; retest is never an
automatic fallback after exhaustion. An explicitly prepared GPQA retest can
remain in the default protocol with a retest disclosure. Its aggregate must
remain separate from any claimed unseen aggregate. The preparation role is
separate from the existing evaluation split label.

Named-history exclusion is a finite local provenance claim. It does not certify
complete browsing or human exposure history, or absence of upstream contamination.
Reports retain that limitation and identify explicit retest families; these
options do not introduce a new unseen-only scoring aggregate.

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

## Native output capacity

Set `output_policy: native` to explore each model's available output capacity
without a shared generation-token cap. Register `native_limits` on every selected
target, keyed by the physical response model, with verified `context_window` and
`max_output_tokens` values. Keep model-specific reasoning settings in
`request_params`. Omit `max_tokens` from both sampling and target overrides.

The single-model adapter uses the vLLM Chat render API to count the actual
prompt, then generates once with the smaller of the configured output capacity
and remaining context. A MoM recipe must use `request_params.default_max_tokens:
auto` on every reachable decision, without a smaller output limit. Its Router
response records the actual selected-model input and output budget. Missing or
inconsistent evidence stops qualification; the harness does not guess a budget
from generated token usage. Native mode currently requires physical response
identities to match selected model identities and one fully accounted dispatch.

The plan derives its evidence token ceiling from the frozen native limits. Allow
sufficient call/run time and output storage for that capacity; idle, cancellation
and repetition controls remain active. Model maximum output and total context are
different values, and normal end-of-answer stopping remains enabled. Native
capacity does not force a model to fill its context. Compare native candidates
against native baselines with the same model limits. Offline replay is unavailable
when equivalent per-call native budget evidence cannot be established.

## Answer grading

The MMLU-Pro and GPQA adapters use `sr-bench-mcq-final-v2`. Capability scoring
extracts an unambiguous answer from the visible final channel: a leading answer
line, an explicit answer declaration, or a boxed choice. Markdown emphasis does
not change the answer. Conflicting declarations and prose without an explicit
answer remain unparsed; the grader does not guess from isolated letters or use
hidden reasoning. This is a conservative sr-bench adaptation, not an exact
reproduction of the upstream extraction heuristics.

Strict answer-format compliance is reported separately from correctness.
Truncated responses still count as output failures. Adapter versions are frozen
in each plan, so different graders cannot silently share a comparison. Existing
run scores remain unchanged. `benchmark regrade RUN_ID` returns a separate,
versioned result from saved final answers without generating or rewriting them.

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
becomes zero or evidence of savings. **Quality only** (`capability_only`) disables
USD-budget stopping; request/case/run time, call and output limits still apply.
Known spend is recorded, but missing prices or usage do not support savings claims.

## Iterate with preview, replay and live evaluation

1. Start with smoke: preview routing, then run a bounded live smoke to check final
   answers, graders, identity receipts, accounting and cancellation.
2. Create an experiment and save a quick/dev single-model matrix once, together
   with the current MoM result. Reuse that frozen baseline for compatible iterations.
3. Make one coherent routing change. Use `config validate`, `config plan` and
   `config apply`; wait for the expected active revision. For restart-required
   changes, use the supported `serve --replace-active-config` flow.
   Update the registered MoM target's `config_hash` to the verified active hash;
   existing runs keep their frozen target definitions.
4. Derive a candidate plan from the saved baseline. It preserves the exact tasks,
   sampling, grader options and limits while selecting registered MoM targets.
   Preview those tasks, then inspect server-qualified replay options. Eligible
   replay reuses saved answers; other routes require live evaluation.
5. Evaluate promising candidates live on the same dev tasks and compare paired
   results. Freeze the chosen policy before the prespecified standard/holdout live
   comparison; do not tune against its failures.

```bash
vllm-sr benchmark experiment create "Routing quality and cost" --idempotency-key study-1
vllm-sr benchmark experiment attach EXPERIMENT_ID --run BASELINE_ID --role baseline
vllm-sr benchmark candidate-plan BASELINE_ID --target balance --mode preview \
  --experiment EXPERIMENT_ID > candidate-review.json
# Inspect the plan, then extract its frozen manifest for submission.
jq '.manifest' candidate-review.json > preview.json
vllm-sr benchmark preview --manifest preview.json --detach
vllm-sr benchmark replay-options --limit 10
vllm-sr benchmark replay-options BASELINE_ID --limit 10
vllm-sr benchmark replay --baseline BASELINE_ID --preview PREVIEW_ID
vllm-sr benchmark comparison-options --limit 10
vllm-sr benchmark comparison-options BASELINE_ID --limit 10
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
A config hash freezes configuration, not evolving Learning/session state. The
harness does not automatically isolate, reset or replay that state across live
candidates. Record the intended state conditions and treat uncontrolled live
state differences as a comparison limitation. Non-Learning selectors that depend
on telemetry can also return state-dependent snapshots.

With automatic output budgets, supported single-backend previews call the
provider's render API to resolve each candidate's input size and available output
capacity. Rendering does not generate an answer. Selection uses the configured
cost forecast, not the maximum output capacity as an expected token count.
Requests needing dynamic enrichment or overflow compression remain unresolved;
inspect `selection_status` and `selection_reason` before relying on a model choice.
These checks cover the supported preview envelope and configured request policy,
not unsupported caller-specific generation fields.

For a single request, `vllm-sr route preview --request-file request.json` accepts
the Router's supported request envelope: role/content/tool-call messages, tools,
function selection, response format, output-budget fields, string metadata and
preview options/context. It is not an arbitrary Chat Completions request;
unsupported fields such as `temperature` and `stream` are rejected. Optional
`--session-id`, `--conversation-id` and `--sampling-seed` describe a read-only
preview; the seed does not fix a later live random draw. In benchmark cases,
explicit `request_metadata` is sent to the provider; benchmark `metadata`, which
may contain reference labels, is never forwarded as request metadata.

Replay rejects state-dependent Learning snapshots, unsupported plugin, agent
or compound execution and missing matrix cells instead of calling a model.
**Replay** and **Compare** list only baselines with at least one compatible
saved run, then offer only compatible choices. Discovery and submission use the
same authoritative validator; submission checks again. Both runs must contain
identical frozen cases and request protocols. Replay also checks actual saved
request inputs, deterministic selection, grader identity and exactly one complete
saved subject generation per selected cell. Case order alone may differ and
receives an explicit receipt. It never weakens frozen content hashes or
silently calls a model. Keep the same pending baseline/preview/idempotency key
after a lost response; do not create a new intent to reconcile it.

Compare accepts completed or failed live runs only when every planned cell has
an explicit terminal outcome. Failed outcomes count as incorrect in the full
planned denominator; missing outcomes and completed but ungraded answers block
comparison. Failed statuses and unknown costs remain visible. This measures
delivered quality under the frozen limits, including execution failures.

The read-only APIs are `GET /api/sr-bench/v1/replay-options` and
`GET /api/sr-bench/v1/comparison-options`. Omit `baseline_run_id` for baselines;
provide it for compatible previews or live candidates. Pages use `limit` (1–25)
and an opaque `after` cursor. CLI `benchmark replay-options` accepts that same
cursor. An empty page with `has_more: true` is an unfinished search: use **Load
more** or the next cursor. `scan_limited: true` means some evidence exceeded the
per-page validation limit; it does not prove that no other compatible runs exist.
A cursor becomes invalid when visible eligible evidence changes; refresh from
the first page. These queries make no model requests.

Experiments link baseline, initial, preview, estimate, candidate, validation and recovery
runs without changing their original receipts. Creating or attaching an experiment,
selecting replay options and deriving a candidate plan generate no model answers.
Experiment membership alone does not establish paired comparability.
A terminal full live baseline can supply a candidate plan's frozen protocol even
if it failed; this neither retries its generations nor changes its evidence.
Recovery children remain labeled as recovery attempts even when only one model
from a mixed baseline is selected. They never replace the full baseline.
Administrators can continue CLI-created experiments; other writers can only add
new attempts to their own experiments. Read-only users can compare accessible saved runs.

Delete a finished experiment from its Dashboard detail or with
`vllm-sr benchmark experiment delete EXPERIMENT_ID`. Deletion removes only the
group and its links; every run, result and artifact remains available. Active
linked runs block deletion. Retrying the same deletion returns its saved receipt;
a deleted experiment's creation key cannot recreate it.

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

Each target and benchmark also carries a `continuity` block. `multi_request_tasks`
counts tasks with two or more subject requests, such as agent and coding tasks;
each task's requests are read in stored order, without judge or simulator calls.
`model_switches` counts consecutive requests whose selected model differs, with
the per-task mean and maximum, and `decision_changed_tasks` counts tasks whose
routing decision changed. `switched_accuracy` and `unswitched_accuracy` score tasks
with and without a switch over `switched_tasks` and `unswitched_tasks`, with
failures counted as incorrect. A request without a selected model is unknown, not
a switch: switches are counted between the known selections around it, and
`unknown_model_requests` records it. A request that made several inference calls
under Fusion, Confidence, Workflows or fallback hides its own model sequence, so
its task is counted only in `multi_inference_tasks`. A switch is reported as a
fact, not a penalty. sr-bench sends no session identity, so these runs measure
routing without session state.

The full sr-bench score uses fixed benchmark weights: MMLU-Pro 10%, SimpleQA 10%,
GPQA 15%, HLE 15%, ARC 10%, LiveCodeBench 10%, SciCode 10%, Terminal-Bench 10% and
τ³ 10%. It requires all nine complete benchmarks. A subset macro result retains
its subset label and is not the full score.

Paired comparison selects the strongest observed single model by the same
aggregate over identical cases and records that selection.
Exact weighted-quality ties prefer the single with the lowest complete known
total cost, then a stable target ID. The report lists every tied-best single.
If any tied-best single has incomplete total cost, savings remain unknown.
Comparisons report **total cost savings** across subject and judge/simulator
calls, with **subject cost savings** shown separately. Both use
`100 × (1 − candidate cost / baseline cost)` with the same scope and complete,
compatible accounting. The API names these `total_cost_saving_percent` and
`subject_cost_saving_percent`; cache-neutral comparisons remain subject-only.
A small dev sample shows direction; a quality
non-inferiority claim needs a prespecified margin and a holdout confidence
interval. Token-equivalent self-hosted prices do not establish GPU invoice savings.

The default paired quality interval is a conservative weighted Hoeffding bound
for independent case differences. It stays nonzero when every observed pair ties,
including all-wrong samples. The stratified bootstrap interval is retained as a
diagnostic; a degenerate `[0, 0]` bootstrap from a small tied sample does not prove
equivalence. Neither interval includes selection of the strongest observed
baseline, tuning selection or dataset contamination uncertainty.

Before reserving a Standard holdout, exclude previously generated, inspected or
tuned-on cases by stable ID and input fingerprint. A different seed or a
`holdout` split label does not establish independence. Retests remain useful,
but report their exposure separately from unseen validation.

Dashboard opens on **Runs**, with filters for name/model, status and mode. Each
row shows the completed denominator, failures, persisted update time and target
kind. Read-only polling reconnects after a temporary network failure and discovers
CLI-created runs. Closing or refreshing the page does not restart a run.

In **Create evaluation**, first choose **smoke**, **quick** or **standard**, then
select one or more benchmarks, or **Select all benchmarks**. These are
the actual run profiles; standard uses the holdout split. Available sources must
share a profile, seed and split. **Review plan** reuses verified prepared groups
and automatically downloads missing groups and supported data dependencies in
one background preparation job. It then composes the frozen sources without
calling a model. Selecting all benchmarks from one source reuses its original
identity; a subset or multi-source composition creates a reusable frozen dataset.
Conflicting selections are rejected rather than silently merged.

Set sampling, budget and request/case limits with the form controls; no JSON
editing is required. Registered target settings override run defaults and remain
read-only. **Route preview** also accepts optional session and conversation
context for inspecting session-dependent routing. Review the frozen plan before
starting; plan review does not generate model answers.

**Datasets → Prepare dataset** is an optional management entry point that downloads
and freezes a built-in source through
the same service used by `benchmark dataset prepare`. Preparation progress and
errors survive page refreshes; completion refreshes the available datasets.
**Datasets** also provides search, profile/benchmark filters and pagination. Open a
dataset to browse its questions, benchmark coverage and subject groups. Questions
load in pages of 25 with benchmark/category filters and text search; opening one
shows the task instructions and choices, including complete code/agent task
inputs when the pinned source is available. Reference answers, hidden tests and
tool credentials are excluded. Source details and hashes are behind disclosure
controls. **Evaluate dataset** reuses the chosen data. Browsing public questions
does not establish that they are unseen; never use standard tasks for tuning.
Each profile's total questions is the sum across its prepared sets. Sets may
overlap, so this is not a count of unique questions or the selected run's denominator.

Run details separate **Results**, **Questions**, **Calls**, **Evidence** and
**Recipe**. Start with the aggregate results, then inspect individual responses,
accounting and frozen configuration as needed.

While a run is active, elapsed time continues updating even when no additional
question has finished. Active calls show their phase, elapsed time, latest
recorded response activity and received bytes. This activity helps distinguish a
long response from one that has stopped arriving; it does not establish answer
quality or billable token usage. Tokens and costs require a complete usage receipt.
Use `vllm-sr benchmark show RUN_ID --calls --active` to read the same activity
through the CLI; `--after` and `--limit` bound each page.

**Compare iterations** guides two choices: a live single-model baseline with
compatible saved outcomes, then any number of eligible candidate runs. Baselines
without a compatible candidate are excluded. Search narrows the candidate list;
**Select all** selects matching available runs across pages.
Changing the baseline clears the candidates, and changing any selection hides
previous comparison results until **Compare runs** is selected. The service still
validates every paired outcome before showing a comparison.

Candidates are ordered by creation time, and selections are preserved in the URL.
Quality/cost and iteration charts accompany paired confidence intervals, savings,
tokens, latency and wall time. Result cards are paginated; charts and CSV/JSON
exports retain all selected comparisons. The interface is not limited to two
tuning iterations. A positive estimate with an interval spanning zero is not proof
of a gain.

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

Calls and results fetch at most 100 rows at a time and display 25 rows per page;
full call bodies load on inspection. Search applies to loaded rows. Metrics and
routing distributions come from the complete report independently of loaded detail
pages. Recovery candidates, exclusions and child attempts are also paginated.

**Run events** shows human-readable saved activity, oldest first, with event-type
filters and 25 rows per page. Opening details fetches at most 1,000 events;
**Load more events** explicitly reads the next saved page. A full API page is
labelled as a loaded count because the endpoint does not provide a total.
Filtering applies only to loaded events. This is an event snapshot: **Refresh
evidence** loads a new snapshot while run progress continues polling independently.
A failed page read preserves its cursor and existing rows. Regrade and training
export reuse saved evidence without new model calls.

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
