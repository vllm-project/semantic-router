---
title: Read the results
---

# Read the results

This page explains report metrics and scores, paired comparisons, the Dashboard
evaluation pages, and explicit recovery of unfinished work.

Reports show full planned denominators, scored/correct/failed counts, per-benchmark
accuracy and uncertainty; four exclusive token buckets; subject and judge/simulator
costs; TTFT and latency percentiles; request-time sum and actual wall time. Missing
metrics remain null. Failed and partial runs remain visible and do not qualify as
completed evaluations.

An explicitly incomplete MoM usage receipt with no call entries means that
accounting is unknown, not that no inference occurred. Usage, cost and inference
count remain null, even if a single-call header is also present. Capability-only
runs can still grade a complete answer; priced calls and explicit inference
bounds fail when the required evidence is unknown.

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
fact, not a penalty. Each continuity block reports the target's `session_mode`.
Targets are `stateless` by default. A target with `session_mode: session_aware`
sends an opaque `x-session-id` on every subject call in a live run. The value
stays the same across calls in one case and differs across runs, cases and
targets. Judge and simulator calls never carry the header. Subject call receipts
retain the value as `session_id`. The Router session policy itself must be
enabled in its configuration. Preview and replay reports remain `stateless`. To
compare both modes on the same tasks, include a target of each mode against the
same Router.

A selected-model response header takes precedence over the model named in the
stream. Successful receipts record `provider_model_observed` separately from
the requested model used for request and pricing identity. If neither the header
nor the stream names a model, the selection stays unknown. Reports keep the
legacy fallback only for older receipts without this observation field.

Subject call receipts include `phase` and `phase_source`. For MoM targets,
sr-bench requests the Router debug response headers to read its session policy
phase. A `router` source means the response contained `x-vsr-session-phase`; a
`request` source means sr-bench derived `tool_loop` or `user_turn` from the
latest messages because that header was absent. An unsupported nonempty Router
phase is stored as `unknown` with `phase_source: "router"`.
`model_switches_by_phase` counts a switch under the phase of the later request.
Provider cache fields retain their presence separately from their numeric values:
`cache_read_reported` and `cache_write_reported` are true only when the provider
sent the corresponding field, including an explicit zero. The target and
benchmark metrics expose `cache_read_ratio` and `cache_read_call_count`; only
subject calls with `cache_read_reported=true` are included, so missing cache
usage stays unknown instead of becoming a zero-hit observation. Older stored
runs remain readable: missing phases group under `unknown`, and cache-read
metrics report a null ratio with a zero sample count. For streamed calls,
`cache_read_usage` retains the normalized usage event that reported cache reads;
later events that omit cache fields do not erase that observation. If that event
lacks normalized prompt usage, its sample is counted but its ratio remains null
because its denominator is unknown.

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

## Cache-observation diagnostics

`cache-token-probe-v1` declares `cache-observation-v1`. A completed result means the
repeat sequence finished through the bounded transport. It has null answer,
correctness and score fields; it is counted as `observed`, not `scored`.
Threshold failure is separate from execution failure.

The `cache_observation` block contains the frozen `native_probes`, configured
target identity and every planned repeat. Each repeat links its call ID, session,
HTTP `response_status`, transport status, output completion, observed model and
config acknowledgement. Undispatched repeats have no invented response or token
values. Requested models and session IDs sent are distinct from Router response
evidence.

Call receipts retain a `provider-cache-observation-v1` observation with the
selected source, present fields, qualification and usage-event index. Full call
detail and bounded SSE files retain the raw evidence. The diagnostic recognizes
the two nested cached-token paths and the verified DeepSeek hit alias.
[DeepSeek's schema](https://api-docs.deepseek.com/api/create-chat-completion/)
defines that alias as the nested cached count and requires hit plus miss tokens
to equal prompt tokens. Duplicate fields must agree, and each count is used once.
The root `usage.cached_tokens` field remains unqualified legacy evidence.

Malformed, conflicting, incomplete and unqualified observations carry distinct
reasons. Earlier malformed/conflicting/unqualified events remain in `event_issues`
even if a later event supplies valid values. Repeated SSE usage snapshots are not
added together. A later event without cache fields does not erase the last
cache-bearing snapshot.

The full-cohort `cached_prompt_ratio` is available only when every planned repeat
completed and has qualified, complete usage. Its denominator includes fresh input,
cached input and cache-write tokens. Missing observations make that ratio null;
the report does not substitute the successful or reported-field subset. Generic
`cache_read_ratio` and billing retain their existing contracts and are separate
from this probe's response-facing observations. Final-response cache fields do
not establish cache reporting for hidden MoM child calls.

The frozen native thresholds produce pass/fail/incomplete predicates under
`validation`. Rates use the native four-decimal rounding and the cache ratio uses
six decimals. Zero-valued numeric thresholds are disabled, with
`enabled: false`; that does not turn an unknown measurement into zero. The default
minimum reporting state is `missing`, so the default gate only requires request
success. A success-only predicate can pass without any qualified cache evidence.
Cache-dependent predicates remain incomplete when they need malformed,
conflicting, unqualified or incomplete usage.

`success_rate` counts HTTP successes over all planned repeats. While outcomes are
missing it is an observed lower bound. Strict sequence completion also requires
complete output. The normal run command can therefore finish successfully even
when a configured observation threshold fails. Gate the report explicitly:

```bash
vllm-sr benchmark report RUN_ID > report.json
jq -e '.status == "completed" and
  ([.benchmarks[] | select(.result_kind == "cache-observation-v1")] |
   length > 0 and all(.[]; .cache_observation.validation.status == "pass"))' report.json
```

Cache and mixed cohorts have no combined quality, Wilson interval or published
sr-bench score. Dashboard labels their accuracy as **Not applicable**, independently
of execution status; observed cases do not become scored cases. Binary-only
comparison, replay, regrade and training export reject
them. Latency, URL differences and a stable session ID do not prove cache reuse,
model equivalence or savings. Local HTTP fixtures establish observation behavior;
real runtime/model qualification, downstream compatibility and owner review still
gate removal of the retained probe.

## Scripted agent-task diagnostics

`agent-task-chat-v1` declares `token-coverage-v1`. A task's credit is the native
case-insensitive substring match fraction, rounded to four decimals. For example,
three matched terms out of seven earn `0.4286`. These terms are required answer
strings, not tokenizer usage. Each completed result retains expected, matched and
missing terms plus an independent `full_credit` flag. Binary `correct` stays null.

The `token_coverage` report block names planned, graded, failed, cancelled and
pending task instances. `mean_credit` is the native four-decimal mean and appears
only when the entire planned cohort is graded. A failed or unattempted grade is
missing, not zero. `full_completion_rate` counts full-credit task completions over
all planned instances; for an incomplete run it is an observed lower bound, not
a success-only quality estimate.

Scripted phases stay in `scripted_phase_counts` and each turn's `scripted_phase`.
`observed_router_phase_counts` includes only Router response evidence;
`derived_request_phase_counts` identifies the request-based fallback. Each turn
links its call receipt, session ID, observed model, finish reason and error.
Unattempted turns remain visible. `scripted_tool_loop_switches` counts observed
model changes at scripted tool-loop turns, without claiming that tools ran.
The separate `continuity` block retains request-level model and decision changes.

These cohorts have no binary accuracy, Wilson interval, macro accuracy or
published sr-bench score. Mixed runs expose each benchmark separately and have no
combined quality score. Binary-only compare, replay, regrade and training export
reject this result kind until those protocols are defined for it. Read the report
JSON for these metrics; this slice does not add Dashboard charts.

Healthy, fully graded cohorts preserve the native metrics. Streaming,
visible-final-only grading, strict output limits and stopping after unexpected
transport failures can change incomplete legacy results. Controlled HTTP fixtures
establish transport and reporting behavior, not model quality or permission to
retire the retained runner.

## Weighted rubric diagnostics

DRACO uses the versioned `weighted-rubric-v1` result kind. Its
`weighted_rubric` report block retains planned, graded, failed, cancelled and
pending counts, with separate domain summaries. Per-case results retain signed
totals, positive ceilings, positive rewards, negative penalties and criterion
verdicts. A valid normalized grade remains clamped to [0, 1]; a negative total
can therefore produce a valid zero.

A missing or failed grade is not zero. Invalid rubric denominators, malformed
judge output and incomplete generations fail explicitly, and the weighted mean
remains null unless every case in that cohort is graded. The legacy comparison's
intersection of successful cases is a different denominator and is not used here.

Weighted cohorts have no binary correctness, accuracy, Wilson interval or
published sr-bench score. Mixed runs retain each binary benchmark's existing
metrics and each weighted benchmark's separate report; they have no combined
macro accuracy. Call counts, costs and completion counts still cover all tasks.
Read the report JSON for weighted metrics; this slice does not add Dashboard
charts for them.

Binary-only compare, replay, regrade and training export reject weighted or mixed
runs. The first DRACO slice covers final-answer execution and reporting, not
paired policy efficacy. Controlled fixture results do not establish population
quality or authorize legacy retirement; those require real-corpus, pinned-judge
qualification and the area's owner review.

## Use the Dashboard

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

## Recover unfinished work explicitly

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
