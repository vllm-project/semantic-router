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
