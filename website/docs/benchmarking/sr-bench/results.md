---
title: Read the results
---

# Read the results

This page explains report metrics and scores, paired comparisons, the Dashboard
evaluation pages, and explicit recovery of unfinished work.

## Report metrics

Every report shows:

- **Denominators and counts:** the full planned denominators, and the scored,
  correct and failed counts.
- **Quality:** per-benchmark accuracy and uncertainty.
- **Tokens:** four exclusive token buckets.
- **Cost:** subject costs, and judge or simulator costs.
- **Time:** TTFT and latency percentiles, the request-time sum and the actual
  wall time.

Missing metrics stay null. Failed and partial runs stay visible, and don't
qualify as completed evaluations.

An explicitly incomplete `model_usage` receipt with no call entries in
`x-vsr-trace` means that accounting is unknown, not that no inference occurred.
Usage, cost and inference count remain null, even if the trace also declares
`inference_calls=1`. Capability-only runs can still grade a complete answer;
priced calls and explicit inference bounds fail when the required evidence is
unknown.

### Reconcile usage after a live run

For a terminal live run, `vllm-sr benchmark reconcile-usage RUN_ID` verifies
retained SSE usage and appends an idempotent, versioned accounting correction,
without model requests:

- **What changes.** Reports, comparisons and later exports use this derived
  accounting. The original call and result detail receipts, and the frozen
  manifest, stay unchanged.
- **What the report shows:** the correction hash, its time, the verified and
  changed counts, and the old and new known spend. Missing or conflicting
  evidence stays unknown.
- **What it can reprice:** single-model calls and proven direct MoM calls. A
  multi-call MoM can't be repriced from its final response stream alone.

### Cache-neutral cost

Reports also provide `cache_neutral_cost_usd`: a counterfactual,
token-equivalent subject cost that prices all prompt tokens at the frozen
fresh-input rate, plus output. Comparisons show its saving percentage against
the same selected baseline, beside the observed four-bucket costs.

Use both when sequential runs warm caches. The cache-neutral figure is not
billed spend, nor a measured cache-free execution.

### Continuity across a task's requests

Each target and benchmark also carries a `continuity` block. It reads each
task's subject requests in stored order, without judge or simulator calls:

| Field | Counts |
| --- | --- |
| `multi_request_tasks` | Tasks with two or more subject requests, such as agent and coding tasks |
| `model_switches` | Consecutive requests whose selected model differs, with the per-task mean and maximum |
| `model_switches_by_phase` | The same switches, each under the phase of the later request |
| `decision_changed_tasks` | Tasks whose routing decision changed |
| `switched_accuracy`, `unswitched_accuracy` | Accuracy of the tasks with and without a switch, over `switched_tasks` and `unswitched_tasks`; failures count as incorrect |
| `unknown_model_requests` | Requests without a selected model |
| `multi_inference_tasks` | Tasks with a request that made several inference calls, such as an executor's or a fallback's |

- **A switch is a fact, not a penalty.**
- **Unknown models aren't switches.** Switches are counted between the known
  selections around a request without a selected model.
- **Multi-inference tasks are counted only there,** because a request that
  made several inference calls hides its own model sequence.

### Session mode

Each continuity block reports the target's `session_mode`:

- **`stateless`** is the default.
- **`session_aware`** sends an opaque `x-vsr-session-id` and the same value as
  `x-conversation-id` on every subject call in a
  live run. The value stays the same across calls in one case, and differs
  across runs, cases and targets. Judge and simulator calls never carry the
  headers, and subject call receipts keep both identities as `session_id` and
  `conversation_id`.
- **The Router's session policy** must be enabled in its own configuration.
  Preview and replay reports stay `stateless`.

To compare both modes on the same tasks, include a target of each mode against
the same Router.

### Phases

Subject call receipts include `phase` and `phase_source`. For MoM targets,
sr-bench requests the Router's `x-vsr-trace` to read its session policy phase:

| `phase_source` | Means |
| --- | --- |
| `router` | The trace contained `session_phase`. An unsupported nonempty Router phase is stored as `unknown`. |
| `request` | The trace had none, so sr-bench derived `tool_loop` or `user_turn` from the latest messages. |

### Provider cache reads

Provider cache fields keep their presence separately from their numeric
values:

- **Reported flags.** `cache_read_reported` and `cache_write_reported` are
  true only when the provider sent the corresponding field, including an
  explicit zero.
- **Ratios.** The target and benchmark metrics expose `cache_read_ratio` and
  `cache_read_call_count` over only the subject calls with
  `cache_read_reported=true`, so missing cache usage stays unknown instead of
  becoming a zero-hit observation.
- **Streamed calls.** `cache_read_usage` keeps the normalized usage event that
  reported cache reads; later events that omit cache fields don't erase it. If
  that event lacks normalized prompt usage, its sample is counted but its ratio
  stays null, because its denominator is unknown.
- **Older runs** stay readable: missing phases group under `unknown`, and
  cache-read metrics report a null ratio with a zero sample count.

## Scores and comparisons

### The sr-bench score

The full sr-bench score uses fixed benchmark weights, and requires all nine
complete benchmarks:

| Benchmark | Weight |
| --- | --- |
| MMLU-Pro | 10% |
| SimpleQA | 10% |
| GPQA | 15% |
| HLE | 15% |
| ARC | 10% |
| LiveCodeBench | 10% |
| SciCode | 10% |
| Terminal-Bench | 10% |
| τ³ | 10% |

A subset macro result keeps its subset label, and is not the full score.

### Paired comparisons

Paired comparison selects the strongest observed single model, by the same
aggregate over identical cases, and records that selection:

- **Ties.** Exact weighted-quality ties prefer the single with the lowest
  complete known total cost, then a stable target ID. The report lists every
  tied-best single; if any of them has incomplete total cost, savings stay
  unknown.
- **Savings.** Comparisons report **total cost savings** across subject and
  judge or simulator calls, and **subject cost savings** separately. Both use
  `100 × (1 − candidate cost / baseline cost)`, with the same scope and
  complete, compatible accounting. The API names them
  `total_cost_saving_percent` and `subject_cost_saving_percent`; cache-neutral
  comparisons stay subject-only.

:::caution
A small dev sample shows direction. A quality non-inferiority claim needs a
prespecified margin and a holdout confidence interval, and token-equivalent
self-hosted prices don't establish GPU invoice savings.
:::

### Confidence intervals

- **The default paired quality interval** is a conservative weighted
  Hoeffding bound for independent case differences. It stays nonzero when
  every observed pair ties, including all-wrong samples.
- **The stratified bootstrap interval** is kept as a diagnostic. A degenerate
  `[0, 0]` bootstrap from a small tied sample doesn't prove equivalence.
- **Neither interval** includes the uncertainty from selecting the strongest
  observed baseline, from tuning selection, or from dataset contamination.

### Keep the holdout unseen

Before reserving a Standard holdout, exclude previously generated, inspected
or tuned-on cases by stable ID and input fingerprint. A different seed or a
`holdout` split label doesn't establish independence. Retests are still
useful, but report their exposure separately from unseen validation.

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

### Runs

Dashboard opens on **Runs**, with filters for name or model, status and mode.
Each row shows the completed denominator, failures, persisted update time and
target kind. Read-only polling reconnects after a temporary network failure
and discovers CLI-created runs; closing or refreshing the page doesn't restart
a run.

### Create an evaluation

In **Create evaluation**:

1. **Choose a profile:** **smoke**, **quick** or **standard**. These are the
   actual run profiles; standard uses the holdout split.
2. **Select benchmarks,** one or more, or **Select all benchmarks**. Available
   sources must share a profile, seed and split.
3. **Review plan.** It reuses verified prepared groups, downloads missing
   groups and supported data dependencies in one background preparation job,
   then composes the frozen sources without calling a model:
   - selecting all benchmarks from one source reuses its original identity;
   - a subset or multi-source composition creates a reusable frozen dataset;
   - conflicting selections are rejected rather than silently merged.
4. **Set sampling, budget and request or case limits** with the form
   controls; no JSON editing is needed. Registered target settings override
   run defaults and stay read-only, and **Route preview** also accepts
   optional session and conversation context, for inspecting
   session-dependent routing.
5. **Review the frozen plan** before starting. Plan review doesn't generate
   model answers.

### Datasets

**Datasets → Prepare dataset** is an optional management entry point that
downloads and freezes a built-in source, through the same service as
`benchmark dataset prepare`. Preparation progress and errors survive page
refreshes, and completion refreshes the available datasets.

- **Find a dataset** with search, profile and benchmark filters, and
  pagination.
- **Open a dataset** to browse its questions, benchmark coverage and subject
  groups. Questions load in pages of 25, with benchmark and category filters
  and text search. Opening one shows the task instructions and choices,
  including complete code and agent task inputs when the pinned source is
  available.
- **Never shown:** reference answers, hidden tests and tool credentials.
  Source details and hashes are behind disclosure controls.
- **Evaluate dataset** reuses the chosen data.
- **Question totals.** Each profile's total is the sum across its prepared
  sets. Sets may overlap, so it's not a count of unique questions, nor the
  selected run's denominator.

:::caution
Browsing public questions doesn't establish that they're unseen. Never use
standard tasks for tuning.
:::

### Run details

Run details separate **Results**, **Questions**, **Calls**, **Evidence** and
**Recipe**. Start with the aggregate results, then inspect individual
responses, accounting and frozen configuration as needed.

While a run is active:

- **Elapsed time** keeps updating, even when no further question has finished.
- **Active calls** show their phase, elapsed time, latest recorded response
  activity and received bytes. That tells a long response from one that has
  stopped arriving; it doesn't establish answer quality or billable token
  usage. Tokens and costs need a complete usage receipt.
- **From the CLI,** `vllm-sr benchmark show RUN_ID --calls --active` reads the
  same activity; `--after` and `--limit` bound each page.

### Compare iterations

**Compare iterations** guides two choices: a live single-model baseline with
compatible saved outcomes, then any number of eligible candidate runs.

- **Baselines** without a compatible candidate are excluded.
- **Candidates.** Search narrows the list, and **Select all** selects matching
  available runs across pages. Candidates are ordered by creation time, and
  selections are kept in the URL.
- **Changing a selection.** Changing the baseline clears the candidates, and
  changing any selection hides earlier comparison results until
  **Compare runs** is selected. The service still validates every paired
  outcome before showing a comparison.
- **Results.** Quality and cost, and iteration charts accompany paired
  confidence intervals, savings, tokens, latency and wall time. Result cards
  are paginated; charts and CSV and JSON exports keep all selected
  comparisons. The interface isn't limited to two tuning iterations.

:::note
A positive estimate with an interval spanning zero is not proof of a gain.
:::

### Frozen recipes

For a MoM target, the operator can register `capture_recipe: true` with its
`config_hash` and canonical `preview_url`. The worker then captures a redacted
recipe projection from the Router config API, but only when:

- the generated and active runtime hashes match the frozen target, before and
  after capture; and
- the source config ETag stays unchanged.

**Frozen recipes** displays and downloads that server-observed snapshot, its
capture time and projection hash; runtime calls separately acknowledge the
active config hash. Deployment wiring and secrets are omitted, so the download
is a recipe artifact, not a complete deployable configuration. Existing runs
without a snapshot show that it's unavailable, rather than borrowing a later
recipe.

### Paging and events

- **Calls and results** fetch at most 100 rows at a time and display 25 per
  page; full call bodies load on inspection. Search applies to loaded rows,
  while metrics and routing distributions come from the complete report.
  Recovery candidates, exclusions and child attempts are paginated too.
- **Run events** shows human-readable saved activity, oldest first, with
  event-type filters and 25 rows per page. Opening details fetches at most
  1,000 events, and **Load more events** reads the next saved page. A full API
  page is labelled as a loaded count, because the endpoint provides no total,
  and filtering applies only to loaded events.
- **Snapshots.** Events are a snapshot: **Refresh evidence** loads a new one,
  while run progress keeps polling independently. A failed page read keeps its
  cursor and existing rows.
- **No new calls.** Regrade and training export reuse saved evidence, without
  new model calls.

## Recover unfinished work explicitly

A stopped worker or a lost acknowledgement is not permission to retry
generation. Inspect saved dispatches and calls first. On a terminal run,
**Review recovery plan** separates two scopes:

- **Continue undispatched cases** selects cells that have never dispatched a
  model call.
- **Retry known failed cases as new attempts** requires explicit cell selection
  and acknowledgement of additional attempts and spend. Ambiguous calls, unknown
  billing, completed answers and already-claimed cells are excluded.

Both create a separate child run and keep the parent:

- **Scope.** Progress, denominator and costs belong to the child; parent spend
  stays separate in the lineage panel. A recovery child doesn't turn a partial
  parent into a completed benchmark.
- **Lost responses.** The Dashboard saves the exact pending recovery
  submission in the current tab, and reuses its idempotency key after a lost
  response.
- **Nothing automatic.** There's no automatic generation retry or worker
  restart.

```bash
vllm-sr benchmark recover-plan RUN_ID --mode undispatched --output recovery.json
# Review eligible/excluded cells. Optionally set selected_cells to a reviewed subset.
vllm-sr benchmark recover RUN_ID --plan recovery.json --idempotency-key recovery-1
```

For `--mode failed`, the final command also requires
`--acknowledge-new-attempt`. Reuse the same plan, cells and idempotency key
when reconciling an uncertain submission.
