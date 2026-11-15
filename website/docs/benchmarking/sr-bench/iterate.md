---
title: Iterate with preview, replay and live evaluation
---

# Iterate with preview, replay and live evaluation

Tune routing on development tasks, then evaluate the chosen policy on a
disjoint holdout. This page covers the preview, replay and live evaluation
loop, experiments, and the read-only replay and comparison APIs.

## The loop

1. **Start with smoke.** Preview routing, then run a bounded live smoke to
   check final answers, graders, identity receipts, accounting and
   cancellation.
2. **Save a baseline.** Create an experiment, and save a quick/dev
   single-model matrix once, with the current MoM result. Reuse that frozen
   baseline for compatible iterations.
3. **Make one coherent routing change.** Use `config validate`, `config plan`
   and `config apply`, and wait for the expected active revision; for
   restart-required changes, use the supported
   `serve --replace-active-config` flow. Then update the registered MoM
   target's `config_hash` to the verified active hash; existing runs keep
   their frozen target definitions.
4. **Derive a candidate plan** from the saved baseline. It keeps the exact
   tasks, sampling, grader options and limits, while selecting registered MoM
   targets. Preview those tasks, then inspect the server-qualified replay
   options: eligible replay reuses saved answers, and other routes need live
   evaluation.
5. **Evaluate promising candidates live** on the same dev tasks, and compare
   paired results. Freeze the chosen policy before the prespecified
   standard/holdout live comparison, and don't tune against its failures.

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

## Preview

Preview has no answer-quality score. With Learning enabled, model selection
runs against a read-only snapshot of active learning state:

- **What it returns:** a concrete model when selection is resolvable, the
  selection status and reason, and `selection_provenance`: config and state
  hashes, capture time, whether local sampling occurred, and its seed. The
  Dashboard shows these fields beside each preview case, and explains
  unresolved selections.
- **Read-only.** The snapshot doesn't update learning state, and a later live
  request can differ as state or sampling changes. Keep Learning enabled
  during this check; disabling it would test a different policy.
- **State isn't frozen.** A config hash freezes configuration, not evolving
  Learning or session state, and the harness doesn't isolate, reset or replay
  that state across live candidates. Record the intended state conditions,
  and treat uncontrolled live state differences as a comparison limitation.
  Non-Learning selectors that depend on telemetry can also return
  state-dependent snapshots.

With automatic output budgets, supported single-backend previews call the
provider's render API to resolve each candidate's input size and available
output capacity:

- **Rendering doesn't generate an answer.** Selection uses the configured cost
  forecast, not the maximum output capacity as an expected token count.
- **Unresolved requests.** Requests needing dynamic enrichment or overflow
  compression stay unresolved; check `selection_status` and
  `selection_reason` before relying on a model choice.
- **Scope.** These checks cover the supported preview envelope and the
  configured request policy, not unsupported caller-specific generation
  fields.

### Preview one request

For a single request, `vllm-sr route preview --request-file request.json`
accepts the Router's supported request envelope: role, content and tool-call
messages, tools, function selection, response format, output-budget fields,
string metadata and preview options or context.

- **Not an arbitrary Chat Completions request.** Unsupported fields such as
  `temperature` and `stream` are rejected.
- **Read-only context.** Optional `--session-id`, `--conversation-id` and
  `--sampling-seed` describe a read-only preview; the seed doesn't fix a later
  live random draw.
- **Metadata.** In benchmark cases, explicit `request_metadata` is sent to the
  provider. Benchmark `metadata`, which may contain reference labels, is never
  forwarded as request metadata.

## Replay and compare

Replay rejects state-dependent Learning snapshots, unsupported plugin, agent or
compound execution, and missing matrix cells, instead of calling a model.
**Replay** and **Compare** list only baselines with at least one compatible
saved run, then offer only compatible choices:

- **One validator.** Discovery and submission use the same authoritative
  validator, and submission checks again. Both runs must contain identical
  frozen cases and request protocols.
- **Replay checks more:** actual saved request inputs, deterministic
  selection, grader identity, and exactly one complete saved subject
  generation per selected cell. Only case order may differ, and that gets an
  explicit receipt.
- **Never weakened.** Replay never weakens frozen content hashes, or silently
  calls a model.
- **Lost responses.** Keep the same pending baseline, preview and idempotency
  key after a lost response; don't create a new intent to reconcile it.

Compare accepts completed or failed live runs only when every planned cell has
an explicit terminal outcome:

- **Failed outcomes** count as incorrect in the full planned denominator.
- **Missing outcomes,** and completed but ungraded answers, block comparison.
- **Visible limits.** Failed statuses and unknown costs stay visible, so the
  result measures delivered quality under the frozen limits, execution
  failures included.

### Option APIs

The read-only APIs are `GET /api/sr-bench/v1/replay-options` and
`GET /api/sr-bench/v1/comparison-options`. Omit `baseline_run_id` for
baselines; give it for compatible previews or live candidates. These queries
make no model requests.

- **Paging.** Pages use `limit` (1–25) and an opaque `after` cursor; CLI
  `benchmark replay-options` accepts the same cursor. An empty page with
  `has_more: true` is an unfinished search: use **Load more** or the next
  cursor.
- **`scan_limited: true`** means some evidence exceeded the per-page
  validation limit. It doesn't prove that no other compatible runs exist.
- **Invalid cursors.** A cursor becomes invalid when visible eligible evidence
  changes; refresh from the first page.

## Experiments

Experiments link baseline, initial, preview, estimate, candidate, validation
and recovery runs, without changing their original receipts. Creating or
attaching an experiment, selecting replay options and deriving a candidate
plan generate no model answers.

- **Membership isn't comparability.** Experiment membership alone doesn't
  establish paired comparability.
- **Failed baselines.** A terminal full live baseline can supply a candidate
  plan's frozen protocol even if it failed; that neither retries its
  generations nor changes its evidence.
- **Recovery children** stay labeled as recovery attempts, even when only one
  model from a mixed baseline is selected. They never replace the full
  baseline.
- **Access.** Administrators can continue CLI-created experiments; other
  writers can only add new attempts to their own experiments. Read-only users
  can compare accessible saved runs.

Delete a finished experiment from its Dashboard detail, or with
`vllm-sr benchmark experiment delete EXPERIMENT_ID`:

- **Only the group goes.** Deletion removes the group and its links; every
  run, result and artifact stays available.
- **Active runs block it.** Active linked runs block deletion.
- **Retries are safe.** Retrying the same deletion returns its saved receipt,
  and a deleted experiment's creation key can't recreate it.

## Claims, regrade and export

- **Claims.** Only live runs support measured capability and savings claims.
- **Offline regrade** supports saved multiple-choice and grid final answers.
  It keeps the original results and makes zero model calls.
- **Export** is limited to explicitly marked development rows; holdout and
  unknown splits are rejected. Export doesn't start training.

## Next

- [Read the results](./results.md)
