---
title: Iterate with preview, replay and live evaluation
---

# Iterate with preview, replay and live evaluation

Tune routing on development tasks, then evaluate the chosen policy on a
disjoint holdout. This page covers the preview, replay and live evaluation
loop, experiments, and the read-only replay and comparison APIs.

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

## Next

- [Read the results](./results.md)
