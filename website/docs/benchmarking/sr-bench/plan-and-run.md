---
title: Plan and run
---

# Plan and run

A run starts from a frozen plan that fixes its dataset, targets, sampling and
limits. This page covers the manifest, the run commands and their limits,
native output capacity, and answer grading. It assumes a prepared dataset from
[Prepare reusable tasks and targets](./tasks-and-targets.md).

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

## Session-routing observations

The six native live session-routing schedules are opt-in custom cases. They do
not add a published benchmark weight or answer-quality score. Prepare a frozen
dataset from a repository checkout. Repeat `--scenario` for each schedule you
want; choices are `balanced`, `tool-heavy`, `frontier-heavy`, `idle-heavy`,
`stateful-heavy` and `drift-heavy`.

```bash
uv run --project src/vllm-sr python -m cli.sr_bench.session_routing \
  --store ~/.local/share/vllm-sr/sr-bench \
  --scenario balanced --scenario tool-heavy \
  --sessions 1 --turns 6 --max-tokens 64 --timeout 60 \
  --profile quick > session-routing-dataset.json
```

Copy the returned `path`, `sha256`, `profile` and `seed` fields into a normal
benchmark manifest. Match the frozen request settings and call limits:

```yaml
version: sr-bench-1.0
name: Session routing observations
mode: live
profile: quick
seed: 0
dataset:
  path: /absolute/path/from/the/preparation/output/cases.jsonl
  sha256: SHA256_FROM_PREPARATION
targets:
  - id: router
    kind: mom
    base_url: http://router.example:8000/v1
    model: vllm-sr/auto
    session_mode: session_aware
cost_policy: capability_only
sampling:
  temperature: 0
  top_p: 1
  max_tokens: 64
limits:
  max_calls_per_case: 6
  total_timeout_s: 60
```

Plan and run it with the standard commands above. The dataset freezes the
scenario messages and pauses. The report separates workload phases, phases
derived from sent request messages, and Router response phases. Missing model,
usage or cache evidence stays unknown. The `stateful-heavy` schedule labels
prompt phases; it does not send Responses API state. Retry and error-recovery
controls are rejected. An idle-boundary request reuses the same session ID, but
does not prove Router recovery or session retirement.

Each call has absolute and idle timeouts, output and repetition limits. Run and
task limits bound elapsed time and call counts. Priced runs reserve estimated
spend before dispatch and stop on reported actual spend. These are not a
provider-enforced universal hard USD cap: missing usage, inaccurate prices or
unsupported backend accounting can leave cost unknown. Unknown usage never
becomes zero or evidence of savings. **Quality only** (`capability_only`) disables
USD-budget stopping; request/case/run time, call and output limits still apply.
Known spend is recorded, but missing prices or usage do not support savings claims.

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

## Next

- [Iterate with preview, replay and live evaluation](./iterate.md)
