# Router Replay

## Overview

Use Router Replay to inspect requests in Dashboard Insights: the selected route,
model, token usage, response, and tool trajectory. Configure storage in
`global.services.router_replay`, and the switch and capture defaults in a
`router_replay` plugin in `global.plugins`; use a recipe's or a decision's
`router_replay` plugin to override only the capture fields that differ.

## What Problem Does It Solve?

Replay keeps request-level evidence together so you can understand a routing
choice and follow a conversation across requests.

## When to Use

Enable Replay when you need request history for inspection or troubleshooting.
Choose retention and capture limits that match the data you are allowed to keep.

## Configuration

### Keep records across restarts

All five built-in MoM recipes, including Vault, enable PostgreSQL Replay by
default. Local `vllm-sr serve` manages PostgreSQL and its persistent volume.
For another deployment method, configure the PostgreSQL connection in the
[shared Replay service](../learning/memory-and-replay#configuration).

The generic `memory` store is useful for temporary inspection but loses records
when the Router restarts or reloads configuration. Use PostgreSQL or Redis when
you need durable history. Set retention for the data you intend to keep.

### Set shared capture defaults

```yaml
global:
  services:
    router_replay:
      store_backend: postgres
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        prompt_capture: none
        capture_request_body: true
        capture_response_body: true
        capture_tool_results: false
        max_records: 10000
        max_body_bytes: 4096
        max_tool_trace_steps: 100
```

Without a `router_replay` plugin, Replay stays disabled. The plugin in
`global.plugins` holds the shared capture defaults; a recipe's plugin
overrides them, and a decision's plugin overrides its recipe's, field by
field. Storage, retention and asynchronous writes are global service settings.

| Capture setting | Default |
| --- | --- |
| `prompt_capture` | `none`: routing data only |
| Request and response bodies | captured, under `raw` |
| Tool results | redacted |
| `max_records` | 10,000 |
| `max_body_bytes` | 4,096 bytes per body |
| `max_prompt_bytes` | unset: the prompt keeps `max_tool_trace_bytes`'s bound when that is set, else 65,536 bytes (64 KiB), under `raw` and `verified` |
| `max_tool_trace_bytes` | unset: 65,536 bytes (64 KiB) per tool text field, raw copies included, under `raw` |
| `max_tool_trace_total_bytes` | unset: 262,144 bytes (256 KiB) of trace steps per record, while `max_tool_trace_bytes` is unset too |
| `max_tool_trace_steps` | 100 |

### Choose what a record keeps

`prompt_capture` decides what a record keeps besides routing data: the
decision, model, signals, scores, timing, token usage and ids.

| Mode | A record keeps | Reads the prompt for personal data |
| --- | --- | --- |
| `none` (the default) | Routing data only | No |
| `raw` | The request and response text the capture switches allow, as it came | No |
| `verified` | The prompt alone, once the recipe's `pii` rules read it whole and find no personal data | Yes, on the decision model |

- **`none`** keeps no request or response text, whatever
  `capture_request_body`, `capture_response_body`, `capture_tool_results` and
  the shadow’s `capture.enabled` say, and reads nothing for personal
  data, so Replay adds no work for the decision model.
- **`raw`** keeps what the capture switches allow without checking it, so
  records can hold secrets and personal data. Restrict who reads them with
  `replay.read` and `replay.detail`, and choose a retention. A record keeps
  the prompt up to [its limit](#bound-the-prompt), 64 KiB unless
  `max_prompt_bytes` or `max_tool_trace_bytes` is set. The request's tool
  trace also keeps each user turn, within [the trace's limits](#bound-the-trace):
  64 KiB per field and 256 KiB of steps per record unless you set them.
- **`verified`** is described in
  [Keep personal data out of Replay](#keep-personal-data-out-of-replay).

`capture_personal_data` is the deprecated form of this setting. The Router
loads it as `prompt_capture`, with a `renamed_field` warning, and
`vllm-sr config migrate` rewrites it:

| Written before | Loads as |
| --- | --- |
| `capture_personal_data: false` | `prompt_capture: verified` |
| `capture_personal_data: true` | `prompt_capture: raw` |
| Neither field | `prompt_capture: none`: records keep routing data only. Set `prompt_capture: raw` to keep the text such a configuration kept before. |

### Limit capture on a decision

Add this fragment to the decision's `plugins` to record bounded request
excerpts and no response text:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      prompt_capture: raw
      capture_request_body: true
      capture_response_body: false
      max_body_bytes: 4096
      max_tool_trace_bytes: 4096
      max_tool_trace_steps: 100
```

How the limits count:

- **UTF-8 bytes,** for bodies and structured text. Excerpts end on complete
  characters, so they can be slightly shorter than the limit.
- **A truncated raw body** may no longer be complete JSON; check its
  truncation flag.
- **The prompt** has a limit of its own, `max_prompt_bytes`, and keeps
  `max_tool_trace_bytes`'s bound where no layer sets it; see
  [Bound the prompt](#bound-the-prompt). In the fragment above it keeps 4,096
  bytes, from `max_tool_trace_bytes`. A cut prompt has
  `prompt_truncated: true`, with the whole prompt's size in
  `prompt_original_bytes` and its SHA-256 in `prompt_original_sha256`, so the
  record still identifies it.
- **Tool traces** have their own limits, separate from the request and
  response bodies. `max_tool_trace_bytes` bounds each structured text field:
  tool definitions, and each trace step's text, arguments and output.
  `max_tool_trace_total_bytes` bounds all of a record's steps together. A cut
  step has `truncated: true`. Set as in the fragment above,
  `max_tool_trace_bytes` keeps the steps' raw copies whole and the trace
  without a total; see [Bound the trace](#bound-the-trace).

What the body switches cover, under `raw` and `verified` (under `none` a
record keeps no text at all):

- **`capture_request_body: false`** keeps all request text out of the record:
  the body, the prompt, tool definitions, and each request step of the tool
  trace, including user turns, tool arguments and tool results.
- **`capture_response_body: false`** does the same for the response: the
  body, the answer and tool calls in the trace, hallucination span excerpts,
  and the provider replies of failed fallback attempts, which keep their
  status with a fixed reason.
- **Kept either way:** step types, tool names, call IDs, result statuses and
  routing metadata. Text left out this way is absent rather than marked
  `content_redacted`, so the Dashboard shows it as not captured.

How the layers combine:

- **Opt out one decision** with `configuration: {enabled: false}`.
- **Change the deployment-wide default** by setting `enabled: false` on the
  `router_replay` plugin in `global.plugins`, or leaving that plugin out. A
  recipe or decision can still opt in.
- **Omitted fields inherit** from the layer below. An explicit `false` is an
  override, and `max_prompt_bytes: 0`, `max_tool_trace_bytes: 0`,
  `max_tool_trace_total_bytes: 0` or `max_tool_trace_steps: 0` removes that
  limit. Where no layer sets `max_prompt_bytes`, the prompt's limit falls back
  as [Bound the prompt](#bound-the-prompt) describes, and where none sets
  `max_tool_trace_bytes`, the trace's defaults apply as
  [Bound the trace](#bound-the-trace) describes.
- **Before a decision is selected,** a rejected request uses its recipe's
  capture defaults; a decision's override applies once it is selected.

These settings don't change provider-side retention or delete existing
records.

:::note Zero values
Explicit `max_body_bytes: 0` falls back to 4,096 bytes, and `max_records: 0`
to 200 records for the memory store. `max_prompt_bytes: 0`, like the tool-trace
limits, removes the limit instead. These zeros override an inherited value,
so omit a field to inherit it.
:::

### Bound the prompt

Under `raw` and `verified` a record keeps the prompt up to a limit, the first
of these that applies:

1. **`max_prompt_bytes`** from the most specific layer that sets it: the
   decision, then its recipe, then `global.plugins`. An explicit value wins
   over the fallbacks below, even over a `max_tool_trace_bytes` set on a more
   specific layer, and `0` keeps the whole prompt.
2. **`max_tool_trace_bytes`**, as the layers resolve it, when no layer sets
   `max_prompt_bytes` and it is above `0`. It bounded the prompt before
   `max_prompt_bytes` existed, so such a configuration keeps its bound.
3. **65,536 bytes (64 KiB)** otherwise.

To bound the prompt apart from tool traces, set both:

```yaml
max_tool_trace_bytes: 262144 # tool definitions and trace steps
max_prompt_bytes: 65536 # the prompt; 0 keeps it whole
```

:::note Upgrading
`max_prompt_bytes` is new. Where no layer sets it:

- **A configuration with `max_tool_trace_bytes` above `0`** keeps the prompt
  bound it had.
- **A `raw` or `verified` configuration with neither limit**, or with
  `max_tool_trace_bytes: 0`, keeps the first 64 KiB of a longer prompt, where
  it kept the whole prompt. Set `max_prompt_bytes: 0` to keep whole prompts.
- **Under `none`** a record keeps no prompt, as before.

A shadow's `capture.enabled: true` now keeps its excerpt of the shadow's
answer under `raw` only; under `verified` the outcome keeps the answer's sizes
and SHA-256 alone. See [Replay and audit capture](#replay-and-audit-capture).
:::

### Bound the trace

Under `raw` the request trace keeps each user turn, tool call and tool result
as a step, up to `max_tool_trace_steps` of them, beside the answer's steps.
Where no layer sets `max_tool_trace_bytes`, two defaults bound it:

- **65,536 bytes (64 KiB) per text field:** the tool definitions, each step's
  text, arguments and output, and the step's raw copies of its arguments and
  output (`raw_arguments`, `raw_output`). A cut step has `truncated: true`, and
  `raw_truncated: true` when its raw copies were cut too.
- **262,144 bytes (256 KiB) of steps per record**, raw copies included
  (`max_tool_trace_total_bytes`). Over it the oldest steps are dropped first,
  as `max_tool_trace_steps` drops them, and the trace has
  `steps_truncated: true` with the number dropped in `dropped_step_count`. The
  newest step is always kept.

A record then keeps at most about 64 KiB of prompt, 64 KiB of tool
definitions, 320 KiB of trace (one step's five fields, should a single step
outgrow the total) and 4 KiB per body: under 0.5 MiB, so 10,000 records in the
memory store stay under 5 GiB.

A `max_tool_trace_bytes` set on any layer, `0` included, keeps the trace as it
was before these defaults: that limit on each step's text, arguments and
output, raw copies whole for readers that replay a call byte for byte, and no
total unless `max_tool_trace_total_bytes` is set too.
`max_tool_trace_total_bytes` set on any layer applies either way, and `0` keeps
every step. The prompt follows only a `max_tool_trace_bytes` a layer sets,
never these defaults.

:::note Upgrading
These defaults are new:

- **A configuration that sets `max_tool_trace_bytes`**, `0` included, keeps its
  traces as before.
- **A `raw` configuration that sets neither limit** keeps each tool text field
  to 64 KiB, raw copies included, and a record's steps to 256 KiB, where it kept
  them whole. Set `max_tool_trace_bytes: 0` to keep whole traces.
- **Under `verified` and `none`** a record's trace keeps no text, as before, so
  nothing changes.

`vllm-sr config migrate` writes no limit: a configuration that relies on whole
traces sets `max_tool_trace_bytes: 0` itself.
:::

### Keep personal data out of Replay

Set `prompt_capture: verified` globally or on a decision's `router_replay`
plugin to keep each request's prompt only when it has no personal data:

- **Kept:** routing evidence, the model, signals and the detected PII types,
  and the prompt once it is found clean, up to [its limit](#bound-the-prompt).
- **Omitted:** request and response bodies, tool definitions and tool traces,
  and the prompt of a request with personal data.
- **Requests never wait for the check.** Unless a routing decision already
  read the prompt with a PII signal, the record is first written without it.
  The recipe's configured PII rules then read the latest user message alone,
  and the prompt is added to the record once that read finds no personal
  data. At most two such reads run at once, each for up to 30 seconds. A read
  that could not finish in that time does not start, and the prompt stays
  out, as it would when the read ran out of time. That covers a message
  longer than the decision model reads in 30 seconds at its measured pace
  and, until that pace is measured, one whose guard scan ran out of time
  during routing.

When PII evidence is unavailable, the `pii` rule's model gives no answer, or
the recipe has no `pii` rule, capture omits the prompt conservatively. A PII
signal is optional; configuring one allows the prompt of requests it identifies
as free of personal data. These reads run on the decision model beside the
guard, so `verified` costs decision-model time that `none` and `raw` don't.
For example:

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
```

### Capture tool results for development and audit

By default a record keeps each client tool result's call ID and whether it
succeeded or failed, but not its output, and a captured request body has its
tool results removed. This applies whether or not a PII signal reads tool
results. To debug an agent's tool loop, set `capture_tool_results: true` with
`prompt_capture: raw` globally or on a decision's `router_replay` plugin:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      prompt_capture: raw
      capture_request_body: true
      capture_tool_results: true
      max_tool_trace_bytes: 16384
```

With it on:

- **Each `client_tool_result` step** stores the result's text in `output`,
  with `content_redacted: false`.
- **A captured request body** keeps the tool results, up to `max_body_bytes`.
- **`max_tool_trace_bytes`** bounds each stored result, and sets the step's
  `truncated` flag when it cuts one.
- **`capture_request_body: false`** still stores no tool results, because they
  are request text.

:::tip Set a limit
Where no layer sets `max_tool_trace_bytes`, each result keeps 64 KiB and a
record's steps 256 KiB in all ([Bound the trace](#bound-the-trace)). With
`max_tool_trace_bytes: 0`, results are stored whole. A large tool output makes
the record larger, and a store can reject a record above its size limit.
:::

Tool results are raw client data, so:

- **`capture_tool_results: true` stores results only under
  `prompt_capture: raw`.** The Router refuses to load a configuration where
  the global defaults, or any decision's effective `router_replay` settings,
  combine `capture_tool_results: true` with `prompt_capture: verified`; under
  `none` it stores nothing.
- **A masking route** never records its request trace or body.
- **Principals without `replay.detail`** still receive tool results redacted.

Keep the option off in production unless the records are an intended audit
store.

:::caution
Request bodies, responses and tool traces can contain secrets or personal
data. Capture only what you need, and restrict access with `replay.read` and
`replay.detail`. See the
[Replay API and privacy controls](../../api/router#router-replay).
:::

## Shadow

The `shadow` block sends a bounded, sampled copy of the decision's approved
request to a group of candidate models, and keeps each outcome on the replay
record. The primary response is never changed or delayed, and the shadow runs
only while the decision records replays.

Promoting a model needs evidence from production-shaped requests: offline
evaluation misses real prompts, and a gray release exposes users to an
unproven model. A shadow fills the gap. The primary model still answers every
request, and the candidate's result is stored as an immutable outcome you can
compare later.

What the shadow copy carries:

- **The approved request.** It starts from the same request the primary
  dispatch was built from, after every request plugin has run.
- **The shadow model's own encoding.** It is rendered through the same encode
  and provider-adaptation steps, so reasoning controls follow the shadow
  model's family. It is addressed like the primary would address that backend,
  including a profile's `chat_path` and an Azure OpenAI `api_version` query.
- **No streaming.** It is always sent non-streaming.
- **Trace context.** W3C `traceparent` and `tracestate` are propagated. Client
  headers, including `baggage`, are never copied.
- **Decision headers.** `forward_headers` allows specific decision header
  mutations. Credential headers stay on the primary path. Each candidate uses
  its own backend credentials and TLS settings.
- **No redirects.** The call goes only to the configured backend address; a
  redirect answer is never followed.

Use a shadow when:

- a candidate model should be evaluated on live traffic before gray release
- the primary response must stay identical whether the shadow succeeds, fails, or times out
- resource use for the observation must be explicit and bounded per route
- replay or audit tooling needs to join primary and shadow observations by request identity

Do not use it to influence the live request, to grade output quality, or to
build a training set. The shadow records outcomes only.

:::note
The `shadow` block replaces the `shadow_dispatch` plugin;
`vllm-sr config migrate` rewrites one into the other.
:::

### Configure the shadow

Declare `shadow` on the decision's Replay plugin. The parent `enabled` switch
controls recording and dispatch; `shadow` has no separate `enabled` field.
Every declared shadow block is validated, even with Replay disabled or
`sample_rate: 0`.

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      shadow:
        models: [candidate-a, candidate-b]
        sample_rate: 0.05
        execution:
          max_concurrency: 2
          max_queue_size: 8
          timeout: 30s
        budget:
          max_response_bytes: 2097152
        capture:
          enabled: false
          max_bytes: 8192
```

Here `N` is the number of configured models. All integer bounds must be positive.

| Field | Default | Meaning |
| --- | --- | --- |
| `models` | required | 1–8 distinct, non-empty logical model names, each with a backend in `providers.models`. A single candidate also uses a list. |
| `sample_rate` | `1.0` | Optional fraction in `[0, 1]`. One sampling decision covers the group; `0` disables dispatch. |
| `execution.max_concurrency` | `2` | In-flight candidate HTTP calls, shared across requests for the same recipe and decision. |
| `execution.max_queue_size` | `8` | Pending request groups for the same recipe and decision, shared across requests. Counts groups, not candidates. |
| `execution.timeout` | `30s` | Shared group deadline, including queue wait and all candidate calls. Positive duration, at most `5m`. |
| `budget.max_response_bytes` | `N × 1048576` | Aggregate response byte budget, divided before dispatch among effective candidates. Minimum `2 × N`: each candidate reserves one byte to detect an oversized response. |
| `capture.enabled` | `false` | Store bounded excerpts of candidate text in replay outcomes under `prompt_capture: raw`. Otherwise only sizes, token usage, and a SHA-256 are kept. |
| `forward_headers` | `[]` | Decision header mutations to copy, matched case-insensitively. Credential headers are excluded. |
| `capture.max_bytes` | `N × 4096` | Aggregate excerpt byte limit, divided among effective candidates. |

The old flat shadow fields are rejected; use the nested structure above.

The primary model is excluded from the group. Each remaining candidate makes
at most one call, without retries. Candidates wait for concurrency slots under
the same deadline. Byte allowances are assigned before dispatch and are not
refunded on failure. Token usage and estimated cost remain observations, not
admission limits; missing usage is unknown.

Requests sampled out or left with no candidates produce metrics only. A
decision whose algorithm is an executor (`escalate`, `fusion`, `remom`,
`ratings` or `workflows`) rejects a shadow at config load, because the shadow
runs only on single-model provider dispatch.

### Fail-open behavior

The request path does one non-blocking group admission check and returns;
everything else runs in bounded workers after the primary dispatch response
has been built. The primary path never waits for candidate responses. Slow or
failed candidates leave the primary response unchanged. Each candidate ends
in one result:

| Result | Reasons |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`, `credential_unresolved`, `encode_failed`, `timeout`, `queue_timeout`, `canceled`, `router_closing`, `transport_error`, `upstream_status`, `redirect_rejected`, `response_too_large`, `malformed_response` |
| `dropped` | `queue_full`, `replay_queue_full`, `same_as_primary`, `internal_request`, `request_unavailable` |
| `sampled_out` | `sampled_out` |

- **Metrics.** Results and reasons are exported as
  `sr_shadow_dispatch_total{decision,result,reason}`, with
  `sr_shadow_dispatch_latency_seconds`, `sr_shadow_dispatch_inflight` and
  `sr_shadow_dispatch_queued`.
- **Drops caused by resource bounds** are reported through metrics and a
  structured `shadow_dispatch_dropped` event, not a replay write, so an
  overloaded shadow lane can't add load to the replay store.

Candidates reserve space in Replay's bounded result queue before calling a
model; a full queue produces `replay_queue_full`.

### Interpreting failures

A `failed` outcome does not always say something about the candidate model. Read the reason together with `status_code`, `attempts`, and the truncated `error` in the outcome metadata:

| Meaning | Reasons | How to read it |
| --- | --- | --- |
| Candidate rejected the request | `upstream_status` with a 4xx `status_code` | The shadow model could not accept the approved request, for example an unsupported parameter or a context window that is too small. Count it against the candidate. |
| Candidate health or capacity | `upstream_status` with a 5xx `status_code`, `timeout`, `transport_error` | The backend was unreachable, overloaded, or too slow within the group's `execution.timeout`. This measures the deployment, not answer quality. |
| Candidate tried to redirect | `redirect_rejected` with a 3xx `status_code` | The backend answered with a redirect. The router never follows it, so the prompt and the shadow credential only reached the configured backend. Point the shadow model at the backend's final address. |
| Candidate output problem | `malformed_response`, `response_too_large` | The backend answered but the body was not a valid response for its wire format or exceeded its share of `budget.max_response_bytes`. |
| Router-side, not about the candidate | `backend_unresolved`, `credential_unresolved`, `encode_failed` | The router could not build or address the shadow call. Fix the configuration and exclude these from any candidate comparison. |

`dropped` and `sampled_out` never reach the replay record. They mean the router chose not to send the shadow, so they carry no signal about the candidate model and are visible only in metrics and logs.

### Replay and audit capture

Each completed or failed candidate appends one outcome to the primary
request's replay record. Outcomes are append-only:

| Part | Holds |
| --- | --- |
| The outcome | `source: shadow_dispatch`, `target: model`, `target_ref: <shadow model>`, the verdict and the reason |
| Its metadata | Primary and shadow request identity; primary and shadow model and backend; decision and recipe; sample rate; enqueue, start and finish timestamps; queue wait and latency; attempts; status code; response size; stop reason; token counts; a SHA-256 of the shadow text |

Replay redaction applies to shadow outcomes as it does to the rest of the
record: viewers without content rights see the routing and timing fields, but
not `target_ref`, `reason` or `metadata`. Shadow excerpts require both `shadow.capture.enabled: true` and
`prompt_capture: raw`. Under `none` and `verified`, the shadow outcome keeps
its sizes and hash without the output text.
See the fragment:
[`config/fragments/plugin/router-replay/shadow.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/shadow.yaml).

### Export a comparison dataset

Captured outcomes become a comparison dataset through
`GET /api/v1/observability/replays/dataset`. The endpoint reads the records
the usual replay filters select, and answers with a versioned manifest of the
primary and shadow arms for each input.

A manifest is identified by its own digest, so the export takes the split plan
as part of the request:

- **`seed`** fixes the split assignment.
- **Each repeated `split`** is written as `name:weight`, for example
  `?seed=2026-q3&split=train:8&split=eval:2`.
- **Reproducible.** The same records under the same seed and splits rebuild
  the same manifest, including the split every example lands in, so later
  observations never move an example that was already placed.

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset?recipe=vault&seed=2026-q3&split=train:8&split=eval:2"
```

What the manifest holds:

- **No text.** It carries identity, output digests and lineage, never prompt
  or response text, so it can be published beside the numbers it supports.
- **Whole observations.** An observation enters whole or not at all: a request
  that failed, is unfinished, lost its input to truncation or never recorded a
  digest is left out, and counted under an exclusion reason. `counts` reports
  what was kept and what was dropped.
- **The whole selection.** A selection larger than 5000 records is refused,
  not exported as a page. Narrow the filters and export again.

One recipe or decision usually dominates live traffic, and a dataset built
from it reads as a statement about the whole router when it's a statement
about that decision. To balance it:

- **`balance_by` and `balance_max`** cap how much one group may contribute,
  grouping by `recipe`, `decision` or `primary_model`. Give both; one alone is
  refused.
- **The seed picks the rows a cap keeps,** not the clock, so a balanced
  dataset samples the traffic, not its arrival time.
- **Rows dropped for balance** are counted under `balance_cap`, like any other
  exclusion.

The export needs `replay.read`, and reads the same records the list API does.
The decisions being compared need `prompt_capture: raw` with request body
capture on, because an observation with no captured request is excluded as
`request_body_missing`.

### Hand the comparison to a judge

`GET /api/v1/observability/replays/dataset/judge-tasks` takes the same
selection and split plan as the export, plus a `blinding_key`. It answers with
pairwise tasks: one primary answer against one shadow answer, with the request
each was given.

- **Both orders.** Every pair appears twice with the sides swapped, so a judge
  that prefers whichever answer comes first shows up as disagreement between
  the two orders.
- **Blind labels.** Each side carries an opaque arm label that names neither
  the model nor which side is the primary, and the labels differ in every
  pair.
- **The key.** Only the `blinding_key` maps a label back to a model, so keep
  it away from the judge. It may not equal the manifest seed, which is
  published.

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset/judge-tasks?recipe=vault&seed=2026-q3&split=train:8&split=eval:2&blinding_key=$JUDGE_KEY"
```

A pair is built only when the request and both answers hash back to the
digests in the manifest:

- **The input digest** covers the request body without its top-level model
  field (`Model` in the recorded request, `model` in a client body), because a
  stored request names the model the Router sent it to. That body is also the
  `input` a judge reads, and each task carries its `input_digest`.
- **Versions.** The manifest is `shadow-dataset.v2` and the tasks are
  `shadow-judge-tasks.v2`. A `shadow-dataset.v1` manifest is refused.
- **Shadow text** comes from the captured excerpt, so the shadow decision
  needs `shadow.capture.enabled` on, and a `shadow.capture.max_bytes` large
  enough for every candidate's share to hold a whole answer.

Pairs that are left out are counted by reason:

| Reason | Cause |
| --- | --- |
| `arm_text_digest_mismatch` | An excerpt was cut, or a response-stage plugin rewrote the primary answer before it was stored. |
| `arm_text_missing` | The pair has no text at all. |
| `input_text_digest_mismatch` | The request doesn't match its digest. |
| `input_text_missing` | The request is missing. |

An answer that names its own model is marked `names_own_model`, because no
label can blind a judge to that.

The tasks carry prompt and answer text, so the route needs `replay.detail`,
and refuses a caller without it rather than redacting the tasks. The judge
runs outside the Router.

### Publish the dataset and its judgments

`tools/calibration/shadow-dataset` publishes a manifest, and optionally the
judgments a judge returned for it, into a storage directory; any object store
mounted as a filesystem works. The tool validates every input before it writes
anything:

- **The manifest** has to match its own digest.
- **The tasks** have to be the ones the manifest and the blinding key produce
  from the text they carry. The key is read for this check and never
  published.
- **The judgments** have to answer those tasks, with no field the judgment
  record doesn't define.

```bash
make run-shadow-dataset GO_TOOL_ARGS="--dest /mnt/shadow-datasets \
  --manifest $PWD/manifest.json --tasks $PWD/tasks.json --judgments $PWD/judgments.json \
  --blinding-key $JUDGE_KEY"
```

| What | Where it lands |
| --- | --- |
| The manifest | `manifests/<digest>/<digest of the file>.json` |
| The judgment set | `judgments/<manifest digest>/<digest of the set>.json` |
| The judgment report | A `.report.json` beside the set |

- **The manifest path** includes the file's own digest, because the manifest
  digest leaves out the counts, and two exports of the same examples can
  differ in what they excluded.
- **The report** counts outcomes, slot picks, and how many pairs followed the
  slot rather than the answer.
- **Content-addressed.** Every name comes from content, so publishing the same
  files twice changes nothing, and a published name is never replaced with
  different bytes.
- **Tasks stay private.** The judge tasks are read to check the judgments, but
  never published, because they carry the prompt and both answers.

Keep datasets out of this repository.

## Read the results

Open **Insights** in the Dashboard, or use the
[Replay API](../../api/router#router-replay) to filter records and inspect a session.
The search finds a record by its request id, its recipe or its session id, so
the session id an agent harness sends, such as Claude Code's
`x-claude-code-session-id`, finds every call of that session. A record's header
says where the request sits in its session, and the session routing history
opens with a summary: the agents that called, the models that answered and how
often, how many times the model switched, how the calls ended, and the time from
the first call's start to the last one's end. Below it each call is one row, the
request on the page marked and a model switch highlighted. Subagents are named
`subagent 1`, `subagent 2` and so on in order of their first call, with the
short hash of the agent identity header in the row's title, and the main
conversation is `main`. The hash is recorded only when `identity.headers.agent`
names that header (see [Session identification](../../api/session-identification));
without it, every call shows `main`. A Codex root thread, whose agent and session
ids are equal, is `main`. A viewer without `replay.detail` gets no hash, as with
the rest of the redacted session policy, so a subagent's call shows `hidden` there.
Costs are [estimates from recorded tokens and configured rates](../../api/router#configured-rate-cost-estimates),
not invoices. Missing prices and usage remain unknown.

Each record's `route_diagnostics` explains the route:

| Field | Holds |
| --- | --- |
| `decision_ranking` | Every record: the strategy, the winner's tier, whether the pool ranked by confidence and which decision stopped it, the `runner_up` with the key that separated them, and a `reason` with the values compared. It matches the eval API's report. |
| `prepared_dispatch` | Detail responses only: a versioned receipt for the exact provider-bound body of the primary dispatch, with its wire format, SHA-256 digest and byte length, but no copy of the body. Executor calls and fallback attempts are not part of it. |
| `looper` | Executor requests that keep an execution trace, such as `escalate`: bounded attempts, token and cost accounting, timings, disposition reasons and the final attempt. |

- `prepared_dispatch` is derived from content, so responses without
  `replay.detail` omit it.
- `looper` excludes prompts, tool arguments, credentials and raw errors. Its
  attempt details need `replay.detail`, and any truncation is marked.

For a complete decision fragment, see
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).
