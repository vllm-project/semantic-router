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
        capture_request_body: true
        capture_response_body: true
        capture_personal_data: false
        capture_tool_results: false
        max_records: 10000
        max_body_bytes: 4096
        max_tool_trace_bytes: 0
        max_tool_trace_steps: 100
```

Without a `router_replay` plugin, Replay stays disabled. The plugin in
`global.plugins` holds the shared capture defaults; a recipe's plugin
overrides them, and a decision's plugin overrides its recipe's, field by
field. Storage, retention and asynchronous writes are global service settings.

| Capture setting | Default |
| --- | --- |
| Request and response bodies | captured |
| Personal data | captured |
| Tool results | redacted |
| `max_records` | 10,000 |
| `max_body_bytes` | 4,096 bytes per body |
| `max_tool_trace_bytes` | no per-field limit |
| `max_tool_trace_steps` | 100 |

### Limit capture on a decision

Add this fragment to the decision's `plugins` to record bounded excerpts:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: true
      capture_response_body: false
      capture_personal_data: false
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

Body limits and structured-text limits count UTF-8 bytes. Excerpts end on
complete characters, so they can be slightly shorter than the limit. A truncated
raw body may no longer be complete JSON; check its truncation flag. Tool traces
have separate limits from raw request and response bodies.

- **Opt out one decision** with `configuration: {enabled: false}`.
- **Change the deployment-wide default** by setting `enabled: false` on the
  `router_replay` plugin in `global.plugins`, or leaving that plugin out. A
  recipe or decision can still opt in.
- **Omitted fields inherit** from the layer below. An explicit `false` is an
  override, and `max_tool_trace_bytes: 0` or `max_tool_trace_steps: 0`
  removes that limit.
- **Before a decision is selected,** a rejected request uses its recipe's
  capture defaults; a decision's override applies once it is selected.

These settings don't change provider-side retention or delete existing
records.

:::note Zero values
Explicit `max_body_bytes: 0` falls back to 4,096 bytes, and `max_records: 0`
to 200 records for the memory store. These zeros override an inherited value,
so omit a field to inherit it.
:::

### Keep personal data out of Replay

Set `capture_personal_data: false` globally or on a decision's `router_replay`
plugin. The record keeps routing evidence, model, signals and detected PII types,
while personal-data requests omit request and response bodies, prompts, tool
definitions and tool traces. The prompt is checked after routing, so requests
never wait for it: unless a routing decision already read it with a PII signal,
the record is written without the prompt, and the recipe's configured PII rules
then read the latest user message alone. The prompt is added to the record once
that read completes without finding personal data.

When PII evidence is unavailable, the `pii` rule's model gives no answer, or
the recipe has no `pii` rule, capture omits content conservatively. A PII signal is
optional; configuring one allows content capture for requests it identifies as
free of personal data. For example:

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
results. To debug an agent's tool loop, set `capture_tool_results: true`
globally or on a decision's `router_replay` plugin:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: true
      capture_tool_results: true
      max_tool_trace_bytes: 16384
```

Each `client_tool_result` step then stores the result's text in `output`, with
`content_redacted: false`. A captured request body keeps the tool results, up
to `max_body_bytes`. `max_tool_trace_bytes` bounds each stored result and sets
the step's `truncated` flag when it cuts one. With the default of `0`, results
are stored whole, so set a limit: a large tool output makes the record larger,
and a store can reject a record above its size limit.

Tool results are raw client data, so `capture_tool_results: true` requires
`capture_personal_data: true`. The Router refuses to load a configuration where
the global defaults, or any decision's effective `router_replay` settings,
combine `capture_tool_results: true` with `capture_personal_data: false`. A
masking route never records its request trace or body. Principals without
`replay.detail` still receive tool results redacted. Keep the option off in
production unless the records are an intended audit store.

Request bodies, responses, and tool traces can contain secrets or personal data.
Capture only what you need and restrict access with `replay.read` and
`replay.detail`. See the [Replay API and privacy controls](../../api/router#router-replay).

## Shadow

The `shadow` block sends a bounded, sampled copy of the decision's approved
request to a candidate model, and keeps the outcome on the replay record. The
primary response is never changed or delayed, and the shadow runs only while
the decision records replays.

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
- **Trace headers only.** Only the W3C `traceparent` and `tracestate` headers
  are propagated. Client headers, `baggage` included, are not.
- **No primary credentials.** Decision header mutations stay on the primary
  path unless `forward_headers` names them. Known credential carriers and
  each profile's `auth.header` are dropped even when listed, and only the
  Router's own static credentials for the shadow backend are used.
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

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      shadow:
        model: candidate-model
        sample_rate: 0.05
        max_concurrency: 2
        max_queue_depth: 8
        timeout_seconds: 30
        max_response_bytes: 1048576
        max_retries: 0
        capture_response_body: false
        max_capture_bytes: 4096
        tls_skip_verify: false
        forward_headers: []
```

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | Configured logical model that receives the shadow copy. Must have a backend in `providers.models`. |
| `sample_rate` | `1.0` | Fraction of eligible requests to shadow, in `[0, 1]`. `0` keeps the shadow declared but never dispatches. |
| `max_concurrency` | `2` | In-flight shadow calls for this decision. |
| `max_queue_depth` | `8` | Calls waiting for a slot. Anything beyond is dropped with reason `queue_full`. |
| `timeout_seconds` | `30` | Deadline for queue wait plus execution, shared by all retries. |
| `max_response_bytes` | `1048576` | Largest shadow response body that is read. Larger bodies fail with `response_too_large`. |
| `max_retries` | `0` | Extra attempts on transport errors or retryable statuses. Capped at `3`. |
| `capture_response_body` | `false` | Store a bounded excerpt of the shadow text in the outcome. Off by default; only sizes, tokens, and a SHA-256 are kept. |
| `max_capture_bytes` | `4096` | Excerpt bound when capture is on. |
| `tls_skip_verify` | `false` | Skip certificate verification for an https shadow backend signed by an internal CA. This setting applies only to the shadow HTTP client; it does not configure primary backend transport. |
| `forward_headers` | `[]` | Decision `header_mutation` names the shadow copy may carry, matched case-insensitively. Nothing else a decision sets for the primary backend is forwarded, so a custom credential such as `X-Internal-Token` stays on the primary path. Known credential carriers (`Authorization`, `Proxy-Authorization`, `Cookie`, `x-api-key`, `api-key`, `x-goog-api-key`, the `x-user-*-key` headers) are rejected at config load and dropped at run time even if listed. |

A shadow is skipped, with a metric but no outcome, when the request is
sampled out or when the primary dispatch already selected the shadow model. A
decision whose algorithm is an executor (`escalate`, `fusion`, `remom`,
`ratings` or `workflows`) rejects a shadow at config load, because the shadow
runs only on single-model provider dispatch.

### Fail-open behavior

The request path does one non-blocking slot check and returns. Everything else runs in a bounded worker after the primary dispatch response has been built. A slow, unavailable, malformed, or overloaded shadow endpoint cannot change the primary response or its latency. Every shadow ends in exactly one result:

| Result | Reasons |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`, `credential_unresolved`, `encode_failed`, `timeout`, `transport_error`, `upstream_status`, `redirect_rejected`, `response_too_large`, `malformed_response` |
| `dropped` | `queue_full`, `queue_timeout`, `router_closing`, `same_as_primary`, `internal_request`, `request_unavailable` |
| `sampled_out` | `sampled_out` |

Results and reasons are exported as `sr_shadow_dispatch_total{decision,result,reason}`, with `sr_shadow_dispatch_latency_seconds`, `sr_shadow_dispatch_inflight`, and `sr_shadow_dispatch_queued`. Drops caused by resource bounds are reported through metrics and a structured `shadow_dispatch_dropped` event rather than a replay write, so an overloaded shadow lane cannot amplify load on the replay store.

### Interpreting failures

A `failed` outcome does not always say something about the candidate model. Read the reason together with `status_code`, `attempts`, and the truncated `error` in the outcome metadata:

| Meaning | Reasons | How to read it |
| --- | --- | --- |
| Candidate rejected the request | `upstream_status` with a 4xx `status_code` | The shadow model could not accept the approved request, for example an unsupported parameter or a context window that is too small. Count it against the candidate. |
| Candidate health or capacity | `upstream_status` with a 5xx `status_code`, `timeout`, `transport_error` | The backend was unreachable, overloaded, or too slow within `timeout_seconds` after `max_retries`. This measures the deployment, not answer quality. |
| Candidate tried to redirect | `redirect_rejected` with a 3xx `status_code` | The backend answered with a redirect. The router never follows it, so the prompt and the shadow credential only reached the configured backend. Point the shadow model at the backend's final address. |
| Candidate output problem | `malformed_response`, `response_too_large` | The backend answered but the body was not a valid response for its wire format or exceeded `max_response_bytes`. |
| Router-side, not about the candidate | `backend_unresolved`, `credential_unresolved`, `encode_failed` | The router could not build or address the shadow call. Fix the configuration and exclude these from any candidate comparison. |

`dropped` and `sampled_out` never reach the replay record. They mean the router chose not to send the shadow, so they carry no signal about the candidate model and are visible only in metrics and logs.

### Replay and audit capture

Completed and failed shadows append one outcome to the primary request's replay record with `source: shadow_dispatch`, `target: model`, `target_ref: <shadow model>`, the verdict, and the reason. The outcome metadata carries primary and shadow request identity, primary and shadow model and backend, decision and recipe, sample rate, enqueue, start, and finish timestamps, queue wait and latency, attempts, status code, response size, stop reason, token counts, and a SHA-256 of the shadow text. Outcomes are append-only.

Replay redaction applies to shadow outcomes the same way it applies to the rest of the record: viewers without content rights see the routing and timing fields but not `target_ref`, `reason`, or `metadata`. Keep `capture_response_body` off unless the replay store and its readers are cleared for prompt-level content. See the fragment:
[`config/fragments/plugin/router-replay/shadow.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/shadow.yaml).

### Export a comparison dataset

Captured outcomes become a comparison dataset through `GET /api/v1/observability/replays/dataset`. The endpoint reads the records the usual replay filters select and answers with a versioned manifest of the primary and shadow arms for each input.

A manifest is identified by its own digest, so the export takes the split plan as part of the request: `seed` fixes split assignment and each repeated `split` is written as `name:weight`, for example `?seed=2026-q3&split=train:8&split=eval:2`. The same records under the same seed and splits rebuild the same manifest, including the split every example lands in, so adding later observations never moves an example that was already placed.

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset?recipe=vault&seed=2026-q3&split=train:8&split=eval:2"
```

The manifest carries identity, output digests, and lineage, never prompt or response text, so it can be published beside the numbers it supports. An observation enters whole or not at all: a request that failed, is unfinished, lost its input to truncation, or never recorded a digest is left out and counted under an exclusion reason, and `counts` reports what was kept and what was dropped. Because the manifest describes the whole selection it was built from, a selection larger than 5000 records is refused rather than exported as a page. Narrow the filters and export again.

One recipe or one decision usually dominates live traffic, and a dataset built from it reads as a statement about the whole router when it is a statement about that decision. `balance_by` and `balance_max` cap how much one group may contribute, grouping by `recipe`, `decision` or `primary_model`. The rows a cap keeps follow the seed rather than the clock, so a balanced dataset is a sample of the traffic and not of when it arrived, and rows dropped for balance are counted under `balance_cap` like any other exclusion. Both parameters are given together; one alone is refused.

The export needs `replay.read` and reads the same records the list API does. Body capture must be on for the decisions being compared, since an observation with no captured request is excluded as `request_body_missing`.

### Hand the comparison to a judge

`GET /api/v1/observability/replays/dataset/judge-tasks` takes the same selection and split plan as the export, plus a `blinding_key`, and answers with pairwise tasks: one primary answer against one shadow answer, with the request each was given. Every pair appears twice with the sides swapped, so a judge that prefers whichever answer comes first shows up as disagreement between the two orders. Each side carries an opaque arm label that names neither the model nor which side is the primary, and the labels differ in every pair. Only the key maps a label back to a model, so keep it away from the judge. It may not equal the manifest seed, which is published.

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset/judge-tasks?recipe=vault&seed=2026-q3&split=train:8&split=eval:2&blinding_key=$JUDGE_KEY"
```

A pair is built only when the request and both answers hash back to the digests in the manifest. A stored request names the model the router sent it to, so the input digest covers the request body without its top-level model field (`Model` in the recorded request, `model` in a client body), which is also the `input` a judge reads, and each task carries that `input_digest`. This changed input identity, deduplication and splits, so the manifest is `shadow-dataset.v2` and the tasks are `shadow-judge-tasks.v2`. A `shadow-dataset.v1` manifest is refused. The shadow text comes from the captured excerpt, so the shadow decision needs `capture_response_body` on and a `max_capture_bytes` large enough for whole answers. An excerpt that was cut, or a primary answer a response-stage plugin rewrote before it was stored, is left out and counted as `arm_text_digest_mismatch`, and a pair with no text at all as `arm_text_missing`. A request that does not match its digest is counted as `input_text_digest_mismatch`, and a missing one as `input_text_missing`. An answer that names its own model is marked `names_own_model`, since no label can blind a judge to that.

The tasks carry prompt and answer text, so the route needs `replay.detail` and refuses a caller without it rather than redacting the tasks. The judge runs outside the router.

### Publish the dataset and its judgments

`tools/calibration/shadow-dataset` publishes a manifest, and optionally the judgments a judge returned for it, into a storage directory. Any object store mounted as a filesystem works as that directory. The tool validates every input before it writes anything: the manifest has to match its own digest, the tasks have to be the ones the manifest and the blinding key produce from the text they carry, and the judgments have to answer those tasks with no field the judgment record does not define. The key is read for that check and never published.

```bash
make run-shadow-dataset GO_TOOL_ARGS="--dest /mnt/shadow-datasets \
  --manifest $PWD/manifest.json --tasks $PWD/tasks.json --judgments $PWD/judgments.json \
  --blinding-key $JUDGE_KEY"
```

The manifest lands at `manifests/<digest>/<digest of the file>.json`, because the manifest digest leaves out the counts and two exports of the same examples can differ in what they excluded, and the judgment set lands at `judgments/<manifest digest>/<digest of the set>.json` beside a `.report.json` that counts outcomes, slot picks and how many pairs followed the slot rather than the answer. Every name comes from content, so publishing the same files twice changes nothing and a published name is never replaced with different bytes. The judge tasks are read to check the judgments against but never published, because they carry the prompt and both answers. Keep datasets out of this repository.

## Read the results

Open **Insights** in the Dashboard, or use the
[Replay API](../../api/router#router-replay) to filter records and inspect a session.
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
