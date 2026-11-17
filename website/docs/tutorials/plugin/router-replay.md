# Router Replay

## Overview

Use Router Replay to inspect requests in Dashboard Insights: the selected route,
model, token usage, response, and tool trajectory. Configure storage in
`global.services.router_replay`, and the switch and capture defaults in a
`router_replay` plugin in `global.plugins`; use a recipe's or a decision's
`router_replay` plugin to override only the capture fields that differ.

Replay is on by default. A configuration without a `router_replay` plugin
records every request's routing data and none of its text, so Insights,
sessions and cost work with no extra configuration.

## What Problem Does It Solve?

Replay keeps request-level evidence together so you can understand a routing
choice and follow a conversation across requests.

## When to Use

Configure Replay when you need request text for inspection or troubleshooting,
records that survive a restart, or no records at all. Choose retention and
capture limits that match the data you are allowed to keep.

## Configuration

### Keep records across restarts

Without a store in `global.services.router_replay`, Replay uses the `file`
store. It writes each record once, when the request ends (completed, failed or
aborted), to `.vllm-sr/router-replay.jsonl` in the configuration file's
directory, a file only its owner can read, and reads the records back when the
Router starts, so Insights keeps its history across restarts and configuration
reloads. `vllm-sr serve` mounts that directory from the host, so the file
outlives the Router container.

- **In progress.** A request still in progress when the Router crashes is not
  in the file. What reaches a record after its request ends, such as a late
  outcome or a review, is written when the file is next rewritten: when the
  Router stops, and whenever the file reaches twice as many lines as the
  records it keeps.
- **Size.** The file keeps at most 1,000 records, or `max_records` when that
  is lower. A record of routing data takes about 8 KB, so the file stays near
  8 MB and reaches about 16 MB just before a rewrite. The `memory` backend
  keeps its default of 10,000.
- **Age.** Reading the file back and rewriting it drop records older than
  `ttl_seconds`; until then the running Router lists them, as the `memory`
  backend does.
- **Memory instead.** Where any `router_replay` plugin captures prompt or body
  text, where that directory cannot be written (the Helm chart mounts the
  configuration read-only), or where another Router process already holds the
  file, Replay keeps all its records in memory and logs a warning that says
  why.

All five built-in MoM recipes, including Vault, enable PostgreSQL Replay by
default. Local `vllm-sr serve` manages PostgreSQL and its persistent volume.
For another deployment method, configure the PostgreSQL connection in the
[shared Replay service](../learning/memory-and-replay#configuration).

The `memory` store keeps records in the Router process only and loses them
when the Router restarts or reloads configuration. Use PostgreSQL or Redis for
records that keep text or that several Routers share. Set retention for the
data you intend to keep.

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

Without a `router_replay` plugin on any layer, Replay records each request's
routing data only, as `prompt_capture: none` does: the recipe and decision,
the requested and selected model, token usage, cost, latency, status, the
session, turn and agent identifiers the request carries, and the answer's
SHA-256 and character count, which pair it with a shadow arm's answer without
keeping its text. It keeps no request or response body and no prompt, and it
writes to the store `global.services.router_replay` names: the `file` store
unless you configure another, at most 1,000 records there. On the `memory`
store, the decisions on this default and the requests that match no decision
share one store of `max_records`; a decision with its own `router_replay`
plugin keeps a store of its own `max_records`. A
`router_replay` plugin on any layer replaces that default and records only
where a layer sets `enabled: true`. The plugin in
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
  the shadow's `capture_response_body` say, and reads nothing for personal
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
  `router_replay` plugin in `global.plugins`. A recipe or decision can still
  opt in. Leaving every `router_replay` plugin out keeps the default record of
  routing data.
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
to 200 records for the file and memory stores. `max_prompt_bytes: 0`, like the
tool-trace limits, removes the limit instead. These zeros override an
inherited value, so omit a field to inherit it.
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

A shadow's `capture_response_body: true` now keeps its excerpt of the shadow's
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

### Choose the savings baseline

A record's savings compare its recorded tokens at the selected model's rates
with the same tokens at a baseline model's rates. By default the baseline is
the most expensive model of the selected recipe's pool, so a request routed to
a model in that pool never shows a loss. Name the model to compare with
instead:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      cost_baseline_model: premium-model
```

`cost_baseline_model` must name a model with pricing, or the configuration does
not load. Like the other fields it layers: set it in `global.plugins` for every
route, or on one recipe or decision. A request sent to a model dearer than the
baseline then shows a negative saving. A record whose baseline is priced in
another currency than the selected model has no baseline and says why
(`baseline_unavailable`); it never falls back to the default. Each record names
how its baseline was chosen in `cost_accounting.baseline_source`, and Insights
shows which baselines the savings cover.

A model with no `pricing` of its own takes its provider's list price from the
built-in catalog when the catalog has one, and Insights marks that cost as a
list price estimate. For a backend on the client's own login
(`auth.credential: caller`) a cost is the API-price equivalent of its tokens,
not what the client's plan charges; set `pricing` on such a model to see it.

Router decision calls are recorded separately in `cost_accounting.router_calls`:
`decision_model` for native decision tasks, `systemone` for remote System One
exchanges, and `embeddings`, `classify` or `rerank` for other runtime tasks.
Each entry names its model, input and output tokens, cost and currency, or an
unpriced reason. Fused runtime tasks count once; cached results and startup or
warmup calls add no request spend. Calls spent before a decision fails remain
in its record and in aggregate router spend. A record's generation cost and
`cost_savings` keep their meaning, while the savings Insights and the aggregate
API report are net of the Router calls of the requests they compare, with the
figure before them kept as `gross_saved`. A request whose Router call has no
price leaves the savings instead of being netted as free. Router spend is also
added to total spend. Auxiliary model calls made by internal Looper hops count
in the parent request's router spend and coverage; hop generation costs count
there once. A hop's own record says so with `cost_accounting.unpriced_reason:
counted_in_parent`, and its `original_model` names the algorithm that made the
call, such as `Ratings`, `Escalate` or `ReMoM`, since no client asked for it.
See the
[aggregate fields](../../api/router#configured-rate-cost-estimates).

Set deployment rates under `global.model_catalog.deployments.<name>.pricing`,
using the same per-million-token fields as `providers.models[].pricing`:

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        pricing: {currency: USD, prompt_per_1m: 0.2, completion_per_1m: 0.4}
```

Explicit deployment pricing wins. Otherwise, a deployment's `model` may name a
provider model whose pricing is used. Omitted currency defaults to USD; an
explicit empty pricing block declares zero rates. A `model_runtime` deployment
with neither, such as `primary` and `vela-embedding`, counts its calls as
free, because it runs on your own hardware and bills no tokens; give it
`pricing` to count a rate. A remote deployment without pricing, or a call
without complete usage, remains unpriced, never implicitly free, so the
requests that call it are left out of savings. A free call is netted at zero
whatever its currency. Remote System One usage
must include both input and output token counters, even when zero. These are token rate estimates; they do not measure local compute or GPU
costs.

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
        # Optional extra candidates: every arm gets the same request under one
        # aggregate per-request budget (issue #3376).
        arms: [candidate-arm]
        budget:
          max_calls_per_request: 0
          max_tokens_per_request: 0
          max_cost_per_request: 0
          price_per_million_tokens: 0
          reserve_tokens_per_arm: 0
          max_concurrency_per_request: 0
          max_response_bytes_per_request: 0
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
| `model` | one of `model`/`arms` required | Configured logical model that receives the shadow copy. Must have a backend in `providers.models`. |
| `arms` | `[]` | Extra shadow candidate models beyond `model`; deduplicated, and every arm must resolve to a backend. All arms of one decision share one per-request `budget`. |
| `budget.max_calls_per_request` | `0` | Arm admissions allowed per request across all arms. `0` = unlimited; this is the only dimension that binds without further settings. |
| `budget.max_tokens_per_request` | `0` | Aggregate token cap. It binds before dispatch only when `reserve_tokens_per_arm` is set; otherwise it is accounted on completion. `0` = unlimited. |
| `budget.max_cost_per_request` | `0` | Aggregate cost cap; requires `price_per_million_tokens`. `0` = unlimited. |
| `budget.price_per_million_tokens` | `0` | Token-to-cost conversion used by the cost cap. |
| `budget.reserve_tokens_per_arm` | `0` | Tokens reserved per arm at admission so concurrently admitted arms cannot overshoot `max_tokens_per_request`/`max_cost_per_request`. |
| `budget.max_concurrency_per_request` | `0` | Arms of one request allowed in flight at once; excess arms are dropped with reason `budget_concurrency_limit`. `0` = unlimited. |
| `budget.max_response_bytes_per_request` | `0` | Aggregate response bytes across the arms of one request. Each admission reserves the per-arm `max_response_bytes`, so the cap binds before dispatch and is swapped for the observed size on completion. `0` = unlimited. |
| `sample_rate` | `1.0` | Fraction of eligible requests to shadow, in `[0, 1]`. `0` keeps the shadow declared but never dispatches. |
| `max_concurrency` | `2` | In-flight shadow calls for this decision. |
| `max_queue_depth` | `8` | Calls waiting for a slot. Anything beyond is dropped with reason `queue_full`. |
| `timeout_seconds` | `30` | Deadline for queue wait plus execution, shared by all retries. |
| `max_response_bytes` | `1048576` | Largest shadow response body that is read. Larger bodies fail with `response_too_large`. |
| `max_retries` | `0` | Extra attempts on transport errors or retryable statuses. Capped at `3`. |
| `capture_response_body` | `false` | Store a bounded excerpt of the shadow text in the outcome, under `prompt_capture: raw` only. Off by default; only sizes, tokens, and a SHA-256 are kept. |
| `max_capture_bytes` | `4096` | Excerpt bound when capture is on. |
| `tls_skip_verify` | `false` | Skip certificate verification for an https shadow backend signed by an internal CA. This setting applies only to the shadow HTTP client; it does not configure primary backend transport. |
| `forward_headers` | `[]` | Decision `header_mutation` names the shadow copy may carry, matched case-insensitively. Nothing else a decision sets for the primary backend is forwarded, so a custom credential such as `X-Internal-Token` stays on the primary path. Known credential carriers (`Authorization`, `Proxy-Authorization`, `Cookie`, `x-api-key`, `api-key`, `x-goog-api-key`, the `x-user-*-key` headers) are rejected at config load and dropped at run time even if listed. |

A shadow is skipped, with a metric but no outcome, when the request is
sampled out or when the primary dispatch already selected the shadow model. A
decision whose algorithm is an executor (`escalate`, `fusion`, `remom`,
`ratings` or `workflows`) rejects a shadow at config load, because the shadow
runs only on single-model provider dispatch.

### Compare several candidates at once {#multi-arm}

`model` plus `arms` turn one shadow into a multi-arm comparison: every
deduplicated model receives the same approved request, and the replay record
keeps one bounded outcome per arm, each with its own `target_ref` and shadow
request identity. The primary response is still built once and never waits for
any arm.

One `budget` governs the whole arm set for a request, so a fan-out cannot
multiply resource use behind the operator's back. `max_calls_per_request`
admits at most that many arms (the rest are dropped with
`budget_call_limit`); `max_tokens_per_request` and `max_cost_per_request` are
accounted on completion unless `reserve_tokens_per_arm` reserves per arm at
admission, which is the only way they bind before dispatch;
`max_concurrency_per_request` bounds arms in flight and
`max_response_bytes_per_request` bounds their aggregate response bytes. A zero
field is unlimited. `max_concurrency` still bounds the decision across
requests, and `max_response_bytes` still bounds one arm's body.

An arm that resolves to the primary model is skipped with `same_as_primary`,
like a single-model shadow. Give the decision's `router_replay` capture and the
export/judge endpoints in the rest of this page the same treatment as a
single-model shadow: each arm is one more arm in the manifest.

### Fail-open behavior

The request path does one non-blocking slot check and returns; everything
else runs in a bounded worker after the primary dispatch response has been
built. A slow, unavailable, malformed or overloaded shadow endpoint can't
change the primary response or its latency. Every shadow ends in exactly one
result:

| Result | Reasons |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`, `credential_unresolved`, `encode_failed`, `timeout`, `transport_error`, `upstream_status`, `redirect_rejected`, `response_too_large`, `malformed_response` |
| `dropped` | `queue_full`, `queue_timeout`, `router_closing`, `same_as_primary`, `internal_request`, `request_unavailable`, `backend_lacks_capability`, `budget_call_limit`, `budget_token_limit`, `budget_cost_limit`, `budget_concurrency_limit`, `budget_response_bytes_limit` |
| `sampled_out` | `sampled_out` |

A request that needs a modality the shadow model's card omits, such as an image sent to a text-only model, is dropped with `backend_lacks_capability` and never reaches the shadow backend. A shadow model with no recognized declaration is not gated.

- **Metrics.** Results and reasons are exported as
  `sr_shadow_dispatch_total{decision,result,reason}`, with
  `sr_shadow_dispatch_latency_seconds`, `sr_shadow_dispatch_inflight` and
  `sr_shadow_dispatch_queued`.
- **Drops caused by resource bounds** are reported through metrics and a
  structured `shadow_dispatch_dropped` event, not a replay write, so an
  overloaded shadow lane can't add load to the replay store.

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

Completed and failed shadows append one outcome to the primary request's
replay record. Outcomes are append-only:

| Part | Holds |
| --- | --- |
| The outcome | `source: shadow_dispatch`, `target: model`, `target_ref: <shadow model>`, the verdict and the reason |
| Its metadata | Primary and shadow request identity; primary and shadow model and backend; decision and recipe; sample rate; enqueue, start and finish timestamps; queue wait and latency; attempts; status code; response size; stop reason; token counts; a SHA-256 of the shadow text |

Replay redaction applies to shadow outcomes as it does to the rest of the
record: viewers without content rights see the routing and timing fields, but
not `target_ref`, `reason` or `metadata`. The shadow's `capture_response_body`
is its own, and keeps its excerpt only under `raw`, which alone keeps the
primary's answer: generated text can repeat the prompt, and `verified` checks
the prompt only. Under `none` and `verified` no record keeps one. Keep it off
unless the replay store and its readers are cleared for prompt-level content.
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
  needs `capture_response_body` on, and a `max_capture_bytes` large enough for
  whole answers.

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
short hash of the agent id in the row's title, and the main conversation is
`main`. Every record keeps that hash in its `agent`, as described in
[Agent runs](#agent-runs). A call without one shows `main`. A Codex root
thread, whose agent and session ids are equal, is `main`. A viewer without
`replay.detail` gets no hash, as with the rest of the redacted session policy,
so a subagent's call shows `hidden` there.
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

### Usage in the terminal

```bash
vllm-sr usage                     # since midnight today, local time
vllm-sr usage --since 1h          # or 30m, 7d
vllm-sr usage --since 7d --json
vllm-sr usage --endpoint http://localhost:8080
```

`vllm-sr usage` prints the figures Insights shows for one window, read from
the replay list and aggregate with `since`:

```text
Usage since 2026-11-16T00:00:00+09:00 (http://localhost:8080)
  Requests      6: 6 completed, 0 failed, 0 aborted
  Sessions      2
  Spend         0.000094 USD
  Net savings   0.001436 USD (gross 0.001436 USD, Router calls netted 0.000000 USD, 6 requests compared)
  Prompt cache  saved 0.000000 USD over 6 requests
  Reviews       2, approval 50% (1 allowed, 1 denied)
Top models
  qwen-chat  3 requests, 0.000071 USD
  qwen-msg   3 requests, 0.000023 USD
Top entrypoints
  vllm-sr/auto    3 requests, 0.000023 USD
  vllm-sr/review  2 requests, 0.000030 USD
  qwen-chat       1 request, 0.000041 USD
```

`Spend` is the aggregate's `total_spend`, the Router's own calls included and
shown apart when they cost anything. `Net savings` is `total_saved`, with
`gross_saved` and `netted_router_call_spend` beside it, over the requests
compared with a baseline; with none it reads `n/a`. Amounts stay per currency.
`Prompt cache` is the aggregate's provider prompt-cache saving, and `Reviews`
appears when the window holds Codex approval reviews. These come from the
aggregate. Requests, sessions and the top three models and entrypoints come
from the window's records, each client request once: a Looper hop's record
counts in the request that made it, not as a request of its own. `Requests`
adds `in progress` and `unknown` when there are any. Models are ranked by
`selected_model`, entrypoints by `original_model`, the model the client asked
for. Above, one request asked for `qwen-chat` directly: it is the entrypoint of
that one request and the selected model of three, the two review requests routed
to it included. A row's spend adds its requests' own costs as `total_spend` adds
them, `actual_cost` and the cost of the Router's calls for each request, per
currency; a request with no price is counted as such, so a row that reads
`no price` has every one of its requests in `unpriced`. `--json` prints the
same, with the aggregate's `by_currency`, provider-cache and `reviews` objects
as the Router returns them and an `unpriced` count on each row.

The command reads the management API at `--endpoint`, by default the local one
on port 8080 plus `VLLM_SR_PORT_OFFSET`, and sends `VSR_MGMT_TOKEN` as the
bearer token when set. When the Router answers `404` with `REPLAY_UNAVAILABLE`,
the command says in one line that Replay is off and how to turn it on, and exits
with status 1; any other error is printed as the Router gave it. It also exits
with status 1, asking for an upgrade, when the Router predates it: one that
still lists a record for a `since` a day ahead has no `since` filter, and one
without `gross_saved` reports savings with its own calls left in.

### Agent approval reviews

When Codex's **Approve for me** asks a reviewer whether to run a planned
action, Replay records that review on the request's record, with
`request_kind: codex_approval_review` and a `review` object. A request is a
review when it names an entrypoint that another entrypoint uses as its
`review_model` (`identification_source: entrypoint`), or, when reviews use the
session model, when it carries Codex's review request: its judge instructions,
the approval request block and the verdict schema
(`identification_source: content`). Recognition only records; it never changes
routing, authorization or masking. The default record keeps each review's
status, verdict, risk, tool and sessions; the action's summary, directory and
arguments and the rationale need `prompt_capture: raw` with body capture on,
on the recipe that answers reviews; see
[Approve for me reviews](../../installation/agent-harness#approve-for-me-reviews).

| Field | Holds |
| --- | --- |
| `status` | `completed` with a verdict, `tool_call` when the reviewer asked for a read-only check and answers in a later request, `invalid` when the answer was missing or not the verdict schema, `failed` when the upstream call failed, `cancelled` when the client left first, and `pending` while the request runs. A failed or cancelled review is never a deny. |
| `assessment` | Completed reviews only: `outcome` (`allow` or `deny`), and `risk_level`, `user_authorization` and `rationale` when the reviewer gave them. The rationale is cut at 2 KiB. |
| `action` | The planned action's `tool`, a one-line `summary` such as the shell command, its `cwd`, and the action as Codex sent it in `arguments`, cut at 16 KiB. |
| `reviewed_session_id`, `thread_id`, `turn_id`, `parent_turn_id` | The agent session whose action was judged, the reviewer's thread and turn, and the session's turn that planned the action. |
| `latency_ms` | The review request's duration. Tokens, model and cost are the record's own fields. |

The verdict is read from the decoded answer, buffered or streamed, before any
response rewrite or body truncation. A review keeps its text as the record
keeps request and response text:

- **The action's summary, directory and arguments** are kept only under
  `prompt_capture: raw` with `capture_request_body: true`, and never on a
  masking route. Otherwise only the tool name is kept.
- **The rationale** is kept only under `prompt_capture: raw` with
  `capture_response_body: true`.
- **List responses** leave out `arguments`; read one record for them.
- **Principals without `replay.detail`** receive the status, verdict, risk and
  tool, with the summary, directory, arguments and rationale cleared.

A review is filed under the session a header names, as every record is. When
none does, it is filed under the session it judged, so the review appears with
that session's requests. Every store keeps the review on the record; PostgreSQL
also keeps `request_kind`, the status, the outcome and the risk in indexed
columns, added to an existing table when the Router starts.

In **Insights**, the **Reviews** switch shows reviews alone: the number of
reviews, the approval rate, denials, failures, p50 and p95 latency and cost,
reviews over time by result, requests by review model, and a table with each
review's time, result, risk, action, session, model, latency, tokens and cost.
Filter by session, result, risk and review model; a row opens the record with
its review section and a link to the reviewed session's reviews. A session's
routing history marks the turn each review judged and counts its reviews in
the summary. Past 2,000 reviews it keeps the newest and says that older reviews
are not shown.

The [Replay API](../../api/router#router-replay) filters with `request_kind`,
`review_status`, `review_outcome` and `review_risk`. With reviews among the
filtered records, the aggregate response carries a `reviews` summary over the
whole filtered set: counts by result, `approval_rate` as allowed verdicts over
allowed plus denied verdicts, latency percentiles of reviews with a verdict,
recorded cost per currency with the number of unpriced requests, reviews over
time, and requests by review model.

### Agent turns

A request Codex sends carries its turn metadata, in the `x-codex-turn-metadata`
header and the Responses request's `client_metadata`, and every record of it
keeps three of its ids:

| Field | Holds |
| --- | --- |
| `thread_id` | The Codex thread that sent the request: the session's own thread, a reviewer's or a subagent's. |
| `turn_id` | The turn of that thread the request belongs to. |
| `parent_turn_id` | The turn that started the thread, such as the turn an approval review judges or a subagent was spawned from. |

The [Replay API](../../api/router#router-replay) filter `turn_id` matches
`turn_id` or `parent_turn_id`, so one query returns a turn's calls, its
approval reviews and the calls of subagents it spawned. The Codex plugin reads
it at the end of each turn to name the models that served it. Every store
keeps the ids on the record; PostgreSQL also keeps them in columns, with
indexes on `turn_id` and `parent_turn_id`, added to an existing table when the
Router starts.

Trajectory routes carry `turn_id` and `parent_turn_id` too. In **Insights**, a
record's Session Routing section shows its Codex thread, turn and parent turn,
and each turn opens the request list filtered to it (`/insights?turn=...`),
with a chip that clears the filter. A session's routing history opens each
Codex turn with a row that links there and counts its calls, so a turn's own
calls and its subagents' calls read as one unit, and a review that names the
turn it judged is marked on that turn's calls.

### Agent runs

An agent run is one session of an agent harness: its main agent, the
subagents it spawned, and every call they made. Replay files each call under
the session the harness names and records the agent that made it, from the
headers the harness already sends, whichever headers `identity.headers`
points protection at:

| Harness | Run | Agent | Parent |
| --- | --- | --- | --- |
| Claude Code | `x-claude-code-session-id` | `x-claude-code-agent-id`, sent on subagent calls only | none: every subagent hangs off the main agent |
| Codex | `session-id`, the root session on every thread | `thread-id`; the root thread, equal to `session-id`, is the main agent | `x-codex-parent-thread-id`, unless it is the root thread |
| Others | the pinned session, `x-vsr-session-id` or a `session_id_headers` header, or the one `identity.headers.session` names | the header `identity.headers.agent` names | none |

A call protection cannot key is filed under its run all the same, not under a
session of its own per call: a Codex Responses call under `scope: conversation`
without `x-conversation-id`, or under an `identity.headers.session` Codex does
not send, is filed under its `session-id`. Each record keeps its agent in
`agent`:

| Field | Holds |
| --- | --- |
| `harness` | `claude_code` or `codex`, when the request carries their identity headers. |
| `id` | The short SHA-256 hash of the agent id, empty for the main agent. |
| `parent_id` | The hash of the agent that spawned it, empty when the main agent did. |
| `kind` | The subagent kind Codex sends in `x-openai-subagent`, such as `collab_spawn` or `review`. |

A Looper hop carries none of its client's headers. Its record is filed in the
run of the request that made it, under that request's agent, with
`parent_request_id` naming the request. A record also keeps the
`reasoning_effort` the request was sent with. A record written before these
fields name a subagent only where protection recorded one.

Two [Replay API](../../api/router#router-replay) routes read runs. `GET /api/v1/observability/replays/runs`
takes the list's filters, `limit` and `offset`, and returns one summary per
session over the records the filters keep, latest activity first.
`GET /api/v1/observability/replays/run?session_id=...` returns one run across
recipes: the same summary in `run` and every call in time order in `calls`.

| Field | Holds |
| --- | --- |
| `session_name` | The name the harness last gave the session, when Replay saw one: see [Session names](#session-names). |
| `prompt` | The main agent's first prompt, when Replay kept it, without `<system-reminder>` blocks. A session-title request is not the task. |
| `harness`, `models`, `duration_ms` | The harness, the calls each model answered with Looper hops included, and the time from the first call's start to the last one's end. |
| `agents` | Each agent: its `label` (`main`, then `subagent 1`, `subagent 2` by first call), the `parent` agent's label, `kind`, `id`, `calls`, first and last call, `spawned_after_record_id` (the parent's last call before this agent's first), `models` and `cost`. |
| `outcome` | `status` from the main agent's last call: `answered` when it gave a final answer, `failed`, `aborted`, `in_progress`, or `unknown` when it asked for a tool and no call followed. With it `final_answer_record_id`, the tool results the calls carried and how many failed, each counted once, and `reported`, the outcomes posted to `/api/v1/observability/outcomes` for the run's calls. |
| `cost` | Spend per currency as the aggregate's `total_spend` adds it, the Router's own calls apart, with the calls missing a price counted, never as free, and the tokens. |
| `events` | Calls per marker: `switch` (another model than the agent's previous call to the same entrypoint), `escalation` (an `escalate` Looper answered by a later model than its first), `guard` (jailbreak, PII or hallucination detected), `block`, `fallback` (a cross-model fallback tried another candidate), `failure`, `tool_failure`. |
| `calls` | Each call's `agent` label, `decision`, matched `signals` as `type:name`, `entrypoint`, `model`, `previous_model` on a switch, `effort`, `selection_method`, `session_action`, `reason` (protection's reason, and the Looper algorithm with its final attempt's reason), lifecycle and status, tokens, `cost`, `router_calls`, `looper_attempts`, `tool_results`, `tool_failures`, `final_answer`, `outcomes`, `markers`, and the Looper `hops` it made. |

Without `replay.detail` a run keeps its tree, labels, markers and counts, and
loses what its records would: the session name, the prompt, the agent hashes,
the reasons and the text of reported outcomes. The runs list reads the filtered
records whole, so bound it with `since` on a large PostgreSQL store.

In **Insights**, the **Runs** view lists one row per run: its session name or
task, the harness, the agents, the models, the outcome, the cost and duration,
and the events. Insights shows a model by its label wherever it names one: the
runs, records and reviews lists, the model filters and charts, a run's lanes,
legend and step panel, and a record's route, routing, candidate scores, cost
text and session history, with the configured name beside it, in its tooltip,
or in parentheses after it. A run opens as a timeline: a lane per agent in
tree order, a dashed tick where a subagent started, and consecutive calls of
one agent on one model drawn as one segment with its calls' markers, so a run
of hundreds of calls reads as a few blocks. A segment lists its calls, and a call shows its decision
and signals, model and effort, tokens and cost, status and reason, with a link
to its record. The table of every call is one click further. A record page
opens its run with **Open run**.

With [tracing](../global/api-and-observability#observability) on, the
Router's request span carries `session.id`, `vsr.harness`, `vsr.agent.id`,
`vsr.agent.parent_id` and `vsr.agent.kind`, and a run is one trace:

- **The harness sends `traceparent`.** The span stays in the harness's trace.
  Claude Code sends one through a custom base URL only with its tracing on
  and `CLAUDE_CODE_PROPAGATE_TRACEPARENT=1`, a trace per user prompt with the
  subagents' calls in it.
- **It sends none.** The span joins its run's trace: the Router creates a
  root span `semantic_router.run` for the session and a `semantic_router.agent`
  span for each subagent, under the agent that spawned it. Their ids derive
  from the session and agent ids, so every call of the run lands in one trace
  whichever Router process served it; each process emits those spans once.
  Codex 0.166.0 sends no `traceparent` on its model calls.

For a complete decision fragment, see
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml).

#### Session names

A run reads by the name its harness shows for the session, where the gateway
can see that name. Each harness names a session with a side request of its own,
which Replay recognises by what the harness marks, never by its prompt, and
records with `request_kind: session_title`:

| Harness | How it names the session | What Replay records |
| --- | --- | --- |
| Claude Code 2.1.273 and later | After the first prompt, an `x-claude-code-request-class: auxiliary` request asking for a JSON object of one `title` field. `/rename` sends nothing; the next request carries the command and its output in its history. | The title the answer gives on the title request's record, and the name a `/rename` gave on the record of each request whose history shows it. The latest names the run. |
| Codex CLI 0.166.0 | Its terminal UI starts a thread of its own whose turn metadata says `thread_source: thread_title`, under a new `session-id`. No field of that request names the thread it titles, and `/rename` sends nothing. `codex exec` names no thread. | Neither name. A session holding only title requests is no run and is left out of the list. |

Claude Code marks its request class only with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`,
which `vllm-sr connect claude-code` sets. The title is response text and a rename
request text, so `session_name` is kept only under `prompt_capture: raw` with
that side's body capture on, never from a masking route's request, and cleared
for a viewer without `replay.detail`. A run without a name reads by its first
prompt, then by its session id.

#### Model labels

A record names the model as the configuration does, such as `qwen-msg`.
`GET /api/v1/observability/replays/models` lists every configured model with
the label a person reads it by, from the catalog the Router routes with: its
card's display name when it is catalog-backed (`catalog:`) or its card gives
one, else the display name of the catalog card its backend's provider maps its
`provider_model_id` to, else that `provider_model_id`:

```json
{"object": "router_replay.models", "data": [
  {"name": "qwen-msg", "label": "Qwen3.6 35B A3B"},
  {"name": "premium-ref", "label": "qwen36"}
]}
```

A record of a model the configuration no longer holds reads by its name.
