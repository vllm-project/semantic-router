---
title: Protocol Compatibility Matrix
description: Match client-facing inference APIs to supported backend model protocols and understand cross-protocol feature boundaries.
---

# Protocol Compatibility Matrix

Semantic Router supports three inference wire formats on both sides of the
data plane. A client request is decoded into a protocol-neutral form, routing
policy selects a model, and that model's `api_format` selects the backend codec.
The response is translated back to the client's original format.

```text
client endpoint -> client codec -> routing -> backend codec -> model endpoint
```

Protocol compatibility is separate from target configuration and deployment
support:

- use [Backend Target Compatibility](backend-target-compatibility) for URLs,
  weights, headers, discovery, and producer preservation; and
- use [Deployment Support](support-matrix) for
  project-maintained stacks, integrations, and hardware profiles.

For client connection settings, virtual model limits, and a tool-loop check,
start with [Connect an agent harness](agent-harness).

## Client-facing protocols

| Client API | Inference endpoint | Buffered | Streaming | Availability |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Supported | Available on the public inference listener. |
| OpenAI Responses | `POST /v1/responses` | Supported | Supported | Requires `global.services.response_api` and its store to be available. Router-owned object operations are not forwarded to a model backend. |
| Anthropic Messages | `POST /v1/messages` | Supported | Supported | Send the Anthropic request shape and an appropriate `anthropic-version` header. Client authentication remains deployment-specific. |

The public listener also serves `GET /v1/models` and Anthropic's
`POST /v1/messages/count_tokens`, which coding agents use to size a file before
reading it. See the
[Router API](../api/router) for the complete method and path inventory,
Responses object operations, and request examples.

System One is a separate typed API, available in either startup mode when
`listeners[].systemone.models` publishes a concrete model. It uses
`POST /v1/systemone` (alias `/v1/decisions`) and
`GET /v1/systemone/models`. Router mode can also publish an explicit native
recipe through an `api: systemone` entrypoint and matching listener grant.
These typed requests do not use the Chat protocol-translation matrix or the
Chat model allowlist. See the
[System One quickstart](../model-runtime/quickstart.md).

## Backend model protocols

Set `api_format` on each `providers.models[]` entry. It describes the wire
contract implemented by that model endpoint, not the provider brand.

| `api_format` | Backend request and response shape | Default upstream path | Notes |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | Default when `api_format` is omitted. `backend_refs[].chat_path` can override the Chat path. |
| `responses` | OpenAI Responses | `/v1/responses` | The backend itself must implement the Responses wire contract; enabling the Router's Responses service does not add that API to a backend. |
| `anthropic` | Anthropic Messages | `/v1/messages` | Configure provider authentication and required version headers on the backend ref. vLLM serves this API from 0.11.1, so a vLLM backend keeps `provider: vllm`, which sends the bearer token vLLM checks for `--api-key`. |

These fields are easy to confuse:

- model `api_format` selects the request, response, error, and streaming codec;
- backend-ref `protocol` selects HTTP or HTTPS transport; and
- backend-ref `provider` supplies provider-specific authentication and path
  defaults. It does not prove that the endpoint implements an API format.

For every backend format, a full URL in `endpoint` names the complete upstream
API root. Its path is retained and the protocol operation suffix is appended
exactly once; the protocol's default `/v1` base path is used only when the URL
has no path. `chat_path` applies only to Chat Completions.

With `--gateway extproc`, the generated Envoy cluster for an HTTPS backend
verifies both the server
certificate chain and its DNS hostname. An HTTPS replica pool must keep one
hostname because the supported Envoy runtime shares its TLS context within a
cluster; use separate model aliases for different HTTPS hosts. IP-literal HTTPS
targets are rejected instead of silently weakening hostname verification. A
custom Envoy image used with `vllm-sr serve` must provide the system CA bundle at
`/etc/ssl/certs/ca-certificates.crt`; startup validation fails rather than
silently disabling verification when that trust store is unavailable.

### Response fields the Router does not model

Providers and gateways add fields of their own to replies: a gateway's cost or
trace ID, provider metadata, or a field a provider ships before the Router
models it. Any backend can send them, and no setting is needed. These two
backends, one serving the Anthropic Messages format through a gateway and one
serving Chat Completions through a relay, need nothing beyond their addresses,
even when every reply carries a field such as `"total_cost": 3e-05` inside
`usage`:

```yaml
providers:
  models:
    - name: claude-via-gateway
      provider_model_id: example-claude
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic-compatible
          base_url: https://gateway.example.com/anthropic
    - name: gpt-via-relay
      provider_model_id: example-gpt
      api_format: openai
      backend_refs:
        - name: relay
          provider: openai-compatible
          base_url: https://relay.example.com/v1
```

- A reply, a stream event, or an error body that carries a field its
  `api_format` does not define still succeeds. A client that uses the
  backend's format receives the reply unchanged, field included. A client of
  another format receives the translated reply without the field.
- Each such field is reported as a `dropped` diagnostic. It is counted in
  `llm_translation_lossy_total` and logged at debug level, and buffered
  replies list it in `x-vsr-protocol-warnings`.
- Fields the Router models are still validated: their types and values, the
  usage totals, the rule that a streamed usage counter never decreases, and
  the rule that no counter is negative. A field the Router does not model
  never counts toward usage, and the response cache stores the reply without
  it.
- When a Chat Completions reply names its usage counters `input_tokens` and
  `output_tokens`, as some relays in front of Anthropic models do, the Router
  reads them as `prompt_tokens` and `completion_tokens`.
- Error bodies in a provider's own shape are decoded by the backend's
  `provider`: `cloudflare-workers-ai`, `openrouter`, and `snowflake-cortex`.
- A Chat reply that carries `fusion`, `flow`, or `reasoning_mom_responses` is
  refused. These names belong to the Router's own Looper output, so a client
  must not receive a backend's value under them.

This applies to the first attempt and to native fallback, and to buffered,
streamed, and gzip-compressed replies.

Earlier releases had a backend-ref setting named `compat_gateway`, also spelled
`openai_compat_gateway`, that enabled this behavior for one backend. It was
removed. A configuration that still sets it:

```yaml
providers:
  models:
    - name: claude-via-gateway
      provider_model_id: example-claude
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic-compatible
          base_url: https://gateway.example.com/anthropic
          compat_gateway: true
```

fails to load with this message. Delete the field:

```text
providers.models[claude-via-gateway].backend_refs[0]: compat_gateway was removed; gateway response fields are accepted by default, so delete it
```

## Client-to-backend matrix

Every client format can route to every backend format. Each cell is covered in
both buffered and streaming mode.

| Client protocol | `openai` backend | `responses` backend | `anthropic` backend |
| --- | --- | --- | --- |
| OpenAI Chat Completions | Supported | Supported through codec translation | Supported through codec translation |
| OpenAI Responses | Supported through codec translation | Supported | Supported through codec translation |
| Anthropic Messages | Supported through codec translation | Supported through codec translation | Supported |

"Supported" means the Router owns the request, response, transport-error, and
streaming translation path. It does not mean every field from one protocol can
be represented by every other protocol, or that every model behind an endpoint
supports the requested capability.

## Feature portability

The Router checks required semantics before encoding a backend request. A
feature that the selected backend format cannot represent fails explicitly
instead of being silently dropped.

| Semantic feature | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| Text, image input, and file input | Supported | Supported | Supported |
| Tools, parallel tool calls, and strict tool schemas | Supported | Supported | Supported |
| Custom (free-form) tools and their calls | Supported | Supported | Not supported |
| Text verbosity (`low`, `medium`, `high`) | Supported | Supported | Not forwarded; reported as `dropped` |
| Strict JSON Schema output | Supported | Supported | Supported |
| Buffered and streaming responses | Supported | Supported | Supported |
| Reasoning content and effort | Supported, signed thinking blocks only with an opt-in | Supported, signed thinking blocks only with an opt-in | Supported |
| Reasoning summary requests (`reasoning.summary`) | Not forwarded; reported as `dropped` | Supported; reported as `dropped` if the provider uses `chat_template_kwargs` for reasoning controls | Not forwarded; reported as `dropped` |
| JSON object mode without a schema | Supported | Supported | Not supported |
| Audio input | Supported | Not supported | Not supported |
| Hosted image-generation lifecycle | Not supported | Supported | Not supported |
| Multiple response candidates | Supported | Not supported | Not supported |
| Prompt-cache directives | Supported | Not supported | Supported |
| Prompt cache key (`prompt_cache_key`) | Supported | Supported | Not forwarded; reported as `dropped` |
| Reasoning token budget | Supported extension | Not supported | Supported |
| Seed and frequency or presence penalties | Supported | Not supported | Not supported |
| `top_k` sampling | Supported extension | Not supported | Supported for nonnegative values |
| `min_p`, repetition penalty, and cache salt | Supported extensions | Not supported | Not supported |
| Stop sequences | Supported | Not supported | Supported |
| Native response or conversation state fields | Not supported | Supported | Not supported |
| Deferred tool loading (`tools[].defer_loading`) | Not supported; opt-in sends tools in full | Not supported; opt-in sends tools in full | Not supported; opt-in sends tools in full |

This table describes codec representation, not model capability. For example,
an OpenAI-compatible server can accept the Chat request shape while rejecting
images or tools for a particular model. Qualify the actual endpoint and model
revision before adding them to a routing pool.

Responses requires a detail level on every input image. An image that arrives
without one, as every Messages image and tool-result image does, reaches a
Responses backend with `detail: auto`, the Responses default.

When translating Chat history to an Anthropic Messages backend, empty text
blocks are omitted. An assistant's `content: ""` beside `tool_calls` therefore
behaves like `content: null`. Empty tool results keep their call IDs and use
`content: []`; an empty system prompt is omitted. Non-empty text, including
whitespace, is preserved. A conversation message containing only empty text
is rejected, as is an empty text block carrying a cache directive, because
removing either would discard a conversation turn or cache boundary.

For an Anthropic Messages client using a Chat Completions or Responses backend,
`thinking.type: adaptive` uses the backend model's default reasoning behavior;
`output_config.effort` is retained. `thinking.display: omitted` removes reasoning
from the translated response, including streaming output. `summarized` and
`updates` (which Claude Code sends) reach only Messages backends and are rejected
for these backends. Explicit `thinking.type: disabled` requires a configured
reasoning family and an effective backend reasoning-off control. Unsupported
controls fail with a typed
request error. A `context_management` edit of `clear_thinking_20251015` with
`keep: all` has no effect and is omitted for these backends; edits that would
change history are rejected. Anthropic `cache_control` boundaries are omitted
when the selected backend uses Responses, which cannot represent them. The
prompt and tool result still dispatch, and `x-vsr-protocol-warnings` reports a
`dropped` diagnostic for `cache_control`. A `system` message inside `messages`,
such as the environment section and the file change notices Claude Code
inserts during a session, reaches these backends as a system message in the
same place. Many open-model chat templates accept a system message only first:
Qwen3.5 and later reject the request, and gpt-oss drops the text. For backends
served with such a template, add `messages[].role` to
`global.router.allow_lossy_translation`, described under
[Validation and failure behavior](#validation-and-failure-behavior). The text
then joins the user turn right before it, or becomes a user turn of its own
after a tool result, and `x-vsr-protocol-warnings` reports an `approximated`
diagnostic for `messages[].role`. Messages backends receive the message in
place either way.

`tools[].defer_loading` asks the provider's tool search to load a tool on
demand. The Messages format carries it, but the Router's protocol-neutral
request has no field for it, so the Router rejects it at ingress with
`unsupported_tools_defer_loading` whatever the backend format, Messages
included. Claude Code sets it on its first call, retries
without it, and then sends every tool definition on every call. Listing
`tools.defer_loading` in `global.router.allow_lossy_translation` keeps tool
search on: the Router sends each deferred tool in full, a `tool_reference`
block in a tool result reaches the backend as the tool's name, and
`x-vsr-protocol-warnings` reports a `dropped` diagnostic for
`tools.defer_loading`.

Reasoning content translated out of an Anthropic backend fails when the backend
attaches a reasoning signature, which its thinking responses carry by default.
A Chat Completions or Responses client then receives a typed
`unsupported_capability` failure on buffered requests; on streaming requests the
failure arrives mid-stream, after the response headers and any earlier deltas,
because the signature reaches the encoder only with the reasoning delta.
Same-format traffic, including Messages clients reading an Anthropic backend, is
unaffected. Listing `reasoning.signature` in
`global.router.allow_lossy_translation` lets these responses through: the Router
drops the signature, keeps the reasoning text, and reports the loss as
`approximated`. The signature cannot come back on later turns, so use it with
backends that do not need it, such as vLLM.

A Responses client can still use `previous_response_id` with a Chat
Completions or Messages backend. The Router retrieves and materializes the
retained history, removes Router-owned object controls, and then encodes the
stateless request in the selected backend format.

Responses custom tools use `type: custom` with a flattened `format` object.
Their `custom_tool_call` and `custom_tool_call_output` items retain free-form
input, tool results, and call IDs through Chat or Responses backends, including
buffered and streaming responses. A Messages backend rejects them with
`unsupported_capability`. Responses `text.verbosity` maps to the Chat
`verbosity` field. Messages has no equivalent, so the Router drops this output
detail hint and reports `text.verbosity` in `x-vsr-protocol-warnings`.

## Configure a backend format

The client can use any supported client-facing endpoint; `api_format` controls
what the selected backend receives:

```yaml
providers:
  models:
    - name: hosted/claude
      provider_model_id: claude-model-id
      api_format: anthropic
      backend_refs:
        - name: anthropic-primary
          endpoint: https://api.anthropic.com
          provider: anthropic
          auth: {api_key_env: ANTHROPIC_API_KEY}
          headers:
            anthropic-version: "2023-06-01"
          weight: 100
```

`api_format` chooses only the backend codec. It does not imply Anthropic,
OpenAI, or any other runtime Provider. Router-owned listeners require a
physical model to declare `backend_refs[].provider`; metadata-only
`listeners: []` configurations leave transport and credentials to the external
gateway. The local `vllm-sr serve` workflow owns backend transport through its
standalone frontend or generated Envoy configuration, so it does not accept a
backendless physical model; use the external-gateway deployment profile for
that topology.

Test the backend directly with its native path and a minimal request first.
Then send the same semantic request through the Router using the client API that
your agent harness or other API client needs. A successful health check does not validate request
schema, streaming, tools, or error translation.

## Validation and failure behavior

- Public requests are decoded into the neutral contract even when client and
  backend formats match. Unknown or unsupported request fields fail closed.
- Backend replies are decoded into the same contract. A reply field that its
  `api_format` does not define is dropped and reported instead of failing the
  reply; see
  [Response fields the Router does not model](#response-fields-the-router-does-not-model).
- Cross-protocol requests preserve shared semantics. Target-specific features
  that cannot be represented return a typed protocol error.
- Anthropic `tool_result.is_error: true` has no equivalent in OpenAI Chat or
  Responses requests. By default the Router rejects this translation with
  `lossy_translation`, while a false or absent flag remains supported and
  Anthropic backends preserve the flag. Coding agents such as Claude Code mark
  every failed tool call this way and do not retry the rejected request, so a
  run against an OpenAI-format backend ends at its first failing command. To
  serve them, list the field in `global.router.allow_lossy_translation`:

  ```yaml
  global:
    router:
      allow_lossy_translation: [tool_result.is_error]
  ```

  The Router then omits the flag, leaves the tool-result content unchanged,
  and reports the loss as `approximated` with field `tool_result.is_error` in
  `x-vsr-protocol-warnings` and in `llm_translation_lossy_total`. The list
  also accepts `messages[].role` for system messages inside the conversation
  and `tools.defer_loading` for deferred tool loading, both described above,
  `reasoning.signature` (see the signed thinking note above), and every other
  lossy translation still fails closed.
- The response keeps the client protocol's JSON or SSE shape. Provider
  transport errors and incomplete streams are translated separately from
  successful model responses.
- For Anthropic-compatible providers, an absent nullable `stop_sequence` is
  interpreted as null with a bounded compatibility diagnostic. This applies to
  buffered Messages, `message_start.message`, and `message_delta.delta`.
  Explicit null needs no diagnostic; a `stop_sequence` stop reason still
  requires a non-empty matched sequence, and terminal deltas still require
  `stop_reason`.
- When an OpenAI-compatible Chat provider names the matched stop string in
  `choices[].stop_reason`, as vLLM does, Anthropic Messages clients receive a
  `stop_sequence` stop reason with that sequence. Chat and Responses clients
  are unaffected.
- When an OpenAI-compatible Chat provider reports an error inside a successful
  stream with an integer HTTP status as its `code`, as vLLM does, streamed
  clients receive the provider's message. Responses and Anthropic Messages
  clients get the Router's error frame, with the status as a string `code` for
  Responses and the mapped error type for Messages. Chat clients get the stream
  as the Router passes it up to the error event, then the Router's error frame
  with a string `code` and no `[DONE]`, however the stream is split into body
  chunks, as long as each frame is within the Router's SSE frame limit.
- A Chat or Responses request that omits its output limit reaches `openai` and
  `responses` backends without one, so the backend's remaining context bounds
  the output. Anthropic Messages requires a limit, so a route to an `anthropic`
  backend sends `max_tokens: 4096` and reports `max_tokens` as `generated`.
  Thinking counts toward that limit. A positive integer in the decision's
  `request_params.default_max_tokens` replaces the 4096 for these requests.
- `x-vsr-client-protocol`, `x-vsr-upstream-protocol`, and
  `x-vsr-protocol-warnings` expose translation details when applicable. See
  [VSR routing headers](../troubleshooting/vsr-headers).
  Diagnostics discovered after streaming headers have been sent are recorded
  in translation-warning metrics and structured debug logs; they cannot be
  added to the already-sent response header.

The repository verifies all three protocols pairwise in codec tests, at the
Envoy ExtProc boundary, and in an 18-cell deployment matrix: three client
formats by three backend formats by buffered or streaming mode. See the
[implemented codec design](../proposals/multi-protocol-adaptor) for the full
verification and extension contract.

## Codex CLI

Codex CLI uses the Responses endpoint and sends several compatibility fields.
The Router handles them as follows:

| Field | Router behavior |
| --- | --- |
| `prompt_cache_key` | Forwarded to `openai` and `responses` backends. A route to an `anthropic` backend omits it and reports `dropped`, because Messages has no equivalent. A decision's `request_params.blocked_params` can remove it before dispatch. |
| `include: ["reasoning.encrypted_content"]` | Accepted and not forwarded. The Router never relays provider-encrypted reasoning, so reasoning items carry no `encrypted_content`. Other `include` values remain unsupported. |
| `client_metadata` | Accepted and not forwarded, because it carries Codex telemetry rather than model input. |
| `input[].internal_chat_message_metadata_passthrough` | Accepted and not forwarded. Codex CLI 0.163 attaches it to every input message to record which instructions the message carries; it is client metadata, not model input. |
| `text.verbosity` | Forwarded to `responses` backends, mapped to `verbosity` for `openai` Chat backends, and reported as `dropped` for `anthropic` Messages backends. The only accepted values are `low`, `medium`, and `high`. |
| Omitted `max_output_tokens` | Codex CLI sends no output limit, so a route to an `anthropic` backend gets `max_tokens: 4096` as described above. A reasoning model can spend all of it before it answers, which Codex CLI reports as `Incomplete response returned, reason: max_output_tokens`. Set a positive integer `request_params.default_max_tokens` on the decisions that send Codex CLI to an `anthropic` backend. |

Each accepted but unforwarded field appears as a `dropped` entry in
`x-vsr-protocol-warnings`.

In Responses output, `usage.input_tokens_details.cache_write_tokens` is omitted
when the backend does not report a cache-write count. Reported zero and positive
counts are preserved. This avoids emitting `null`, which Codex CLI 0.163.0
rejects, without treating unknown usage as a measured zero.

Codex sends each step of its history as separate items: reasoning, the
assistant message and every call. Toward Chat and Messages backends the Router
joins adjacent items in that order into one assistant turn, as those formats
represent a step. Sent one message per item, a backend such as vLLM renders
each item as its own turn, and the model, seeing turns that end in text with
no call, can end its own turn the same way, which ends the Codex run.

Codex can also request reasoning summaries. The Router forwards
`reasoning.summary` to a Responses backend. A Chat Completions or Messages
backend cannot request a summary, so the Router accepts the turn and reports
the dropped setting in `x-vsr-protocol-warnings`. Multi-agent namespace tools
and hosted web search remain unsupported; disable those in the Codex
`config.toml` that points at the Router. These settings were checked with
Codex CLI 0.156.1:

```toml
web_search = "disabled"

[features]
multi_agent = false
```
