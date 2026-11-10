---
title: Protocol Compatibility Matrix
description: Match client-facing inference APIs to supported backend model protocols and understand cross-protocol feature boundaries.
---

# Protocol Compatibility Matrix

Semantic Router supports three inference wire formats on both sides of the
data plane, and the Images API for image generation. A client request is decoded into a protocol-neutral form, routing
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
| OpenAI Images | `POST /v1/images/generations` | Supported | Not supported | Answered only by a decision whose `output` is an image. See [Images API](../api/router#images-api). |

The public listener also serves `GET /v1/models` and Anthropic's
`POST /v1/messages/count_tokens`, which coding agents use to size a file before
reading it. See the
[Router API](../api/router) for the complete method and path inventory,
Responses object operations, and request examples.

System One is a separate typed API:

- **Endpoints:** `POST /v1/systemone` (alias `/v1/decisions`) and
  `GET /v1/systemone/models`.
- **Availability.** It's available in either startup mode when
  `listeners[].systemone.models` publishes a concrete model. Router mode can
  also publish an explicit native recipe through an `api: systemone`
  entrypoint and a matching listener grant.
- **Not translated.** These typed requests don't use the Chat
  protocol-translation matrix or the Chat model allowlist.

See the [System One quickstart](../model-runtime/get-started/quickstart.md).

## Backend model protocols

Set `api_format` on each `providers.models[]` entry. It describes the wire
contract implemented by that model endpoint, not the provider brand.

| `api_format` | Backend request and response shape | Default upstream path | Notes |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | Default when `api_format` is omitted. `backend_refs[].chat_path` can override the Chat path. |
| `responses` | OpenAI Responses | `/v1/responses` | The backend itself must implement the Responses wire contract; enabling the Router's Responses service does not add that API to a backend. |
| `anthropic` | Anthropic Messages | `/v1/messages` | Configure provider authentication and required version headers on the backend ref. vLLM serves this API from 0.11.1, so a vLLM backend keeps `provider: vllm`, which sends the bearer token vLLM checks for `--api-key`. |
| `images` | OpenAI Images | `/v1/images/generations` | Generates images only, for a decision whose [output](../tutorials/decision/output) is an image; it never streams. |

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
verifies both the server certificate chain and its DNS hostname:

- **One hostname per pool.** The supported Envoy runtime shares its TLS
  context within a cluster, so an HTTPS replica pool must keep one hostname.
  Use separate model aliases for different HTTPS hosts.
- **No IP literals.** IP-literal HTTPS targets are rejected, instead of
  silently weakening hostname verification.
- **A CA bundle.** A custom Envoy image used with `vllm-sr serve` must provide
  the system CA bundle at `/etc/ssl/certs/ca-certificates.crt`. Startup
  validation fails when that trust store is missing, rather than silently
  disabling verification.

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
          endpoint: https://gateway.example.com/anthropic
    - name: gpt-via-relay
      provider_model_id: example-gpt
      api_format: openai
      backend_refs:
        - name: relay
          provider: openai-compatible
          endpoint: https://relay.example.com/v1
```

- A reply, a stream event, or an error body that carries a field its
  `api_format` does not define still succeeds. A client that uses the
  backend's format receives the reply unchanged, field included. A client of
  another format receives the translated reply without the field.
- Each such field is reported as a `dropped` diagnostic. It is counted in
  `llm_translation_lossy_total` and logged at debug level, and buffered
  replies list it in `x-vsr-warnings`.
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

The backend-ref setting `compat_gateway`, also spelled
`openai_compat_gateway`, no longer exists. A configuration that still sets
it:

```yaml
providers:
  models:
    - name: claude-via-gateway
      provider_model_id: example-claude
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic-compatible
          endpoint: https://gateway.example.com/anthropic
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

### Images and empty text

Responses requires a detail level on every input image. An image that arrives
without one, as every Messages image and tool-result image does, reaches a
Responses backend with `detail: auto`, the Responses default.

When Chat history is translated to an Anthropic Messages backend, empty text
blocks are omitted:

| Chat input | Sent to Messages as |
| --- | --- |
| An assistant's `content: ""` beside `tool_calls` | `content: null` |
| An empty tool result | Its call ID, with `content: []` |
| An empty system prompt | Nothing; it's omitted |
| Non-empty text, including whitespace | The same text |
| A message with only empty text, or an empty text block that carries a cache directive | Nothing; the request is rejected, because removing it would discard a turn or a cache boundary |

### Anthropic Messages clients on other backends

For an Anthropic Messages client using a Chat Completions or Responses
backend:

| Messages control | What happens |
| --- | --- |
| `thinking.type: adaptive` | The backend model's default reasoning behavior applies; `output_config.effort` is kept. |
| `thinking.display: omitted` | Reasoning is removed from the translated response, including streaming output. |
| `thinking.display: summarized` or `updates` (Claude Code sends these) | Rejected: they reach only Messages backends. |
| `thinking.type: disabled` | Needs a configured reasoning family and an effective backend reasoning-off control. |
| A `context_management` edit `clear_thinking_20251015` with `keep: all` | Omitted, since it has no effect. Edits that would change history are rejected. |
| `cache_control` boundaries, toward a Responses backend | Omitted, since Responses can't represent them. The prompt and tool result still dispatch, and `x-vsr-warnings` reports a `dropped` diagnostic for `cache_control`. |

Other unsupported controls fail with a typed request error.

A `system` message inside `messages`, such as the environment section and the
file-change notices Claude Code inserts during a session, reaches these
backends as a system message in the same place. Many open-model chat
templates accept a system message only first: Qwen3.5 and later reject the
request, and gpt-oss drops the text.

For backends served with such a template, add `messages[].role` to
`global.router.allow_lossy_translation` (see
[Validation and failure behavior](#validation-and-failure-behavior)):

- The text joins the user turn right before it, or becomes a user turn of its
  own after a tool result.
- `x-vsr-warnings` reports an `approximated` diagnostic for
  `messages[].role`.
- Messages backends receive the message in place either way.

### Files a model can't read

A model whose declared capabilities leave out `file_input` can't receive a
file, so a request that holds one fails with `unsupported_capability`. That
matters for coding agents: an agent that reads a PDF returns it as a document
in a tool result, and every later call of the session resends it.

Listing `content.file` in `global.router.allow_lossy_translation` changes
that:

- **Each file becomes a note.** Such a model receives a short note saying the
  file wasn't sent, so the agent can read it another way.
- **Reported.** `x-vsr-warnings` reports a `dropped` diagnostic for
  `content.file`.
- **Capable models are unaffected.** A model that declares `file_input`, or
  declares no capabilities, still receives the file.

### Deferred tool loading

`tools[].defer_loading` asks the provider's tool search to load a tool on
demand. The Messages format carries it, but the Router's protocol-neutral
request has no field for it, so the Router rejects it at ingress with
`unsupported_tools_defer_loading` whatever the backend format, Messages
included. Claude Code sets it on its first call, retries without it, and then
sends every tool definition on every call.

Listing `tools.defer_loading` in `global.router.allow_lossy_translation` keeps
tool search on:

- **Tools in full.** The Router sends each deferred tool in full.
- **References by name.** A `tool_reference` block in a tool result reaches the
  backend as the tool's name.
- **Reported.** `x-vsr-warnings` reports a `dropped` diagnostic for
  `tools.defer_loading`.

### Signed reasoning from Anthropic backends

Reasoning content translated out of an Anthropic backend fails when the
backend attaches a reasoning signature, which its thinking responses carry by
default. Same-format traffic, including Messages clients reading an Anthropic
backend, is unaffected. A Chat Completions or Responses client receives a
typed `unsupported_capability` failure:

- **Buffered requests** get the failure as the response.
- **Streaming requests** get it mid-stream, after the response headers and any
  earlier deltas, because the signature reaches the encoder only with the
  reasoning delta.

Listing `reasoning.signature` in `global.router.allow_lossy_translation` lets
these responses through: the Router drops the signature, keeps the reasoning
text and reports the loss as `approximated`. The signature can't come back on
later turns, so use this only with backends that don't need it, such as vLLM.

### Stored completions

A Chat Completions client asks the provider to keep the completion with
`store: true`. A Messages backend keeps no completions:

- **`store: false`** asks for nothing such a backend doesn't already do, so
  the request reaches it without the member.
- **`store: true`** fails with `unsupported_capability` (`request_storage`),
  because the backend can't keep the completion.

Listing `store` in `global.router.allow_lossy_translation` sends such a
request without `store` and reports a `dropped` diagnostic for `store` in
`x-vsr-warnings`. Chat Completions and Responses backends still receive
`store` as the client sent it.

### Responses state, custom tools and verbosity

- **Stored state.** A Responses client can still use `previous_response_id`
  with a Chat Completions or Messages backend. The Router retrieves and
  materializes the retained history, removes Router-owned object controls,
  then encodes the stateless request in the selected backend format.
- **Custom tools.** Responses custom tools use `type: custom` with a flattened
  `format` object. Their `custom_tool_call` and `custom_tool_call_output`
  items keep free-form input, tool results and call IDs through Chat or
  Responses backends, buffered and streaming. A Messages backend rejects them
  with `unsupported_capability`.
- **Verbosity.** Responses `text.verbosity` maps to the Chat `verbosity`
  field. Messages has no equivalent, so the Router drops this hint and reports
  `text.verbosity` in `x-vsr-warnings`.

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

`api_format` chooses only the backend codec; it doesn't imply Anthropic,
OpenAI or any other runtime provider. Who supplies transport and credentials
depends on the deployment:

| Deployment | Transport and credentials |
| --- | --- |
| Router-owned listeners | Each physical model must declare `backend_refs[].provider`. |
| Metadata-only, `listeners: []` | Left to the external gateway. |
| Local `vllm-sr serve` | Owned by its standalone frontend or generated Envoy configuration, so a backendless physical model isn't accepted. Use the external-gateway deployment profile for that topology. |

To check a new backend:

1. Test the backend directly, with its native path and a minimal request.
2. Send the same request through the Router, using the client API your agent
   harness or other client needs.

A successful health check doesn't validate the request schema, streaming,
tools or error translation.

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
  `x-vsr-warnings` and in `llm_translation_lossy_total`. The list
  also accepts `messages[].role` for system messages inside the conversation
  and `content.file` for files a model cannot read and `tools.defer_loading`
  for deferred tool loading, all described above,
  `reasoning.signature` (see the signed thinking note above), `store` (see
  stored completions above), and every other lossy translation still fails
  closed.
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
- When an OpenAI-compatible Chat provider reports an error inside a
  successful stream, with an integer HTTP status as its `code` as vLLM does,
  streamed clients receive the provider's message:
  - **Responses and Anthropic Messages clients** get the Router's error frame,
    with the status as a string `code` for Responses and the mapped error type
    for Messages.
  - **Chat clients** get the stream as the Router passes it, up to the error
    event, then the Router's error frame with a string `code` and no
    `[DONE]`. That holds however the stream is split into body chunks, as long
    as each frame is within the Router's SSE frame limit.
- A Chat or Responses request that omits its output limit:
  - reaches `openai` and `responses` backends without one, so the backend's
    remaining context bounds the output;
  - reaches an `anthropic` backend with `max_tokens: 4096`, because Messages
    requires a limit, and reports `max_tokens` as `generated`. Thinking counts
    toward that limit, and a positive integer in the decision's
    `request_params.default_max_tokens` replaces the 4096.
- `x-vsr-warnings` and the `client_protocol` and `upstream_protocol` members of
  `x-vsr-trace` expose translation details when applicable. See
  [VSR routing headers](../api/headers).
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
`x-vsr-warnings`.

In Responses output, `usage.input_tokens_details.cache_write_tokens` is omitted
when the backend does not report a cache-write count. Reported zero and positive
counts are preserved. This avoids emitting `null`, which Codex CLI 0.163.0
rejects, without treating unknown usage as a measured zero.

Codex sends each step of its history as separate items: reasoning, the
assistant message and every call. Toward Chat and Messages backends, the
Router joins adjacent items in that order into one assistant turn, as those
formats represent a step.

Sent one message per item, a backend such as vLLM would render each item as its
own turn. The model, seeing turns that end in text with no call, could then end
its own turn the same way, which ends the Codex run.

- **Reasoning summaries.** The Router forwards `reasoning.summary` to a
  Responses backend. A Chat Completions or Messages backend can't request a
  summary, so the Router accepts the turn and reports the dropped setting in
  `x-vsr-warnings`.
- **Unsupported:** multi-agent namespace tools and hosted web search. Disable
  them in the Codex `config.toml` that points at the Router; these settings
  were checked with Codex CLI 0.156.1:

```toml
web_search = "disabled"

[features]
multi_agent = false
```

Point Codex at the Router through a model provider of your own, not
`openai_base_url`:

- **Why.** The built-in `openai` provider opens a Responses WebSocket first,
  which the Router doesn't serve. Codex CLI 0.163.0 retries it for about six
  seconds at the start of every run before it sends the first turn over HTTP,
  and tries again before later turns. A provider defined in `config.toml` uses
  HTTP from the start, since `supports_websockets` defaults to `false`.
- **The model.** Set `model` to a Router entrypoint as well, such as
  `vllm-sr/auto`; a name that matches no entrypoint is sent to a backend model
  of that name.

```toml
model = "vllm-sr/auto"
model_provider = "vllm-sr"

[model_providers.vllm-sr]
name = "vLLM Semantic Router"
base_url = "http://localhost:8899/v1"
wire_api = "responses"
```
