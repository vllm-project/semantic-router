---
title: Protocol Compatibility Matrix
description: Match client-facing inference APIs to supported backend model protocols and understand cross-protocol feature boundaries.
---

# Protocol Compatibility Matrix

Semantic Router supports three inference wire formats on both sides of the
data plane, the Images API for image generation, and the Speech API for
text-to-speech. A client request is decoded into a protocol-neutral form, routing
policy selects a model, and that model's `api_format` selects the backend codec.
Chat, Responses, Messages and Images responses are translated back to the client's
original format. A Speech response is the backend's audio bytes, forwarded unchanged.

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
| OpenAI Speech | `POST /v1/audio/speech` | Supported | Not supported | Routed on the input text to a model with `api_format: speech`. The audio bytes are forwarded unchanged. See [Speech API](../api/router#speech-api). |

The public listener also serves `GET /v1/models` and Anthropic's
`POST /v1/messages/count_tokens`, which coding agents use to size a file before
reading it. See the
[Router API](../api/router) for the complete method and path inventory,
Responses object operations, and request examples.

System One is a separate typed API:

- **Endpoints:** `POST /v1/systemone` (alias `/v1/decisions`) and
  `GET /v1/systemone/models`.
- **Availability.** It's available in either startup mode when
  `listeners[].apis.systemone.models` publishes a concrete model. Router mode can
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
| `speech` | OpenAI Speech | `/v1/audio/speech` | Speaks the input text. `stream: true` and `stream_format` are rejected. The response is the backend's audio, not a JSON completion, so response policy, the response cache, token limits and cost estimates do not apply. Request-text checks still run, on `input` only: `instructions`, `ref_text` and the other fields are forwarded unchecked. A model that cannot speak fails closed with `unsupported_capability`. |

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
  context within a cluster, so an HTTPS replica pool must keep one hostname
  and one `tls` block. Use separate model aliases for different HTTPS hosts.
- **No IP literals.** IP-literal HTTPS targets are rejected, instead of
  silently weakening hostname verification.
- **A CA bundle.** A backend's `tls.ca_file`, else
  `providers.defaults.tls.ca_file`, names the bundle its certificate must
  chain to; see [Trust a private CA](backend-target-compatibility#trust-a-private-ca).
  Without either, a custom Envoy image used with `vllm-sr serve` must provide
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
| Multiple response candidates (`n` above 1) | Buffered only. A streamed request fails with `stream_multiple_choices` before dispatch, unless `request_params` `max_n: 1` lowers it to one choice. A Looper decision, such as Ratings, that would stream every model's call fails with the same error | Not supported | Not supported |
| Prompt-cache directives | Supported | Not supported | Supported |
| Prompt cache key (`prompt_cache_key`) | Supported | Supported | Not forwarded; reported as `dropped` |
| Reasoning token budget | Supported extension | Not supported | Supported |
| Seed and frequency or presence penalties | Supported | Not supported | Not supported |
| `top_k` sampling | Supported extension | Not supported | Supported for nonnegative values |
| `min_p`, repetition penalty, and cache salt | Supported extensions | Not supported | Not supported |
| Token logit bias (`logit_bias`) | Supported | Not supported | Not supported |
| Stop sequences | Supported | Not supported | Supported |
| Native response or conversation state fields | Not supported | Supported | Not supported |
| Deferred tool loading (`tools[].defer_loading`) | Not supported; opt-in sends tools in full | Not supported; opt-in sends tools in full | Not supported; opt-in sends tools in full |
| Anthropic server tools, their result blocks and citations | Not supported | Not supported | Supported toward an `anthropic` or `anthropic-compatible` backend |

This table describes codec representation, not model capability. For example,
an OpenAI-compatible server can accept the Chat request shape while rejecting
images or tools for a particular model. Qualify the actual endpoint and model
revision before adding them to a routing pool.

### Images and empty text

Responses requires a detail level on every input image. An image that arrives
without one, as every Messages image and tool-result image does, reaches a
Responses backend with `detail: auto`, the Responses default.

A Chat tool message carries text only. Toward a Chat backend, the images a
tool result holds, as Claude Code's `Read` of a picture or Codex's
`view_image` returns, go in one user message right after the tool messages
that answer the turn, opened with "The tool results above returned these
images."; a tool message whose result held only images says that they follow.
`x-vsr-warnings` reports the move as `approximated` with field
`messages[].content.tool_result.image`. Other media in a tool result, such as
a document, are still refused with `tool_result_media`.

Messages has no image detail level and no document file name. Toward an
`anthropic` backend, an image `detail` other than `auto` and a file `filename`
are omitted, and `x-vsr-warnings` reports each as `dropped`.

When Chat history is translated to an Anthropic Messages backend, empty text
blocks are omitted:

| Chat input | Sent to Messages as |
| --- | --- |
| An assistant's `content: ""` beside `tool_calls` | `content: null` |
| An empty tool result | Its call ID, with `content: []` |
| An empty system prompt | Nothing; it's omitted |
| Non-empty text, including whitespace | The same text |
| A message with only empty text, or an empty text block that carries a cache directive | Nothing; the request is rejected, because removing it would discard a turn or a cache boundary |

### Empty answers

A model can end its turn with nothing to say, which Anthropic answers with
`end_turn` and no content block. The Router passes that on as an empty
completion, buffered or streamed: a Messages client gets `content: []` with
`end_turn`, a Chat Completions client an assistant message with `content: ""`
and `finish_reason: stop`, and a Responses client a `completed` response with
no output. Router Replay records it `completed`, with its usage and cost. A
reply with no output under any other stop, such as a tool call that names no
call, and a stream without its stop reason or its `message_stop`, are still
invalid upstream answers.

### Chat request fields the Router refuses

The Router fails closed on a Chat Completions field it can't carry to every
backend: the request gets `400` with the code below and reaches no backend. A
value that asks for nothing a backend wouldn't do without the field is
accepted instead. The backend receives the request without the field, and
`x-vsr-warnings` reports a `dropped` diagnostic for it. A `null` value counts
as no field, and `logprobs: false` asks for the default, so neither is
reported.

| Field | Code | Accepted |
| --- | --- | --- |
| `audio` | `unsupported_audio` | None |
| `function_call` | `unsupported_function_call` | None |
| `functions` | `unsupported_functions` | None |
| `logprobs` | `unsupported_logprobs` | `false` |
| `modalities` | `unsupported_modalities` | Exactly `["text"]` |
| `moderation` | `unsupported_moderation` | None |
| `prediction` | `unsupported_prediction` | None |
| `prompt_cache_options` | `unsupported_prompt_cache_options` | None |
| `prompt_cache_retention` | `unsupported_prompt_cache_retention` | None |
| `safety_identifier` | `unsupported_safety_identifier` | Any string: an end-user ID for the provider's abuse monitoring, which a backend doesn't use |
| `service_tier` | `unsupported_service_tier` | `"auto"` and `"default"` |
| `top_logprobs` | `unsupported_top_logprobs` | None |
| `web_search_options` | `unsupported_web_search_options` | None |
| `messages[].audio`, output audio in replayed history | `unsupported_output_audio` | None |
| `messages[].function_call`, a legacy function call in history | `unsupported_legacy_function_call` | None |
| `messages[].tool_calls[].function.TokenizedArguments` | `unsupported_messages_tool_calls_function_tokenized_arguments` | None |
| A content part the message's role can't hold, such as an image in a system message | `unsupported_content` | None |
| A tool whose `type` is not `function` or `custom` | `unsupported_tool` | None |
| `tool_choice` of type `allowed_tools` | `unsupported_tool_choice` | None |
| A `response_format` other than `text`, `json_object` or `json_schema` | `unsupported_output_format` | None |

A Responses request's `service_tier` and `safety_identifier` follow the same
rule. Its other top-level fields the Router can't carry, `background`,
`context_management`, `conversation`, `max_tool_calls`, `moderation`,
`prompt`, `prompt_cache_options`, `prompt_cache_retention` and
`top_logprobs`, fail with `unsupported_<field>`.

### Anthropic Messages clients on other backends

For an Anthropic Messages client using a Chat Completions or Responses
backend:

| Messages control | What happens |
| --- | --- |
| `thinking.type: adaptive` | The backend model's default reasoning behavior applies; `output_config.effort` is kept. |
| `thinking.display: omitted` | Reasoning is removed from the translated response, including streaming output. An answer whose output limit went to reasoning comes back with no content blocks and the `max_tokens` stop reason, buffered or streamed. |
| `thinking.display: summarized` or `updates` (Claude Code sends these) | The backend's reasoning comes back as thinking, as it does with no display, since the Router can't summarize it. `x-vsr-warnings` reports an `approximated` diagnostic for `thinking.display` when the model has a reasoning family and the decision doesn't turn reasoning off. |
| `thinking.type: disabled` | Needs a configured reasoning family and an effective backend reasoning-off control. The backend runs with reasoning off whatever the decision chose, and the `reasoning` member of `x-vsr-trace` reads `off`. |
| A `context_management` edit `clear_thinking_20251015` with `keep: all` | Omitted, since it has no effect. Edits that would change history are rejected. |
| `cache_control` boundaries, toward a Responses backend | Omitted, since Responses can't represent them. The prompt and tool result still dispatch, and `x-vsr-warnings` reports a `dropped` diagnostic for `cache_control`. |

Other unsupported controls fail with a typed request error.

On every backend, `max_tokens` covers thinking as well as the answer. A
request without `thinking` whose decision turns reasoning on can spend a small
`max_tokens` on thinking and come back without text. With
`providers.defaults.reasoning_min_output_tokens` set, such a request below that
limit reaches the backend with the model's reasoning off; see
[Model reasoning](model-reasoning#turn-reasoning-off-for-small-output-limits).

A stop sequence can also end such a request before any text, because
backends such as vLLM match `stop_sequences` against the thinking as well as
the answer. Such a request that sets `stop_sequences` therefore reaches the
backend with the model's reasoning off; see
[Model reasoning](model-reasoning#turn-reasoning-off-for-requests-with-stop-sequences).

Replies reach these clients in the shape an `anthropic` backend gives them:

- **Thinking signatures.** Every thinking block carries a `signature`, empty
  when the backend gave none. A stream sends it on `content_block_start` and
  ends the block with an empty `signature_delta`. When a later turn sends such
  a block back, the Router accepts it whatever the backend's format.
- **Identifiers.** The message id starts with `msg_` and each tool call id
  with `toolu_`. Both are derived from the backend's ids, so a buffered and a
  streamed reply carry the same ones, and the `tool_result` blocks the client
  sends back pair with their calls.
- **Input usage on `message_start`.** These backends report usage only when
  the answer ends, so a stream's `message_start` reports the Router's estimate
  of the input tokens. The final `message_delta` reports the backend's count.

### Request extension fields

Some backends accept top-level request members their wire contract does not
declare — a vLLM fork may read `vllm_xargs` for an acceleration algorithm,
for example. The Router fails such requests closed: an unmodelled member
would be silently erased by the neutral translation that model selection
and plugins perform, so it is rejected instead of lost.

Listing a member in `global.router.allow_request_extension_fields` changes
that:

- **Captured at ingress.** The Router lifts the member out of the request
  before decoding, bounded by an extension budget, and carries it opaquely
  through routing, model rewrite, and policy plugins.
- **Merged on same-format dispatch.** The backend receives the member with
  its original value; a replayed request keeps the original bytes.
- **Dropped across formats, reported.** Dispatch that translates the
  request to another wire format drops the member with a `dropped`
  diagnostic in `x-vsr-warnings`.
- **Never interpreted.** The Router does not read the member's value, and
  it never influences routing decisions. Every member not listed keeps
  failing closed.

```yaml
global:
  router:
    allow_request_extension_fields:
      - vllm_xargs
```

### System and developer messages inside the conversation

System and developer messages before the first conversation message lead
the request: Chat Completions and Responses backends receive them first, and
a Messages backend receives them in `system`. A later one stays where the
client put it, whichever client protocol sent it. Claude Code inserts one
for its environment section and its file-change notices, and Chat and
Responses clients send one when an instruction changes during a session.

- Chat Completions and Responses backends receive it as a system or
  developer message in the same place.
- A Messages backend receives it as a system message in the same place.
- A system or developer message between a tool call and its output moves
  after the output for a Chat Completions or Messages backend, whichever
  client protocol sent it. This also applies to same-format requests. The
  backend needs the output right after the call. `x-vsr-warnings` reports an
  `approximated` diagnostic for `input` on Responses clients and `messages`
  on Chat or Messages clients. A Responses backend keeps the original order
  unless `messages[].role` converts the instruction to user text; that
  projection moves it after the output and reports both diagnostics.
- Empty instruction text creates no user turn under `messages[].role`.
  Chat and Responses clients' empty instructions are omitted for Messages
  backends, just like empty top-level instructions. A per-message
  `output_config.effort` control is retained. Instructions omitted by
  `messages[].role` produce no placement warning. Re-encoding a Messages
  request with no messages left fails with `anthropic_messages_required`.

For templates that accept one system message, only first, see
[System messages a chat template refuses](#system-messages-a-chat-template-refuses).

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

A `tool_addition` block, which Claude Code puts in a `role: "system"` message
when a deferred tool becomes available mid-conversation, adds the tool it
defines to the request's tools, after the declared ones and once per name, on
every backend format. The system message keeps its other blocks, and a message
that held nothing else is left out.

### Auto mode's classifier context

Claude Code in auto mode sends a top-level `safeguards` array for Anthropic's
server-side classifier review. The Router carries it unchanged to a Messages
backend. A Chat or Responses backend has no use for it, so the Router leaves it
out and `x-vsr-warnings` reports a `dropped` diagnostic for `safeguards`. The
array holds Claude Code's working directory, permission rules and git state, so
on a decision with the `masking` plugin its string values are masked like a
structured tool result's, and an array that will not decode refuses the request.

### Anthropic server tools

A Messages tool with a `type`, such as `web_search_20250305`,
`web_fetch_20250910` or `code_execution_20250825`, is one Anthropic defines,
and the server tools among them run on Anthropic's side. Claude Code's
WebSearch sends `web_search_20250305` in a request of its own. The Router has
no neutral form for these tools, so it carries every typed tool as written,
including those a client runs, such as `bash_20250124`, whose calls come back
as ordinary `tool_use` blocks:

- **Toward Anthropic.** A Messages backend whose `provider` is `anthropic` or
  `anthropic-compatible` receives the tool definitions unchanged, and the
  `server_tool_use` and `*_tool_result` blocks and the citations on text that
  a client sends back in later turns. Its reply, buffered or streamed,
  reaches a Messages client with every block and citation, and a reply the
  Router passes through as sent keeps Anthropic's `usage.server_tool_use`
  counts.
- **Toward any other backend.** A Chat Completions or Responses backend, or a
  Messages backend with another provider, such as `vllm`, cannot run the
  tool. A request that declares one is refused before it is sent, with HTTP
  400 and the code `unsupported_server_tool`, for example `Anthropic tool
  "web_search" (web_search_20250305) runs only on an Anthropic backend, and
  model "qwen-chat" is served by provider vllm over openai.chat.v1`. Selection
  and fallback pass over such a model when the decision has one that runs the
  tool, and a decision with none returns the refusal. Router Replay records it
  as `request_rejected` with the status the client got.
- **History on another backend.** A later turn that declares no server tool
  but carries the blocks an earlier one left can still go to any backend.
  The Router leaves out the calls and results, whose search results Anthropic
  encrypts, keeps the text that cites them without its citations, and
  `x-vsr-warnings` reports a `dropped` diagnostic for `content.server_tool`.
- **Masking.** On a decision with the `masking` plugin, the string values of
  these blocks and citations are masked like a structured tool result's.
- **Cost.** The Router prices the tokens Anthropic reports, which include the
  search results a turn reads. It does not price Anthropic's fee per search,
  $10 per 1,000 web searches on the Claude API on 2026-11-17, or code
  execution time: `pricing` has no per-request rate, so neither enters a
  call's cost in Router Replay or a spend budget.

The request field `container`, which picks up a code execution container
again, stays unsupported, so each request starts in a new one.

### Sampling controls a backend rejects

Some backends reject a sampling control their wire format defines. A vLLM
backend that runs speculative decoding, for example, answers any request that
carries `min_p` or `logit_bias` with HTTP 400. List such controls in the
model's `unsupported_capabilities`:

```yaml
providers:
  models:
    - name: qwen3-mtp
      api_format: chat_completions
      unsupported_capabilities: [sampling_min_p, sampling_logit_bias]
      backend_refs:
        - name: mtp-pool
          endpoint: mtp-server:8000
          provider: vllm
```

- **Left out for that model.** A request sent to the model, as the decision's
  choice or as a fallback candidate, reaches it without the listed fields.
- **Reported.** `x-vsr-warnings` reports a `dropped` diagnostic for each field
  left out, `min_p` or `logit_bias`.
- **Other models are unaffected.** Another model of the same decision still
  receives the fields.

The list accepts `sampling_min_p`, `sampling_logit_bias` and `developer_role`
([below](#developer-messages-a-backend-drops)); any other name fails at load. The gateway spreads a model's requests across its
`backend_refs`, so list a control when any of them rejects it. To remove a
control for every model a decision routes to, list it in the decision's
[`request_params`](../tutorials/plugin/request-params) `blocked_params`
instead.

### System messages a chat template refuses

A request can carry several system and developer messages. A Chat
Completions client may give each instruction its own system message, a
Responses request has its `instructions` and developer items, and coding
agents insert system messages inside the conversation. Many open-model chat
templates accept one system message, only first: Qwen3.5 and later reject the
request with "System message must be at the beginning.", and gpt-oss drops
the text. The same request then passes or fails with the model the decision
picks.

For backends served with such a template, add `messages[].role` to
`global.router.allow_lossy_translation` (see
[Validation and failure behavior](#validation-and-failure-behavior)):

```yaml
global:
  router:
    allow_lossy_translation: ["messages[].role"]
```

- **One system message, first.** A Chat backend receives the system and
  developer messages that lead the conversation as one system message, their
  texts in order with a blank line between them: a Chat request's leading
  system and developer messages, or a Responses request's `instructions` and
  leading developer and system items. A `cache_control` boundary stays at the
  end of the text that set it. A request with one leading system message
  reaches the backend unchanged.
- **Later messages as user text.** A system or developer message inside the
  conversation reaches Chat and Responses backends as user text in the same
  place, whichever client protocol sent it: it joins the user turn right before
  it, or becomes a user turn of its own after an assistant turn or a tool
  result. One between a tool call and its results first moves after the results.
- **Reported.** `x-vsr-warnings` reports one `approximated` diagnostic for
  `messages[].role`.
- **Leading instructions on other backends are not merged.** Responses
  backends keep each leading instruction's role, and Messages backends receive
  them in `system`. Messages backends receive later instructions as system
  messages inside the conversation, with or without the opt-in.

Without the opt-in, Chat and Responses backends receive each system and
developer message in its own role and position, except for instructions moved
out of a tool call's span as described above.

### Developer messages a backend drops

OpenAI reads a `developer` message as an instruction, like a `system` one. Chat
Completions clients send instructions in the `developer` role, and a Responses
request's `instructions`, and its `developer` input items, reach a Chat backend
as developer messages. DeepSeek V4's encoding, which vLLM follows, treats a
developer message as a turn of its own instead: when the model thinks and the
request has no tools, it drops every developer message before the last user
message, so the model never sees those instructions. List `developer_role` in
such a model's `unsupported_capabilities`:

```yaml
providers:
  models:
    - name: deepseek-v4-flash
      api_format: chat_completions
      unsupported_capabilities: [developer_role]
      backend_refs:
        - name: deepseek-pool
          endpoint: deepseek-server:8000
          provider: vllm
```

- **Sent as system for that model.** Every developer instruction and message
  of a request sent to the model, as the decision's choice or as a fallback
  candidate, reaches it in the `system` role.
- **Reported.** `x-vsr-warnings` reports an `approximated` diagnostic for
  `messages[].role`.
- **Other models are unaffected.** Another model of the same decision still
  receives the `developer` role.

With `messages[].role` listed, a Chat backend receives the leading developer
messages in its one system message
([above](#system-messages-a-chat-template-refuses)), and the request reports
that diagnostic alone.

### Control tokens in user and tool text

A model server such as vLLM renders the chat template to one string and
tokenizes it with the tokenizer's added tokens recognized, so text that
spells one of those tokens becomes the token. `<think>` and `</think>` typed
by a user, or returned inside a tool result, become a Qwen3.5 model's
reasoning delimiters (its tokenizer does not even mark them special), and a
typed `<|im_end|>` ends the user's turn.

The Router keeps such text text. For a model whose own tokenizer it has
(`providers.models[].tokenizer`, or the Hugging Face repo it counts with: see
[Check prompts against the context window](model-configuration#check-prompts-against-the-context-window)),
it puts U+2060 WORD JOINER, a character with no width, after the first
character of every added token's text in user and tool text before the
request leaves:

- **Every added token, every protocol.** Special or not, as the model
  server's tokenizer matches them, in Chat, Responses and Messages requests,
  for the decision's model and for each fallback candidate.
- **Only user and tool text.** The text of user messages and of tool results
  is escaped. System and developer instructions, the assistant's turns and
  tool calls go as the client sent them. A token split across two text parts
  of one message is broken where the parts meet.
- **Nothing else changes.** Text without an added token's text reaches the
  backend byte for byte. A model that repeats escaped text may repeat the
  word joiner too.
- **Models without their own tokenizer** (none named, or one that has not
  loaded) receive the text unchanged.

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

A Messages client sends that thinking back, signature included, on every
later turn. When a session an `anthropic` backend answered moves to a Chat
Completions or Responses backend, the request fails before routing with
`unsupported_capability`, and the message names the first signed block:

```text
messages[1].content[0].signature: the selected model cannot accept signed thinking; an operator can accept this loss by listing reasoning.signature in global.router.allow_lossy_translation
```

The same setting lets these requests through:

- **Text kept.** Such a backend receives each thinking block's text without
  its signature.
- **Reported.** `x-vsr-warnings` reports an `approximated` diagnostic for
  `reasoning.signature`.
- **Anthropic backends are unaffected.** An `anthropic` backend, as the
  decision's choice or as a fallback candidate, still receives the signature.

A thinking block whose signature is empty, `"signature": ""`, carries none, so
it reaches every backend without the setting.

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
  then encodes the stateless request in the selected backend format. The
  developer and system items at the start of the stored input come back as
  instructions, after the current request's top-level `instructions`. The
  current input follows the stored turns, so its developer and system items
  stay in their place there, at its start or later. The earlier request's
  top-level `instructions` are not carried over.
- **Conversations.** The Router keeps no Conversations objects, so it can
  neither load a conversation's items nor add a response to them. A request
  that names a `conversation`, as an ID or an object, fails with `400`
  `unsupported_conversation` and `param: "conversation"` before it reaches a
  backend; `conversation: null` names none. Continue a stored response with
  `previous_response_id` instead.
- **Custom tools.** Responses custom tools use `type: custom` with a flattened
  `format` object. Their `custom_tool_call` and `custom_tool_call_output`
  items keep free-form input, tool results and call IDs through Chat or
  Responses backends, buffered and streaming. A Messages backend rejects them
  with `unsupported_capability`.
- **Verbosity.** Responses `text.verbosity` maps to the Chat `verbosity`
  field. Messages has no equivalent, so the Router drops this hint and reports
  `text.verbosity` in `x-vsr-warnings`.
- **Response objects.** The response object, buffered and in every streamed
  `response.*` event, repeats the request's `instructions`, `tools`,
  `tool_choice`, `parallel_tool_calls`, `max_output_tokens`, `temperature`,
  `top_p`, `text`, `reasoning`, `metadata`, `truncation`, `user` and
  `prompt_cache_key` as the client sent them, whatever routing changed for the
  backend. A member the request leaves out or sets to `null` keeps its
  default. `created_at` is the backend's creation time, or the Router's when
  the backend reports none, as Messages backends don't. When the output limit
  ends the answer, its message item is `incomplete`, in
  `response.output_item.done` and in the terminal response.
- **Streamed items.** A stream closes each output item before the next one
  starts, as OpenAI's streams do: the item's `response.output_text.done` or
  `response.reasoning_text.done`, its `response.content_part.done` and its
  `response.output_item.done` come before the next
  `response.output_item.added`, and `response.completed` or
  `response.incomplete` holds every item. A Chat Completions backend ends its
  outputs only when its stream ends, so the Router closes an item when the
  backend moves on to another one: from reasoning to the answer, from the
  answer to a function call, from one call to the next. A function call whose
  arguments don't form a JSON object yet stays open until they do, so a
  backend that interleaves calls still streams them, and content that comes
  back to an item already closed, such as answer text after reasoning that
  followed it, goes into a new item of its kind. Responses and Messages
  backends end each item themselves, and their order is kept.
- **Reasoning text.** Reasoning text is a content part of its reasoning item:
  `response.content_part.added` and `response.content_part.done` with a
  `reasoning_text` part bracket its `response.reasoning_text.delta` and
  `response.reasoning_text.done` events, whatever the backend format. A
  reasoning summary keeps its own `response.reasoning_summary_part.added` and
  `response.reasoning_summary_part.done` events.

### Structured output with tools

Model servers such as vLLM apply an output format (Chat `response_format`,
Responses `text.format`, Messages `output_config.format`) to the whole
completion. A model given callable tools and a JSON Schema then can't call
them; it answers with made-up JSON in the schema. The Router keeps the OpenAI
contract for such a turn: the model may call tools, and only a final answer
without tool calls is held to the schema.

This applies to a request with `json_schema` or `json_object` output, at least
one tool, and a `tool_choice` other than `none`:

- **First call without the format.** The backend receives the turn with its
  tools and `tool_choice` as the client sent them, without the output format.
- **A tool call is the answer.** An answer that calls a tool reaches the client
  as the backend gave it. So does one that refuses or doesn't end on its own:
  cut at the output limit, filtered, or failed.
- **A final answer is asked for again.** For an answer without tool calls, the
  Router sends the same backend a second call with the same request,
  `tool_choice: none` and the output format, and returns that call's answer. A
  Chat Completions backend receives this call without its tools, like any
  `tool_choice: none` request; Messages and Responses backends keep them. The
  first answer's text and reasoning are discarded.
- **Usage counts both calls.** The answer's usage, `x-vsr-cost`, spend, the
  rate limiter's token counts and the Router Replay record add up the two
  calls. The replay record also carries a `structured_output` outcome with
  each call's output tokens.
- **Streams wait for the answer.** A streamed request is held until the first
  call ends. A tool call then arrives as the stream the backend sent, and a
  final answer arrives as the second call's answer, all at once. Such a turn
  doesn't stream incrementally.
- **Failure.** When the second call fails, the client receives the backend's
  error with its status, or `503` (`504` on timeout) when no backend
  answered. A streamed request receives it as the stream's error event. The
  first answer isn't sent instead, because it doesn't follow the format.
- **Timeout.** The second call is bounded by the shortest `total_timeout` or
  `per_try_timeout` that the decision's and the model's `reliability` set,
  else by 300 seconds, and always by the request's own deadline. Behind
  Envoy, the ext_proc `message_timeout` bounds it too.

Every backend format takes the two calls, including APIs that already allow
tool calls beside an output format. `x-vsr-warnings` still reports what the
format loses toward the backend, such as `output_format.name` toward Messages,
although the first call carries no format. A format the backend can't carry is
refused before any call, as without tools.

These requests keep one call with the format: `tool_choice: none`, no tools,
several candidates (`n` above 1), fallback candidates, Looper decisions, and
metadata-only models whose transport an external gateway owns.

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
  backend formats match. Unknown or unsupported request fields fail closed
  with `400` `invalid_request_error`, naming the field as OpenAI does:

  | Request | Code | Message |
  | --- | --- | --- |
  | A member the API does not define | `unknown_parameter`, `param` the member's path | `Unrecognized request argument supplied: new_field`; inside a message, `messages[].new_field` |
  | Valid JSON with a value of the wrong type | `invalid_type`, `param` the member's path | `Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.` |
  | A body that is not JSON | `invalid_json` | `request JSON is invalid` |
  | A member the Router knows but cannot carry | `unsupported_<field>` | Names the field |

  Anthropic Messages clients get the same message in Anthropic's
  `invalid_request_error`, which has no code field.
- A request whose history holds a tool call that no later tool result
  answers fails with `400` `unanswered_tool_call` and reaches no backend,
  which could otherwise answer in place of the tool. The message is the one
  the client's API gives for such a history, naming the call:

  | Client API | `param` | Message |
  | --- | --- | --- |
  | Chat Completions | `messages` | `An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: call_1` |
  | Responses | `input` | `No tool output found for function call call_1.`, or `custom tool call` for a custom tool |
  | Anthropic Messages | none | ``messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_1. Each `tool_use` block must have a corresponding `tool_result` block in the next message.`` |

  A result anywhere later in the history answers a call, and a client that
  interrupts or denies a call answers it with a result that says so. A
  Responses request with `previous_response_id` is judged with the stored
  history: a function call that the stored response ended with is answered by
  a `function_call_output` in the request's `input`.
- A backend's `4xx` error reaches the client in the client's error format,
  with the Router's codes and the field names the client sent; see
  [request budget errors](../api/router#request-budget-errors).
- Chat requests accept and don't forward the fields that client libraries add
  to the assistant messages they replay: LiteLLM's `provider_specific_fields`,
  which holds the reply fields its message type lacks, and `parsed` and
  `tool_calls[].function.parsed_arguments`, the openai SDK's own parse
  results. One that carries a value is reported as `dropped`. Like the other
  Responses input items, a reasoning item may carry `status`, which the openai
  SDK's stream helper sets.
- A function tool whose `parameters` is `null`, which typed OpenAI clients
  send for a tool that takes no arguments, is read like a tool without
  `parameters`, in Responses `tools[]` and Chat `tools[].function` alike:
  every backend receives the schema `{"type":"object"}` in place of the
  `null`.
- Backend replies are decoded into the same contract. A reply field that its
  `api_format` does not define is dropped and reported instead of failing the
  reply; see
  [Response fields the Router does not model](#response-fields-the-router-does-not-model).
- Cross-protocol requests preserve shared semantics. Target-specific features
  that cannot be represented return a typed protocol error. A
  `lossy_translation` error for a loss an operator may accept, such as
  `tool_result.is_error`, names the `global.router.allow_lossy_translation`
  setting that accepts it, and so do the `unsupported_tools_defer_loading`
  error and the `unsupported_capability` error for signed thinking in a
  request. The `unsupported_capability` errors for files, signed reasoning in
  a response and stored completions don't; their sections above name the
  setting.
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
  also accepts `messages[].role` for system and developer messages a chat
  template refuses and `content.file` for files a model cannot read and `tools.defer_loading`
  for deferred tool loading, all described above,
  `reasoning.signature` for signed thinking in a response or in a Messages
  request's history (see signed reasoning above), `store` (see stored
  completions above), and every other lossy translation still fails closed.
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
- vLLM's serving metadata never fails a reply, whatever its value:
  `prompt_logprobs`, `prompt_token_ids`, `prompt_text`,
  `prompt_routed_experts`, `kv_transfer_params`, `ec_transfer_params` and
  `metrics` at the top level of a reply or stream chunk, and `token_ids`,
  `routed_experts` and `weight_versions` in its choices. No client receives
  them, nor the server's `system_fingerprint`, which names vLLM's version,
  commit and parallel layout; the Router has no stable fingerprint of its own
  to send instead, and `x-vsr-config-version` names the configuration that
  served. A Chat client of a Chat backend receives the rest of the reply as
  the backend sent it, buffered and streamed. A metadata value other than null
  or empty is reported as `dropped`. In a vLLM-Omni stream, a chunk whose
  `modality` is not `text` fails with `unsupported_upstream_audio` (or
  `unsupported_upstream_modality`), because its `delta.content` carries
  encoded media, not answer text.
- A request whose `tool_choice` is `none` reaches a Chat Completions backend
  without its `tools`, `tool_choice` and `parallel_tool_calls`, and
  `x-vsr-warnings` reports `tools` as `dropped`. Model servers such as vLLM
  render declared tools into the prompt whatever the choice, so a model could
  start a call the server then drops, leaving only the sentence before it.
  Messages backends receive the tools, which they require while the history
  holds `tool_use` blocks, and Responses backends receive them with
  `tool_choice: "none"`, which the Responses API enforces.
- A Chat provider ends a forced named tool call with `finish_reason: "stop"`.
  Anthropic Messages clients receive `stop_reason: "tool_use"` for a turn that
  ends that way and carries `tool_use` blocks. A turn that ends at the output
  limit, on a stop sequence, or on a refusal keeps that reason.
- When the output limit cuts a buffered reply inside a tool call, the client
  gets the reply with the limit stop: `finish_reason: "length"` for Chat, an
  incomplete response with reason `max_output_tokens` for Responses, and
  `stop_reason: "max_tokens"` for Anthropic Messages. Chat and Responses
  clients get the call's arguments as the backend sent them. Because
  `tool_use.input` must be an object, Messages clients get the members the
  model completed before the cut, and the Router reports an `approximated`
  diagnostic for `content.tool_use.input`. Only the reply's last call can be
  cut. Arguments that don't start a JSON object, or a cut call that more
  output follows, still fail the reply.
- A backend can report a call the output limit cut as stopping for its tool
  calls: vLLM's buffered Chat reply says `finish_reason: "tool_calls"`
  whenever its tool parser found a call, and some parsers close the cut
  arguments as `{}`. When a reply that stops for its tool calls counts at
  least as many output tokens as the limit the backend received (the
  request's `max_tokens`, `max_completion_tokens` or `max_output_tokens` as
  the decision's `request_params` leave it), the Router reads it as stopped
  at that limit: every client gets the limit stop and the call
  exactly as described above, and the reply's `x-vsr-warnings` reports an
  `approximated` diagnostic for `response.stop_reason`. A stream that ends with
  `tool_calls` after that many tokens, as some vLLM tool parsers end one, ends
  the same way; a Chat client of a Chat backend gets the stream as the backend
  sent it with only that `finish_reason` changed to `length`. Without usage,
  below the limit, or with several choices, whose usage counts them all, the
  backend's stop stands.
- A Chat or Responses client can send such a cut call back in the history of
  its next request, as agents keep it. A function call whose arguments stop
  before their JSON object closes is accepted as the last call of its
  assistant turn: the last of an assistant message's `tool_calls`, or the
  last `function_call` item before a tool result, a user message or the end
  of the input. Every backend receives the members the model completed before
  the cut, as for a Messages client above, and `x-vsr-warnings` reports an
  `approximated` diagnostic for `tool_call.arguments`. A cut call that
  another call or text of its turn follows, and arguments that don't start a
  JSON object or break JSON before the cut, still fail with
  `invalid_tool_call`. A Messages client's history carries `tool_use.input`
  as an object, so it never holds a cut call.
- A Chat stream chunk can carry several `tool_calls` entries with the same
  `index`, as vLLM's last chunk does when its tool parser flushes the rest of
  a call. They continue one call in order, as OpenAI's SDKs join them.
  Entries with the same index that name different calls fail with
  `duplicate_stream_tool_index`.
- A Chat Completions stream whose request sets `stream_options.include_usage`
  carries `"usage": null` on every chunk before its final usage chunk, as
  OpenAI's does, whatever the backend's format. The final chunk carries the
  totals and `choices: []`. A Chat backend's chunks keep their other bytes.
  Without `include_usage`, no chunk carries `usage`.
- An Anthropic `model_context_window_exceeded` stop reaches Chat clients as
  `finish_reason: "length"` and Responses clients as an incomplete response
  with reason `max_output_tokens`, as vLLM reports output that reaches
  `max_model_len`. The Router reports an `approximated` diagnostic for
  `response.stop_reason`. A `pause_turn` stop still fails for these clients.
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
- When the output limit ends a streamed tool call inside its arguments, or
  right after its name with none, the Router passes the partial call on with
  the limit stop, as the providers do: `max_tokens` for Messages, `length` for
  Chat, and `incomplete` with `max_output_tokens` for Responses. For
  Chat-to-Responses streaming only, a function call confirmed to have finished
  normally with omitted, empty, or whitespace-only arguments is normalized to
  `{}`. The joined argument deltas, arguments-done event, completed item, and
  terminal output agree; whitespace-only argument prefixes are held until an
  object follows or the call finishes. This is a wire-level guarantee, not tool
  schema validation: required-field validation remains with the tool consumer.
  A wire `arguments: null` can be a streaming placeholder: later object argument
  text is accepted, but a normally finished call with only null and no non-whitespace
  argument text is rejected. Arrays, other non-objects, and malformed non-empty JSON
  still end the stream with `invalid_stream_tool_arguments`; other translation
  directions retain their existing validation. A cut response is not cached,
  stored, or kept as the replay response body.
- A Responses reasoning item can carry different non-empty `encrypted_content`
  strings in `response.output_item.done` and the corresponding output item in
  `response.completed` or `response.incomplete`. The Router accepts this
  difference while checking every other item field, output count, and output
  order. A non-empty string becoming missing, null, or empty still fails with
  `stream_terminal_output_mismatch`. For Responses clients, the Router keeps
  the item-done ciphertext and reuses it in terminal output. Other client
  formats omit it with the reasoning continuity diagnostic.
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
| `include: ["reasoning.encrypted_content"]` | Forwarded to OpenAI-hosted Responses targets. Responses clients receive `encrypted_content` verbatim on reasoning items, including encrypted-only items. Other `include` values remain unsupported. |
| `client_metadata` | Accepted and not forwarded, because it carries Codex telemetry rather than model input. |
| `input[].internal_chat_message_metadata_passthrough` | Accepted and not forwarded. Codex CLI 0.163 attaches it to every input message to record which instructions the message carries; it is client metadata, not model input. |
| `text.verbosity` | Forwarded to `responses` backends, mapped to `verbosity` for `openai` Chat backends, and reported as `dropped` for `anthropic` Messages backends. The only accepted values are `low`, `medium`, and `high`. |
| `text.format` with `strict: false` | Codex CLI's Approve for me review asks for its answer this way, with a schema that requires only some of its properties. Forwarded with the flag to `openai` and `responses` backends. Messages always enforces an output schema, so an `anthropic` backend receives the schema without the flag, and `x-vsr-warnings` reports `output_format.strict` as `approximated`. The same applies to a Chat `response_format`. |
| Omitted `max_output_tokens` | Codex CLI sends no output limit, so a route to an `anthropic` backend gets `max_tokens: 4096` as described above. A reasoning model can spend all of it before it answers, which Codex CLI reports as `Incomplete response returned, reason: max_output_tokens`. Set a positive integer `request_params.default_max_tokens` on the decisions that send Codex CLI to an `anthropic` backend. |

Responses reasoning replay is projected separately for each selected backend,
including fallbacks and shadows. A model is an OpenAI-hosted Responses target
only when every endpoint in its pool has provider `type: openai` and a resolved
address or profile base URL whose host is exactly `api.openai.com` or
`chatgpt.com`, or ends in `.openai.azure.com`. A shadow checks only its own
endpoint. Hosted targets receive encrypted reasoning items with `content: []`;
unsigned plaintext-only reasoning items are omitted. Signed thinking stays
subject to the signature capability check. Other targets receive the existing
plaintext representation, with the Responses ciphertext and its `include`
omitted. Internal Chat paths, including Looper, context recovery, and memory
history, also drop Responses blobs and the include. Other reasoning continuity
formats still fail closed on paths that cannot carry them. These omissions are
reported through translation diagnostics.
The Router's `previous_response_id` store retains the ciphertext with the item.
A blob's format does not establish its account or model provenance; replay still
requires a compatible provider context.

Each accepted but unforwarded field appears as a `dropped` entry in
`x-vsr-warnings`.

In Responses output, `usage.input_tokens_details.cached_tokens` is always an
integer, as the Responses schema requires: a backend that reports no cache-read
count, such as vLLM's Chat server without `--enable-prompt-tokens-details`,
gives `0`, which is also what vLLM's own Responses server sends.
`cache_write_tokens` is omitted when the backend does not report a cache-write
count. Reported zero and positive counts are preserved. Codex CLI fails a turn
on a `null` count (0.163.0 for `cache_write_tokens`, 0.166.0 for
`cached_tokens`) and on a details object without `cached_tokens`.

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
- **Unsupported:** hosted web search. Disable it in the Codex `config.toml`
  that points at the Router:

```toml
web_search = "disabled"
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
- **The catalog.** On a ChatGPT login (`requires_openai_auth = true`, see
  [Agent Harness](agent-harness#codex-on-its-own-chatgpt-login)), Codex also
  asks for its own model catalog with `GET /v1/models?client_version=…`. The
  Router answers that request with an empty `models` catalog beside the model
  list, so Codex keeps its default model metadata instead of logging that it
  cannot read the list.

```toml
model = "vllm-sr/auto"
model_provider = "vllm-sr"

[model_providers.vllm-sr]
name = "vLLM Semantic Router"
base_url = "http://localhost:8899/v1"
wire_api = "responses"
```

### Namespace tools: multi-agent and MCP

Codex groups tools into Responses `namespace` tools: `multi_agent_v1` holds
`spawn_agent` and the other multi-agent tools when `multi_agent` is on, and each
MCP server's tools arrive in a namespace named `mcp__<server>` by default. The
Router accepts them, so `multi_agent` no longer needs to be disabled. This was
checked with Codex CLI 0.166.0 on Chat, Responses and Messages backends.

- Every backend, a Responses backend included, receives each namespaced tool as
  one tool named `<namespace>__<name>`, for example `mcp__echo__echo` or
  `multi_agent_v1__spawn_agent`. The namespace's description precedes the
  tool's own.
- A flat name keeps to 64 characters of letters, digits, `_` and `-`. A longer
  name, or one with other characters, is shortened and made unique with a hash
  of the namespace and name; a call in a later turn maps to the same name.
- A request in which two different tools would get the same flat name fails
  with `tool_namespace_collision`, whether the tools are declared or called in
  the input history or in history retained behind `previous_response_id`. For
  example, `mcp__a` with `b__c` and `mcp__a__b` with `c` both flatten to
  `mcp__a__b__c`. A namespace inside a namespace, a hosted tool inside a
  namespace, and a named `tool_choice` that selects a namespaced tool by its
  own name are rejected.
- The model's calls reach Codex with `namespace` and the tool's own `name`,
  buffered and streamed. Calls in the input history, including history
  retained behind `previous_response_id`, are sent to the backend under the
  flat name.
- A custom tool inside a namespace follows the custom-tool rules above, so it
  cannot reach a Messages backend.
- Tool filters, conversation signals and Router Replay see the flat name.

A multi-agent child is a separate Codex request. Codex CLI 0.166.0 sends it with
the parent's `session-id`, its own `thread-id`, `x-codex-parent-thread-id`
naming the parent, and `x-openai-subagent: collab_spawn`. Its tools keep the
MCP namespaces but not `multi_agent_v1`.

`spawn_agent` can start a child on another model. Codex checks the model name
against its own model catalog before it sends the child's request, and the
Router answers a name its entrypoints do not publish with `model_not_found`.
Publish the name in an entrypoint's `model_names`:

```yaml
entrypoints:
  - model_names: [vllm-sr/auto, vllm-sr/cheap]
    recipe: default
```

Then give Codex a catalog that lists every name it may use, with
`model_catalog_json` in `config.toml`. Codex also decides the request shape from
the catalog entry, so base each entry on an entry the Router accepts.
`codex debug models --bundled` prints the bundled catalog. In Codex CLI 0.166.0,
copy its `gpt-5.5` entry, set `slug` and `display_name` to the Router model name,
and set `supports_search_tool` to `false`, because Codex otherwise replaces the
namespaces with a `tool_search` tool, which the Router does not support. For a
Messages backend, also set `apply_patch_tool_type` to `null`, because Codex
otherwise sends `apply_patch` as a custom tool. The `gpt-6` entries send tools
in an `additional_tools` input item with `reasoning.context`, which the Router
rejects. `vllm-sr connect codex` writes such a catalog: the bundled entries and
one for the profile's model, with `apply_patch_tool_type` set to `null`
whatever the backends; add an entry to it for each other name.
