# Router API

The router data plane accepts model requests on the configured listeners. The
Router serves them itself in standalone mode, the default; with
`--gateway extproc`, Envoy serves them and calls the Router over ext_proc. In the
standard local stack, the listener is `http://localhost:8899`; a recipe can
choose a different address or port under `listeners`.

Use the data plane for inference. Use the management API, normally bound to
`127.0.0.1:8080`, for health checks, configuration, diagnostics, and replay
queries. See [Router management API](./apiserver).

## Supported inference paths

| Method | Path | Client format | Notes |
| --- | --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions | Main routed inference endpoint |
| `POST` | `/v1/responses` | OpenAI Responses | Requires the Responses service to be enabled |
| `GET` | `/v1/responses/{id}` | OpenAI Responses | Reads a stored response |
| `DELETE` | `/v1/responses/{id}` | OpenAI Responses | Deletes a stored response |
| `GET` | `/v1/responses/{id}/input_items` | OpenAI Responses | Reads stored input items |
| `POST` | `/v1/messages` | Anthropic Messages | The router translates when the selected backend uses another protocol |
| `POST` | `/v1/messages/count_tokens` | Anthropic Messages | Counts with the Messages backend of the named model, or of the model that last served the session, at its `count_tokens` path; otherwise answers with its own estimate, which counts text by tokenizer-style pieces rather than bytes. No turn is routed or recorded. A recipe with candidate requirements, or with a decision that reads content (any signal but conversation shape and context size) or has a plugin, always gets the estimate. Rate limits, token budgets included, admit a count by its input before it is forwarded |
| `POST` | `/v1/images/generations` | OpenAI Images | Routed on the prompt like any text request, and answered only by a decision whose `output` is an image. See [Images API](#images-api) |
| `POST` | `/openai/deployments/{deployment}/chat/completions` | Azure OpenAI Chat Completions | The deployment names the Router model; `api-version` is accepted and not forwarded |
| `POST` | `/openai/responses` | Azure OpenAI Responses | Accepts the dated `api-version`; the model is in the request body and the Responses service must be enabled |
| `POST` | `/openai/v1/responses` | Azure OpenAI Responses | The model is in the request body and the Responses service must be enabled |
| `POST` | `/openai/v1/chat/completions` | Azure OpenAI Chat Completions | The model is in the request body |
| `GET` | `/v1/models` | OpenAI Models | Lists models exposed by the active router configuration; in Engine mode, the listener's System One models, also as `models: [{name, description, release_date}]` for the System One SDKs |
| `POST` | `/v1/systemone`, `/v1/decisions` | Native System One | Direct decision models or explicitly published native recipes on a standalone listener |
| `GET` | `/v1/systemone/models` | Native model discovery | Lists that listener's published System One models |

`GET /v1/responses/{id}/input_items` includes nonempty stored `instructions` as a
system-message item before the stored input items. Item IDs, `first_id`, and
`last_id` remain stable across repeated reads of an unchanged stored response.

Engine mode serves the native System One paths with recipe routing disabled.
A listener serves the native API when it lists `systemone` in `apis`
(`listeners[].apis.systemone`, optionally with `models`); the LLM APIs'
`models` do not grant native access. All APIs use the listener's API keys, and
`apis.systemone.schema` sets the System One schema the listener speaks
(`vllm-sr/5.0` by default, or `vllm-sr/4.0`); see
[System One schemas](../system-one/use/schemas.md).
See the [model runtime quickstart](../model-runtime/get-started/quickstart.md) for a request.

The System One paths read bodies compressed with one layer of gzip or zstd
(`415` with `unsupported_media_type` for any other coding) and
`multipart/form-data` bodies whose image parts are the request's `assets`.
The Router decodes these before it reads the request and forwards the JSON
request, with each part's image as `data_base64`, to the model. On
`/v1/systemone` a request with a failed question is answered as a whole with
`question_failed`, the question's `question_error`, fixed `reason` and ID
(in a batch, also its entry as `state`), while `/v1/decisions` answers each
question in place. A model the listener does
not publish returns `403` `model_not_allowed` with the models it does publish.
See [System One compatibility](../model-runtime/reference/http-api.md#system-one-compatibility)
for assets, Box, track and long context.

### Route a System One request

In Router mode, publish an `entrypoints` item with `api: systemone` and grant
its name in `listeners[].apis.systemone.models`. The Chat default `vllm-sr/auto`
does not automatically publish a native entrypoint. Follow the
[System One escalate guide](../tutorials/algorithm/looper/escalate.md#on-system-one) to connect
local deployments or remote Engine and compatible System One services.

```bash
curl -sS http://localhost:8899/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "state": "Please explain how to reset my password.",
    "questions": {
      "task": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "account": "Account access or authentication",
          "billing": "Payments, invoices or refunds"
        }
      }
    }
  }'
```

Auto currently accepts explicit `choice`, `score` and `noul` questions, including
multiple named states. The selected native response retains its answers,
probabilities, usage and other fields. `model` stays the public entrypoint;
the additional `routing` object identifies the execution. Set
`options.return_meta: true` to include the selected model's runtime metadata.
Calibrated escalations collect this provenance internally even when
you leave response metadata disabled.

```json
{
  "recipe": "native-escalate",
  "decision": "classify",
  "algorithm": "escalate",
  "stage": "strong",
  "selected_model": "vega",
  "quality": "uncalibrated",
  "model_calls": 2
}
```

`model_calls` counts physical inference attempts across signal evaluation and
the selected algorithm, including remote decision-model calls and transport
retries. Signal timeouts remain upper bounds inside the shared request deadline.
It is not a count of GPU forwards. `usage` belongs to the returned native
candidate and is not total escalation billing.
The `quality` field names the configured acceptance method; it is not an
accuracy score. Native discovery marks recipe entrypoints with `routing: true`
and concrete models with `routing: false`.

Malformed or unsupported auto questions return `400` with `invalid_request`,
and a message saying what auto cannot route. A request that no candidate model
can answer returns `400` with `unsupported_capability`: for example, a `box`
question when no model of the decision answers box questions or reads images.
Its message names what is missing and each candidate, with what it lacks and
where that came from (its card, its backend's listing, or what its deployment
declares); see
[Route by what each model answers](../tutorials/algorithm/looper/escalate.md#route-by-what-each-model-answers).
If no complete answer passes the declared acceptance rule, the request returns
`503` with `systemone_unresolved`. Exhausting the shared physical-call limit returns
`503`; exhausting its deadline returns `504`. Both budget errors use
`systemone_budget_exhausted` and include the reason and `model_calls` count.
A model whose auth is `credential: delegated` answers `401`
(`delegated_credential_missing`) when the request carries no delegated
credential, before the call, as a request the Dashboard forwards does, and `403`
(`delegated_router_call_refused`) when no client request is behind the call,
as for an operator's routing diagnostic.
Failed candidates never become empty successful answers. Direct model requests
keep their existing native API contract and do not run the escalation.

A remote Engine backend must expose a concrete model. Native backend calls
carry `X-VSR-SystemOne-Backend: 1`; a receiving Router returns `409`
(`systemone_nested_routing`) if that request targets another recipe or remote
forwarding alias. This prevents recursive cascades. The header grants no access:
the receiving listener still checks its API key and model allowlist. An external
compatible provider's internal execution remains outside the Router's call
ledger.

Auto execution exports `sr_systemone_stage_total` and
`sr_systemone_stage_duration_seconds` by configured algorithm, stage and model,
plus `sr_systemone_auto_requests_total` and
`sr_systemone_auto_duration_seconds`. These distinguish accepted, rejected,
invalid and failed attempts, and stages skipped because their model cannot
answer the request (`outcome="incapable"`, also the request outcome when no
candidate can). They measure serving behavior, not label-based evaluation
quality. The auto duration covers the algorithm; preceding recipe
signals are outside that metric.

Other `/v1/*` paths fail closed. In particular, `/v1/files`,
`/v1/vector_stores`, and Router Replay paths are not available on a public
inference listener. Router-owned file and vector-store operations use
`/api/v1/storage/files` and `/api/v1/storage/vector-stores` on the management
listener. Other `/openai/*` operations, such as embeddings and stored-response
reads, return `404`.

Every `POST` path above requires a JSON request body. A request with an empty
body returns `400` from the Router and is never forwarded to a backend.

See [Protocol Compatibility](../installation/protocol-compatibility) for the
client-to-backend translation matrix, backend `api_format` values, and
field-level portability boundaries.

### System One API stability

The public System One API is what a Router's listener serves:
`POST /v1/systemone`, `POST /v1/decisions` and `GET /v1/systemone/models`,
with `/health` and `/ready`. It is the only System One interface with a
compatibility promise:

- Within a release line it changes only by addition: new optional request
  fields, new response fields and new error codes, each documented with its
  HTTP status.
- A removal or a change of meaning is announced in the release notes at least
  one release before it takes effect.

The Router checks a request's `options` against this API before a model reads
them. The options are `deadline_ms` (a number greater than 0),
`long_context` (`true` or `false`), `max_tokens` (an integer of at least 1) and
`return_meta` (`true` or `false`); `null` sets none. Any other option, or a value of another type, returns `400` with
`invalid_request` and a message that names it. A request the Router sends to
its model directly waits 30 seconds for the answer, or its `deadline_ms` when
that is longer, up to the listener's `apis.systemone.max_deadline_ms` (600000, 10
minutes, by default); a `long_context` request without `deadline_ms` waits
that whole limit. `options.profile` was removed in
[#5071](https://github.com/vllm-project/semantic-router/pull/5071): a request
runs with its deployment's [engine arguments](../model-runtime/deploy/engine-args.md),
and `meta` reports `dtype` and `numerics` instead of `profile`.

The model runtime's HTTP API, which the Router calls, belongs to one release
and makes no such promise. A Router accepts only a runtime that serves exactly
its contract version; see
[Attach to a runtime you run](../model-runtime/deploy/connect-the-router.md#attach-to-a-runtime-you-run).

## Send a routed request

Use `vllm-sr/auto` or an explicitly declared recipe entrypoint when you want the router to select a
backend. Use a concrete model name when you want to bypass semantic model
selection and target that model directly.

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that merges two sorted lists."
      }
    ]
  }'
```

The response keeps the client protocol's shape. Its model, content, token
usage, and optional router headers depend on the selected backend and recipe.
A reply the backend produced names the model as the backend reports it,
whether it streams or not and whatever format the backend speaks. A stream
whose backend names no model in its first event keeps the logical name. A
reply the Router composes itself, from a Looper algorithm or its own refusal,
names the logical model, the name `x-vsr-selected-model` reports. See
[VSR routing headers](../api/headers) for the stable
observability contract.

The model names accepted by the Router come from canonical provider entries.
`name` is the logical alias used by decisions and clients,
`provider_model_id` is sent to the upstream provider, and
`providers.models[].backend_refs[]` identifies the physical endpoint:

```yaml
providers:
  models:
    - name: local-small
      provider_model_id: served-model
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0
        completion_per_1m: 0
      backend_refs:
        - name: local-vllm
          endpoint: model-server:8000
          protocol: http
          provider: vllm
          weight: 1
```

Pricing is operator-supplied deployment metadata, not a live quote. It stays on
`providers.models[]`; `providers.models` only describes semantic capabilities.
`currency` is optional and resolves to `USD` for accounting when omitted. When set,
it must be an uppercase three-letter code. All per-million-token rates must be finite
and non-negative. `cached_input_per_1m` and `cache_write_per_1m` are optional, and an
explicit zero represents a free rate.

`api_format` declares the upstream wire contract: `openai` for Chat
Completions, `responses` for the OpenAI Responses API, or `anthropic` for
Anthropic Messages. The client may use any supported inference path; the Router
translates once at the provider boundary and returns the client's original wire
format.

### vLLM Chat controls

For Chat backends that implement these vLLM extensions, the Router preserves
these fields through routing and request edits:

| Field | Accepted values |
| --- | --- |
| `top_k` | Integer: `-1` or `0` disables filtering; positive values limit candidate tokens. |
| `min_p` | Number from 0 to 1. |
| `repetition_penalty` | Finite number greater than 0. |
| `cache_salt` | String of 1–128 characters, excluding `@`, `/`, `\`, and NUL. |

`cache_salt` selects a backend prefix-cache namespace without changing the
prompt. Reuse a salt for requests that may share cached prefixes. The Router
also preserves `chat_template_kwargs`, such as `enable_thinking`. These
extensions are rejected when the target protocol cannot represent them.

`logit_bias`, which maps token IDs to a bias from -100 to 100, reaches Chat
backends the same way and is rejected for Responses and Messages backends. A
vLLM backend that runs speculative decoding rejects `min_p` and `logit_bias`;
[list them in the model's `unsupported_capabilities`](../installation/protocol-compatibility#sampling-controls-a-backend-rejects)
to send that model requests without them.

### Responses API

```bash
curl -sS http://localhost:8899/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "input": "Summarize the trade-offs of retrieval-augmented generation."
  }'
```

Creating, retrieving, and deleting Responses API objects requires its backing
service and store. When Responses API support is disabled, the collection
endpoint returns `404` and stored-object handling is unavailable. A configured
service retains objects according to its own storage and retention settings.

### Anthropic Messages

```bash
curl -sS http://localhost:8899/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "vllm-sr/auto",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": "Explain semantic routing in one paragraph."
      }
    ]
  }'
```

### Images API

```bash
curl -sS http://localhost:8899/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "prompt": "Draw a lighthouse at dusk.",
    "size": "1024x1024"
  }'
```

The Router routes an Images API request on its prompt, like any text request.
The request must reach a decision whose
[output contract](../tutorials/decision/output) is an image, such as
`output: {kind: image, action: generate}`. That decision's model generates the
image, whether it answers on the Images API (`api_format: images`) or on the
Responses API with the hosted `image_generation` operation, and the client
receives the image in the Images API's own shape:

```json
{"created": 1700000000, "data": [{"b64_json": "iVBORw0KGgo..."}]}
```

The request takes `model`, `prompt`, `n`, `size`, `quality`, `background`,
`moderation`, `output_format` and `output_compression`. `response_format` must
be `b64_json`, and the API does not stream. A request that reaches no image
decision (it matches none, matches one that answers in text, or names a model
directly instead of an entrypoint such as `vllm-sr/auto`) gets `400`
`image_route_required` before any model is called. A Chat Completions or
Anthropic Messages request that reaches an image decision gets `400`
`unsupported_capability`, since neither API can return an image. One Responses
API call generates one image, so `n` above 1 needs a decision with a model on
the Images API; a decision whose models are all on the Responses API answers it
with `400` `unsupported_capability` before any model is called, unless its
[`request_params`](../tutorials/plugin/request-params) `max_n` caps `n` to 1
or its `blocked_params` removes `n`.

### Azure OpenAI clients

Clients built for Azure OpenAI can call the Router as if it were an Azure
resource. The deployment Chat path takes the model name from the URL; the
Responses and v1 Chat paths take it from the request body. In either form,
A recipe entrypoint selects a route and a concrete model name targets that model directly.
The Router checks the client's `api-key` header against the listener's
`api_keys` when they are set, and removes the header before provider dispatch.

```bash
curl -sS 'http://localhost:8899/openai/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'api-key: YOUR-LISTENER-KEY' \
  -d '{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Explain semantic routing in one paragraph."}]}'
```

For GitHub Copilot CLI, set `COPILOT_PROVIDER_TYPE=azure`, point
`COPILOT_PROVIDER_BASE_URL` at the listener, and set
`COPILOT_PROVIDER_WIRE_MODEL` to the Router model name. With
`COPILOT_PROVIDER_WIRE_API=responses`, the CLI uses `/openai/v1/responses`, or
`/openai/responses` when `COPILOT_PROVIDER_AZURE_API_VERSION` is set. The
supported Chat paths are `/openai/v1/chat/completions` and the deployment
route. The Router accepts Copilot's `reasoning.summary` on Responses
turns. It forwards the setting to a Responses backend. With a Chat Completions
or Messages backend, the turn still runs, but the summary request is dropped
and reported in `x-vsr-warnings`.

Protocol translation is limited to fields the router supports. When a request
crosses protocols, inspect the `client_protocol` and `upstream_protocol`
members of `x-vsr-trace` and any `x-vsr-warnings` response header.

Chat Completions accepts OpenRouter's `reasoning_details` on assistant messages
and streaming deltas. Text, summary, encrypted, and server tool call blocks keep
their order, IDs, format, index, and signatures when responses are returned or
replayed in a follow-up Chat request. Send the original blocks back with tool
results; do not summarize or reorder them. A buffered reply carries the plain
reasoning text as `reasoning_content`. A same-format Chat stream passes the
provider's deltas through and keeps its field name, which for OpenRouter is
usually `reasoning`; a Chat stream that the Router renders itself, such as a
cache replay, uses `reasoning_content`. In a follow-up request the blocks take
precedence, and the Router does not forward the `reasoning` or
`reasoning_content` text a second time.
Stream fragments accumulate in order when they share a type and an index or ID.
A fragment of another type under the same index starts a new block, as Gemini
models do for the encrypted signature of a tool call. Blocks without either
identifier remain separate; the Router does not infer their identity.

Responses separately carries its own encrypted reasoning state as
`encrypted_content` on reasoning items, including items with no text. Send
`include: ["reasoning.encrypted_content"]` to request it. Buffered replies and
streams preserve the blob verbatim; streams reuse the item-done ciphertext in
the terminal output. The Router's Responses store retains it for
`previous_response_id` replay.

Before dispatch, the Router projects Responses reasoning for each backend,
fallback, and shadow. A model qualifies as an OpenAI-hosted Responses target
only when every endpoint in its pool has provider `type: openai` and a resolved
address or profile base URL host exactly `api.openai.com` or `chatgpt.com`, or
ending in `.openai.azure.com`. A shadow checks its single endpoint. Hosted
targets receive encrypted items with `content: []` and the include. Unsigned
plaintext-only reasoning items are omitted for those targets; signed thinking
remains subject to the signature capability check. Other targets keep the
existing plaintext behavior and omit the Responses blob and include, with
translation diagnostics. Internal Chat paths for Looper, context recovery,
and memory history also drop Responses blobs and the include. Other continuity
formats still fail closed there. Chat and Messages clients omit Responses blobs
with the reasoning continuity diagnostic. A blob's format does not prove that
another account or model can use it.

Only Chat Completions carries the OpenRouter blocks described above. A Responses
or Messages client of a Chat backend still gets the reply: the Router drops the blocks and keeps the
reasoning text. A buffered reply reports the drop in `x-vsr-warnings`
as `dropped;...;reasoning_details`. A stream has already sent its headers when
the drop is found, so the drop is recorded only in the
`llm_translation_lossy_total` metric and a `translation_lossy` debug log event;
the stream carries no warning.

A request whose history contains the blocks cannot be sent to a Responses or
Messages backend and fails with `unsupported_capability`. Candidate selection
does not avoid those backends unless `candidate_requirements.capabilities` is
`declared`. Looper decisions and context compression recovery build their
internal calls in a form that cannot carry the blocks, so the Router refuses
those calls rather than send them without the blocks. A Looper decision returns
400. A refused recovery follows the plugin's `failure_mode`: `fail_closed`
returns 502, and `fail_open`, the default, returns the original reply without
its retrieval calls.

## Routing errors

When the Router cannot route a request, it answers the request itself and calls
no backend. The error uses the client's protocol. In OpenAI Chat Completions and
Responses errors, `error.code` is a stable reason code and `error.message` a
short message. Apart from the budget and context-length errors, and the model
errors, which name the model the client sent, the message names no model,
decision, or request content:

```json
{"error":{"type":"invalid_request_error","code":"no_route","message":"no route matched the request","param":null}}
```

| Code | Status | `error.type` | Meaning |
| --- | --- | --- | --- |
| `model_not_found` | 404 | `invalid_request_error` | The request names a model this Router does not serve on any listener: no entrypoint name, provider model or LoRA adapter. The message is OpenAI's, ``The model `x` does not exist or you do not have access to it.``, which OpenAI SDKs read as an unknown model. Anthropic Messages clients get `not_found_error`. |
| `model_not_allowed` | 403 | `permission_error` | The listener's [model allow-list](../installation/gateway-modes#model-allow-list) leaves out a model the Router serves on other listeners. The message names the model. |
| `no_route` | 400 | `invalid_request_error` | No decision matched, and no default model applies. A recipe entrypoint falls back to `providers.defaults.model` when it is configured. Looper entrypoints follow the same recipe rules; their names do not select an algorithm. |
| `context_length_exceeded` | 400 | `invalid_request_error` | The Router's count puts the request beyond the band around the `context_window_size` of every model that could serve it (3% with a model's own tokenizer, 30% with the proxy tokenizer): beside the signals, for every model the request can reach, which cancels the signals still running, or after the decision, for its candidates. `param` is `messages`, and the message names the widest window and the request's estimated count, as OpenAI's does: `This model's maximum context length is 1048576 tokens. However, your request has an estimated 1100000 input tokens. Please reduce the length of the input.` Anthropic Messages clients get `prompt is too long: 1100000 tokens > 1048576 maximum`. See [the context window check](../installation/model-configuration#check-prompts-against-the-context-window). The [request budget check](#request-budget-errors) answers the same code. |
| `max_output_tokens_exceeded` | 400 | `invalid_request_error` | The requested output exceeds the configured model limit. See [request budget errors](#request-budget-errors). |
| `overloaded` | 429 | `rate_limit_error` | The request's guard could not be read because its decision model was overloaded, and `global.model_catalog.on_signal_overload` is `retry`. Either only rules that failed closed on it refuse the request, or a rule it could not read would refuse it, even one that fails open. Anthropic Messages clients get `529` with `overloaded_error`. The response carries `Retry-After` and `x-vsr-warnings: guard_overloaded`. See [when the decision model is overloaded](../system-one/deploy/performance#when-the-decision-model-is-overloaded). |
| `decision_unresolved` | 503 | `server_error` | A decision could not be evaluated because a signal it needs was unavailable, and its `rules.on_unknown` is `fail_request`. The `on_unknown` member of `x-vsr-trace` names the decision. |
| `no_eligible_model` | 503 | `server_error` | The selection policy rejected every candidate model of the matched decision. |
| `image_route_required` | 400 | `invalid_request_error` | An [Images API](#images-api) request reached no decision whose output is an image. |
| `image_generation_forbidden` | 400 | `invalid_request_error` | The matched decision answers with a generated image, and the request's `tool_choice` is `none`. |
| `tool_selection_failed` | 503 | `server_error` | The matched decision's [tool selection](../tutorials/plugin/tools.md#selection) failed or timed out, and its `select.on_error` is `fail`. |
| `budget_exceeded` | 429 | `rate_limit_error` | A [spend budget](../tutorials/global/api-and-observability.md#spend-budgets) does not cover the call's estimated cost, or has nothing left. The message names the budget, its limit, what it has left and the estimate, as `x-vsr-budget-scope` and `x-vsr-budget-remaining` do. A user budget's refusal carries `Retry-After` until its window starts again. |
| `budget_unpriced` | 403 | `permission_error` | The selected model has no price in the spend budgets' currency, and `budgets.unpriced` is `refuse`. |

The Router logs each of these failures at `WARN`, under the request's
`x-request-id`, with the code and its own reason: the model the request named,
the recipe and decision it reached, and the error. Anthropic Messages clients
get the same status and message in Anthropic's error envelope, which has no
code field.

When routing decision processing exceeds its deadline, the Router returns
`504` with `request_timeout` and the message `request timed out`. A canceled
decision returns `500` with `request_canceled` and the message `request canceled`
if the connection still permits a response. Both use `server_error` in OpenAI
Chat Completions and Responses. Anthropic Messages uses `timeout_error` for the
deadline and `api_error` for cancellation. A disconnected client may receive no
response.

Router-generated immediate responses preserve HTTP status codes supported by
Envoy. Unsupported or invalid status codes fall back to `500`; an error body
is never intentionally sent with a success status.

## Request budget errors

With `candidate_requirements.context: known_limits`, the Router checks estimated
input plus the effective output allowance against the candidates' configured
limits. When the full estimate leaves fewer candidates than the decision needs,
the Router checks again without the estimate's reserves for structured data and
media and keeps the candidates that count fits. If all candidates still fail
only the budget check, it returns HTTP 400:

| Error code | Meaning |
| --- | --- |
| `context_length_exceeded` | The prepared input and requested output do not fit. |
| `max_output_tokens_exceeded` | The requested output exceeds the configured model limit. |

Missing capabilities, unknown limits, unavailable selection evidence, and mixed
failures retain their selection-error behavior, `no_eligible_model`. Budget checks do not
truncate requests by themselves; opt into
[context compression](../tutorials/plugin/context.md#compression-and-budget) when appropriate.

These counts are estimates. A backend can still reject a request. Its HTTP
status is kept, and a `4xx` error reaches the client in the client's error
format and in the Router's terms:

- **Beyond the window.** A refusal of a prompt or an output limit beyond the
  backend's window gets `context_length_exceeded`: vLLM's `param:
  input_tokens`, its `'max_tokens' or 'max_completion_tokens' is too large`
  and `max_completion_tokens=... cannot be greater than max_model_len=...`
  errors, a message about the context length, window or size, or Anthropic's
  `prompt is too long`.
- **Other refusals.** Any other `400` gets `invalid_request`. Another `4xx`
  keeps the backend's code, except vLLM's integer status, which becomes the
  Router's code for that status, such as `not_found`. A `5xx` is passed on as
  it is.
- **The client's names.** `param` and the message name the member the client
  sent: `max_tokens`, `max_completion_tokens` or `max_output_tokens` for the
  output limit, although the Router sends a Chat backend
  `max_completion_tokens`, and `messages` or `input` for the prompt. The
  message keeps the backend's numbers and drops its own setting names, such as
  `max_model_len`, and vLLM's `(parameter=..., value=...)` suffix.

A Chat request with `max_tokens: 5000000` to a model with a 262,144-token
window gets:

```json
{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens, and your request has 12 input tokens. Please request fewer output tokens.","param":"max_tokens"}}
```

A streaming request rejected before generation receives the same non-2xx JSON
error, not a successful SSE stream. When Replay is enabled, it records the
failed status and body; Router budget rejections use
`terminal_reason: request_budget_exceeded`.

## Router Replay

Router Replay records routing decisions and selected request lifecycle data.
It is useful for debugging, evaluation, and Router Learning, but it does not
change routing merely because a record is read.

Without a `router_replay` plugin, Replay records each request's routing data,
and no request or response text, in the `file` store, which
[keeps records across restarts](../tutorials/plugin/router-replay#keep-records-across-restarts). A `router_replay`
plugin in `global.plugins` replaces that default, and `enabled: false` turns
Replay off:

```yaml
global:
  plugins:
    - type: router_replay
      configuration:
        enabled: false
```

The `memory` backend is suitable for local inspection. Use a configured
persistent backend when records that keep text must survive process restarts,
and set retention appropriate to the data being captured.

Replay queries go to the management API:

```bash
curl -sS 'http://localhost:8080/api/v1/observability/replays?limit=20' \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/observability/replays` | List and filter records |
| `GET` | `/api/v1/observability/replays/{id}` | Read one record |
| `GET` | `/api/v1/observability/replays/aggregate` | Aggregate routing and cost metadata |
| `GET` | `/api/v1/observability/replays/trajectory?session_id=...&recipe=...` | Reconstruct one recipe's session trajectory |
| `GET` | `/api/v1/observability/replays/runs` | List agent runs, one summary per session |
| `GET` | `/api/v1/observability/replays/run?session_id=...` | Read one agent run as a tree of its agents and calls |
| `GET` | `/api/v1/observability/replays/models` | List the configured models with the label the Dashboard's **Requests** page shows for each, no two alike |

The runs list takes the list's filters, `limit` and `offset`; a run reads every
recipe of its session. See
[Agent runs](../tutorials/plugin/router-replay#agent-runs) for both responses
and [Model labels](../tutorials/plugin/router-replay#model-labels) for the
labels.

List and aggregate requests accept filters such as `recipe`, `decision`,
`model`, `session_id`, `turn_id`, `cache_status`, and `search`, and, for agent
approval reviews, `request_kind=codex_approval_review`, `review_status`,
`review_outcome`, and `review_risk`. `turn_id` matches a record's `turn_id` or
`parent_turn_id`: the calls one Codex turn sent, and the approval reviews and
subagent calls it started. `since` keeps the records at or after an RFC 3339
timestamp, such as `2026-11-15T15:00:00Z` (send an offset's `+` as `%2B`), so
the list and the aggregate cover one window; [`vllm-sr usage`](../tutorials/plugin/router-replay#usage-in-the-terminal)
reads both that way. Pagination uses `limit`
and `offset`; `limit` is capped at 100. `showDetails=true` requests large body
fields, so use it only when those fields are needed. See
[Agent approval reviews](../tutorials/plugin/router-replay#agent-approval-reviews)
for the review record and the aggregate's `reviews` summary.

`harness` keeps the records sent by the harnesses it lists, comma-separated:
`claude_code`, `codex`, `opencode`, `deepseek_harness`, `cursor`, `openclaw`, and `api_client` for
the requests no harness sent, such as a script calling the API directly; a
record's harness is its `agent.harness`, described in
[Agent runs](../tutorials/plugin/router-replay#agent-runs). The list response
carries `harness_counts`, the records every other filter selects counted by
harness, such as `{"api_client": 51234, "claude_code": 1820, "codex": 977}`,
so a page of one harness's records still shows how many each harness sent.

With `since`, the aggregate also takes `interval`, a duration of at least `1m`
such as `5m` or `1h` that splits the window into at most 1,000 intervals. The
response then carries `traffic`: `since`, `interval_seconds`,
`latency_p50_ms` and `latency_p95_ms` over the window's completed requests
(estimated from 1,024 samples in a busier window), and `buckets` from `since`
to now, oldest first and empty intervals included. Each bucket
has its `start`, `requests`, `cached`, and `failed` (failed or aborted)
counts, `latency_p50_ms` and `latency_p95_ms` over its completed requests
(absent without one, and estimated from 256 samples in a busier interval),
and `cost`, its spend per currency in the shape of `by_currency`. Without
`interval` the response is unchanged.

The aggregate and the runs list read the whole retained history. A Router
answers each filter combination from one scan, shared by the requests that
arrive while it runs, and keeps the answer for 30 seconds, so a record written
in that time counts from the next scan. A read stops when its client
disconnects, and after 25 seconds at the most. A PostgreSQL store applies the
filters itself, so a filtered aggregate, such as one `since` window, reads only
the records it counts.

Trajectory queries use the exact recipe name. Omitting `recipe` is supported
only when the session's records belong to one recipe; an ambiguous session
returns `400`. An explicit empty `recipe=` selects older, unscoped records.
The response includes each request's route, latency, and lifecycle, including
multiple requests at the same turn index.

Records, trajectory routes, and messages include `conversation_id` when an
explicit conversation identity is available. Messages are grouped by conversation
and turn, so separate conversations in one session can each start at turn zero.
The Dashboard's **Requests** page shows their boundaries and complete IDs. Records and trajectory routes
also carry a Codex request's `turn_id` and `parent_turn_id`; see
[Agent turns](../tutorials/plugin/router-replay#agent-turns).

The Dashboard's **Requests** page shows these routes alongside recorded signals, projections,
candidate scores, and session-switch reasons. In `observe` mode, candidate and
hold explanations describe what protection would have done; the selected model
and route history still describe actual dispatch. Protection's `candidate_models`
lists eligible models independently of their scores; an unrecorded score appears
as `—`, while a recorded zero remains zero. Missing identity or evidence
is displayed explicitly. Replay capture uses `global.services.router_replay`
defaults and the selected decision's `router_replay` plugin overrides. Rejected
requests without a selected decision use the global defaults layered with their
recipe's own plugins.
`prompt_capture` decides what a record keeps besides routing evidence:
`none`, the default, keeps no request or response text; `raw` keeps the text
the capture switches allow; `verified` keeps the prompt alone when the recipe's
PII rules find it clean, and leaves it out when personal data is detected or
PII evidence is unavailable. `capture_personal_data` is its deprecated form:
`false` loads as `verified` and `true` as `raw`.

Client tool results are stored only with `capture_tool_results: true` under
`prompt_capture: raw`. A `client_tool_result` step then carries
the result text in `output` with `content_redacted: false`, cut to
`max_tool_trace_bytes` and flagged `truncated` when cut, and a trajectory's
tool message carries it as `content` with `content_truncated`. Otherwise the
step keeps only its call ID and `status`. Without `replay.detail`, outputs,
request bodies and trajectory content are returned redacted either way.

`capture_request_body: false` leaves request text out of the whole record: the
body, `prompt`, `tool_definitions`, and the text, arguments and results of each
request step in `tool_trace`. `capture_response_body: false` does the same for
the response body, the response steps, `hallucination_spans`, and the provider
replies in a fallback outcome's `execution_record`, where each failed attempt
keeps its status code and a fixed reason. Those steps keep their type, tool
name, call ID and status. The left-out text is absent, not marked
`content_redacted`, which keeps marking content hidden from the reader's role
and tool results that `capture_tool_results` keeps out. `max_tool_trace_bytes`
cuts step text too, and a trajectory's user and assistant messages then carry
`content_truncated`. Where no layer sets it, each field keeps 65,536 bytes
(64 KiB), the steps' `raw_arguments` and `raw_output` included, which then
carry `raw_truncated: true`, and `max_tool_trace_total_bytes`, 262,144 bytes
(256 KiB) then, bounds the steps together: the oldest are dropped and
`tool_trace` carries `steps_truncated: true` with `dropped_step_count`, as for
`max_tool_trace_steps`.

`prompt` is cut on a UTF-8 character boundary to its limit: `max_prompt_bytes`,
or a `max_tool_trace_bytes` a layer sets where none sets `max_prompt_bytes`, or
65,536 bytes (64 KiB) when neither is set; `max_prompt_bytes: 0` keeps it whole. A cut
record carries `prompt_truncated: true` with the whole prompt's size in
`prompt_original_bytes` and its SHA-256 in `prompt_original_sha256`. The hash
is redacted with the prompt for callers without `replay.detail`.

Every Replay read route applies the same redaction. Without `replay.detail`,
trajectory `routes` return `selection_reasoning` and `session_reason` empty,
as the record's `route_diagnostics` do.

### Configured-rate cost estimates

The Dashboard's **Requests** page uses recorded token usage and configured input, cached-input,
cache-write, and output rates. These are estimates, not invoices or GPU running
costs; infrastructure charges and billing adjustments are excluded.

| Field | Meaning |
| --- | --- |
| `actual_cost` | Estimated cost of the selected model for the recorded usage. |
| `baseline_cost` | The baseline model's estimate for that same usage: `cost_baseline_model` when the request's `router_replay` plugin sets one, else the highest same-currency estimate in the selected recipe's model pool. |
| `baseline_model` | The model used for that comparison. Equal costs use model-name order. |
| `cost_savings` | The difference between the baseline and actual estimate. It is negative when the baseline costs less. |
| `currency` | Currency of the recorded estimate; no exchange-rate conversion is applied. |
| `cost_accounting` | Where the figures came from and what they leave out (below). |

`cost_accounting` labels each new record:

| Field | Meaning |
| --- | --- |
| `pricing_source` | `configured` for the model's `providers.models[].pricing`, or `catalog_list_price` when none is set and the built-in catalog lists a price for the model's provider. A list price is an estimate. |
| `caller_login` | The model's backends take the client's own login (`auth.credential: caller`). Its cost is what the tokens would cost at API prices, not what the client's plan charges. |
| `unpriced_reason` | Why `actual_cost` is missing: `no_usage`, `pricing_not_configured`, or `counted_in_parent` for a Looper hop's own record. |
| `baseline_source` | `configured_model`, `recipe_most_expensive` (the default), or `selected_model` for a direct request. |
| `baseline_unavailable` | Why a priced record has no baseline: `pricing_not_configured` or `currency_mismatch`. A configured baseline that cannot be used never falls back to the default. |
| `router_calls` | Model calls the Router made itself to answer the request, entries by task `kind` and `model`. |
| `budget` | The tightest [spend budget](../tutorials/global/api-and-observability.md#spend-budgets) of the request: its `scope` (`run` or `user/<name>`), `currency`, `limit`, what it had `remaining` after the request, the `estimate` the call reserved, what the request was `charged` with its Router calls, and the `settlement`: `usage`, `estimate` for a reply without usage that can be priced, or `released` when no provider answered. A refused request is `charged` only what its Router calls cost, and has no `charged` when they cost nothing. `unpriced_calls` counts the calls with no price, which spent nothing. |

A Looper request's cost is its calls, so its record has no `actual_cost` and one
`router_calls` entry of kind `looper`: `calls`, the request's `prompt_tokens`
and `completion_tokens`, `cost` and `currency` for the calls that have a price,
and `unpriced_calls` with `unpriced_reasons` for the others. A call is unpriced
when its model has no price, it reported no usage or incomplete usage, it is in
another currency than the first priced call, or no attempt trace kept it
(`attempt_not_traced`: a call past the trace's bound of 100 calls or 64 KiB).
Every Looper algorithm traces its calls, so each one is priced at its own
model's rates. An unpriced call is never counted as free.

Request-scoped runtime calls also appear here: `decision_model` for native
questions and selections, `systemone` for remote System One exchanges, and
`embeddings`, `classify` or `rerank` for other model tasks. These entries include
`model`, input (`prompt_tokens`) and output (`completion_tokens`) usage. Each
dispatched task counts once, including fused tasks; result-cache hits, startup,
warmup and background work do not add request spend. A call no deployment
answered, whose connection failed, was reset or timed out before any reply,
charged nothing and is left out, as a request no provider answered is left
out of the priced and unpriced counts (`unanswered_records`). Usage returned with a
failed decision is retained. Aggregate router call spend and coverage include
these calls even when the request fails. Aggregate savings are net of the
calls made for the completed requests they compare (below); a record's own
`cost_savings` stays the model comparison alone.

Deployment pricing is `global.model_catalog.deployments.<name>.pricing`, with
the same per-million-token rates as provider model pricing. An explicit block
wins (an empty block means zero rates); otherwise a deployment's `model` may
name a priced provider model, and a `model_runtime` deployment with neither
costs nothing. Currency defaults to USD. Calls without usage or rates carry
`no_usage` or `pricing_not_configured`, respectively. Local compute costs are
not measured by these token-rate estimates.

The default baseline covers the recipe's models across all its decisions: model
references, explicit candidate-iteration models, and route destinations. A
permitted default-model fallback is included only for decisions that can use it;
auxiliary planners and judges do not enlarge the pool. Other recipes, unpriced
models, and different currencies are excluded. Direct requests without a recipe
compare against themselves. Alternative-model eligibility and tokenization are
not re-evaluated; this is a rate comparison, not another inference.

Missing usage, price, currency, or baseline stays unknown, with the reason shown
on the Dashboard's **Requests** page. Explicitly configured free rates remain zero. Cache hits have zero
additional model-inference cost; cache storage and lookup costs are excluded.
Existing records keep their captured prices and baseline. **Price not recorded**
means the historical record has no price; configuring rates today does not
backfill it.

### Provider prompt-cache breakdown

New Replay records expose `provider_cache` in both the list and detail APIs.
The Dashboard's **Requests** page shows it under **Provider Prompt Cache**, separately from
Router response-cache hits and model-routing savings.

Each of `uncached_input`, `cache_read`, `cache_write`, and `output` contains
`tokens`, `provenance`, and an optional `cost` in the record's currency.
`cache_write_5m` and `cache_write_1h` retain available retention-tier evidence;
their costs are already included in `cache_write.cost` and must not be added
again. The four main component costs sum to `actual_cost` for complete usage.

`state` is `complete`, `partial`, `unavailable`, or `invalid`. Provenance preserves
`authoritative`, `derived`, `estimated`, and `unknown` evidence. Missing tokens
and costs stay absent; an explicitly reported or priced zero remains zero.
Estimated usage is not used to calculate cache savings. Missing prices do not
erase token evidence, and historical records are not repriced.

With complete usage and configured prices:

- `no_cache_cost` prices all input at the **selected model's standard input
  rate**, plus the recorded output at that model's output rate.
- `cache_savings = no_cache_cost - actual_cost`. It can be negative when
  cache-write premiums exceed cache-read savings.
- Existing `baseline_cost` and `cost_savings` continue comparing model choices
  using observed cache usage. They are distinct counterfactuals, not additional
  provider cache savings.

For example, 10 uncached input, 10 cache-read, 30 five-minute writes, 50 one-hour
writes, and 2 output tokens at rates of 2, 1, 4, 8, and 10 per million cost
`0.00057`. The same-model no-cache cost is `0.00022`, so provider cache savings
are `-0.00035`. The input total remains 100; the tier counts partition the
80 cache-write tokens.

Partial, unavailable, or invalid usage does not produce a no-cache estimate or
cache savings. A Router response-cache hit makes no new provider request, so
its record has no `provider_cache`; the **Requests** page identifies it separately instead
of showing replayed provider usage as new provider savings.

Aggregates use `summary.by_currency`, a sorted array with `currency`,
`total_saved`, `gross_saved`, `netted_router_call_spend`, `baseline_spend`,
`actual_spend`, `router_call_spend`, `total_spend`, and `cost_record_count` for
each currency. With one currency, the flat summary fields mirror that group.
With multiple currencies, flat `currency` is omitted and flat amounts are zero
placeholders: use `by_currency`, not those placeholders. There is no combined
cross-currency total.

`gross_saved`, `baseline_spend`, and `actual_spend` cover the
`cost_record_count` completed records compared with a baseline. `gross_saved`
is `baseline_spend - actual_spend`, the saving of the model choice alone.
`total_saved`, the saving the Dashboard shows, is net of the calls the Router made
itself for those same records: `total_saved = gross_saved -
netted_router_call_spend`. For example, a request whose selected model costs
`0.00010` against a `0.00100` baseline saves `0.00090` gross; a System One call
priced at `0.00012` for it leaves `0.00078` net.

Every `router_calls` kind of a compared record is netted, and
`coverage.netted_router_calls` counts them by kind. A Looper request is never
compared with a baseline, since it has no selected-model cost: its calls are
its answer and stay in `router_call_spend`. Codex approval reviews and memory
requests sent to a `review_model` or `memory_model` entrypoint are not Router
calls: the client sends them, and each is a record compared with a baseline
like any other request. A record with a Router call that has no price, or a
price in another currency, is left out of savings, gross and net, and counted
in `coverage.net_unavailable`: an unknown cost is never netted as zero. A call
priced at zero rates is netted at zero.

`total_spend` adds every priced cost, with or without a baseline, including
`router_call_spend`, the priced calls the Router made itself. A request that
failed or has not finished still counts the calls it paid for, such as the
attempts of an escalation that then failed.
`excluded_record_count` includes non-completed requests and records missing the
data needed for a savings comparison; it does not imply that every excluded
request lacks pricing.

`summary.coverage` says how much the amounts leave out:

| Field | Meaning |
| --- | --- |
| `priced_records` | Records whose every cost has a price. |
| `partly_priced_records` | Records with some costs priced, such as a Looper request with an unpriced call. |
| `unpriced_records` | Records with no price at all. |
| `unpriced_reasons` | The partly priced and unpriced records by the reason of their first missing price, for example `pricing_not_configured`, `no_usage`, `not_completed` for a request that failed after a provider answered or has not finished, which is never fully priced unless the Router refused the answer of a priced call, or `price_not_recorded` for a record written before the Router recorded why. |
| `looper_hop_records` | Records of Looper hops, left out of the three counts because the record of the request that made each hop counts its call. |
| `refused_records` | Records of requests the Router refused, or that ended with a `4xx`, before any model answered them, such as a block, also one a stream carries as an error event under its `200`, or a spend budget's refusal, left out of the three counts because they cost nothing. |
| `unanswered_records` | Records of requests no provider answered: the upstream connection failed, was reset or timed out before any response (`terminal_reason` `upstream_reset`, `upstream_connect_failure`, `upstream_timeout` and the like, with no usage), left out of the three counts because nothing was charged. A provider's own error reply (`upstream_error_response`), and a stream cut after its `200` began, got a response and stay unpriced. |
| `list_price_records` | Priced records whose rate is a catalog list price. |
| `caller_login_records` | Records served on a client's own login, priced or not. |
| `router_calls`, `unpriced_router_calls` | Calls the Router made itself, and how many have no price. A call no deployment answered is none. |
| `baseline_sources` | The records compared with a baseline, by `baseline_source` (`unrecorded` for older records). |
| `baseline_unavailable` | Priced completed records without a baseline, by why. |
| `netted_router_calls` | The Router calls netted out of savings, by `kind`. |
| `net_unavailable` | Records with a baseline left out of savings because a Router call they made has no price in their currency, by the reason of the first one, such as `pricing_not_configured`, `no_usage` or `currency_mismatch`. |

### Net provider prompt-cache saving

The aggregate's `provider_cache` totals the same-model no-cache comparison over
the records that match the request's filters:

| Field | Meaning |
| --- | --- |
| `no_cache_spend` | Sum of `provider_cache.no_cache_cost`. |
| `actual_spend` | Sum of the same records' actual provider cost. |
| `net_savings` | `no_cache_spend - actual_spend`. |
| `priced_records` | Records included in the totals. |
| `excluded_records` | Matching records left out of the totals. |

Only settled records are included: completed, not served from the Router
response cache, with `complete` usage and a captured no-cache cost. The total
is not a sum of per-request savings clamped at zero. A request whose cache
writes cost more than its cache reads saved lowers `net_savings`, and the
total can be negative. For example, one request that saves `0.00009` and
another that loses `0.00035` give a net saving of `-0.00026`, not `0.00009`.
The costs are the prices captured when each record settled; records are not
repriced with today's rates.

`by_currency`, `currency`, and the flat amounts follow the same rules as
`summary`. `by_decision` and `by_selected_model` split the totals per
decision or model and currency, without a top-N cut, so each currency's
entries add up to its total.

When bearer authentication is enabled, replay callers need `replay.read`.
Prompt, response, tool, and other sensitive details remain redacted unless the
principal also has `replay.detail`. Treat replay storage as potentially
sensitive even when the API normally returns a redacted view.

Replay lifecycle values describe what the recorder observed:

- `in_progress`: no terminal response frame has been recorded yet.
- `completed`: the response finished normally.
- `failed`: the Router refused the request, the Router failed, or the
  upstream response failed. The record keeps the status the client received,
  and `terminal_reason` says why, as in the table below.
- `aborted`: the stream ended without a valid terminal frame, for example
  after a disconnect or timeout.

An HTTP `200` response header alone does not make a streaming record
`completed`. A stream that reached its terminal frame does, even when the
record cannot keep its answer, such as one that was all thinking the client
asked to omit.

| `terminal_reason` of a `failed` record | When |
| --- | --- |
| `request_rejected` | The Router refused the request after routing it, with a `4xx`: before it sent the request to a backend, such as for a capability the selected model's wire format lacks or a `fast_response` that sets `block: true`, or in place of the answer, such as a `response_guard` block. A held stream's refusal arrives as its error event and keeps the `200`. A guard overload is a refusal its client retries, whether as `429` or as `529`, and so is a fallback candidate that needs a delegated credential the request does not carry. |
| `request_budget_exceeded` | The request does not fit the [request budget](#request-budget-errors) of any candidate. |
| `budget_exceeded`, `budget_unpriced` | A [spend budget](../tutorials/global/api-and-observability.md#spend-budgets) refused the call before it was sent. |
| `request_processing_failed` | The Router failed after routing. It answered with its own `5xx`, such as for a failed RAG retrieval, context compression that fails closed, a cached answer it cannot serve, a fast response it cannot encode, or a failed context recovery follow-up, or the request failed with an internal error. In standalone mode the record of such an error keeps the gateway's `500`. Behind Envoy, the ext_proc filter's `failure_mode_allow` decides what the client gets, so the record has no status. |
| `upstream_invalid_response` | The Router cannot read or pass on the upstream answer, such as a cut body or one with no output, and answers `502` in place of the upstream's `200`. A held stream it read to its end but could not rebuild an answer from ends this way too, under its `200`. |
| `upstream_error_response` | The upstream answered with an error status, or its stream carried an error event, which keeps the `200` the stream began with. A fallback chain that ends without a success keeps the answer of its last attempt, in both modes. |
| `upstream_<failure>` | No backend answered: in standalone mode, the gateway's `503` or `504` local reply, such as `upstream_connect_failure` or `upstream_reset`, and, in both modes, a fallback chain whose last attempt did not answer. |

## Which port should I use?

| Task | Surface |
| --- | --- |
| Send model traffic | Standalone frontend, or Envoy with `--gateway extproc`; `8899` in the standard local stack |
| List public models | `GET /v1/models` on the inference listener |
| Check health or readiness | Management API on `8080` |
| Read or change configuration | Management API on `8080` |
| Inspect replay records | Management API on `8080` |
| Manage Router-owned files or vector stores | Management API on `8080` under `/api/v1/storage/*` |

Do not expose the management port as a substitute for the public inference
listener. Its endpoints can reveal configuration and operational data or make
state-changing requests.
