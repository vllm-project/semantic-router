# Router API

The router data plane accepts model requests through an Envoy listener. In the
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
| `POST` | `/openai/deployments/{deployment}/chat/completions` | Azure OpenAI Chat Completions | The deployment names the Router model; `api-version` is accepted and not forwarded |
| `GET` | `/v1/models` | OpenAI Models | Lists models exposed by the active router configuration |

Other `/v1/*` paths fail closed. In particular, `/v1/files`,
`/v1/vector_stores`, and Router Replay paths are not available on a public
inference listener. Router-owned file and vector-store operations use
`/api/v1/storage/files` and `/api/v1/storage/vector-stores` on the management
listener. Other `/openai/deployments/*` operations, such as embeddings, return
`404`.

See [Protocol Compatibility](../installation/protocol-compatibility) for the
client-to-backend translation matrix, backend `api_format` values, and
field-level portability boundaries.

## Send a routed request

Use an auto-model or recipe entrypoint when you want the router to select a
backend. Use a concrete model name when you want to bypass semantic model
selection and target that model directly.

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
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
See [VSR routing headers](../troubleshooting/vsr-headers) for the stable
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
`providers.models[]`; `routing.modelCards` only describes semantic capabilities.
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

### Responses API

```bash
curl -sS http://localhost:8899/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
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
    "model": "auto",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": "Explain semantic routing in one paragraph."
      }
    ]
  }'
```

### Azure OpenAI clients

Clients built for Azure OpenAI can call the Router as if it were an Azure
resource. The deployment in the path is the model name, so `auto` selects a
route and a concrete model name targets that model directly. The Router checks
the client's `api-key` header against the listener's `api_keys` when they are
set, and removes the header before dispatch.

```bash
curl -sS 'http://localhost:8899/openai/deployments/auto/chat/completions?api-version=2024-10-21' \
  -H 'Content-Type: application/json' \
  -H 'api-key: YOUR-LISTENER-KEY' \
  -d '{"messages":[{"role":"user","content":"Explain semantic routing in one paragraph."}]}'
```

For GitHub Copilot CLI, set `COPILOT_PROVIDER_TYPE=azure`, point
`COPILOT_PROVIDER_BASE_URL` at the listener, and set
`COPILOT_PROVIDER_AZURE_API_VERSION` so the CLI uses the deployment route.
`COPILOT_PROVIDER_WIRE_MODEL` is the Router model name.

Protocol translation is limited to fields the router supports. When a request
crosses protocols, inspect `x-vsr-client-protocol`,
`x-vsr-upstream-protocol`, and any `x-vsr-protocol-warnings` response header.

## Request budget errors

With `candidate_requirements.context: known_limits`, the Router checks estimated
input plus the effective output allowance against the candidates' configured
limits. If all candidates fail only the budget check, it returns HTTP 400:

| Error code | Meaning |
| --- | --- |
| `context_length_exceeded` | The prepared input and requested output do not fit. |
| `max_output_tokens_exceeded` | The requested output exceeds the configured model limit. |

Missing capabilities, unknown limits, unavailable selection evidence, and mixed
failures retain their existing selection-error behavior. Budget checks do not
truncate requests by themselves; opt into
[context compression](../tutorials/plugin/context-compression.md) when appropriate.

These counts are estimates. A backend can still reject a request; its valid
HTTP status and meaningful message are retained. vLLM integer codes are exposed
as strings in OpenAI-compatible errors: `BadRequestError` with `code: 400`
becomes `invalid_request_error` with `code: "400"`.

A streaming request rejected before generation receives the same non-2xx JSON
error, not a successful SSE stream. When Replay is enabled, it records the
failed status and body; Router budget rejections use
`terminal_reason: request_budget_exceeded`.

## Router Replay

Router Replay records routing decisions and selected request lifecycle data.
It is useful for debugging, evaluation, and Router Learning, but it does not
change routing merely because a record is read.

Replay is disabled unless the service is enabled:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: memory
```

The in-memory backend is suitable for local inspection. Use a configured
persistent backend when records must survive process restarts, and set
retention appropriate to the data being captured.

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

List and aggregate requests accept filters such as `recipe`, `decision`,
`model`, `session_id`, `cache_status`, and `search`. Pagination uses `limit`
and `offset`; `limit` is capped at 100. `showDetails=true` requests large body
fields, so use it only when those fields are needed.

Trajectory queries use the exact recipe name. Omitting `recipe` is supported
only when the session's records belong to one recipe; an ambiguous session
returns `400`. An explicit empty `recipe=` selects older, unscoped records.
The response includes each request's route, latency, and lifecycle, including
multiple requests at the same turn index.

Records, trajectory routes, and messages include `conversation_id` when an
explicit conversation identity is available. Messages are grouped by conversation
and turn, so separate conversations in one session can each start at turn zero.
Insights shows their boundaries and complete IDs.

Dashboard Insights shows these routes alongside recorded signals, projections,
candidate scores, and session-switch reasons. In `observe` mode, candidate and
hold explanations describe what protection would have done; the selected model
and route history still describe actual dispatch. Protection's `candidate_models`
lists eligible models independently of their scores; an unrecorded score appears
as `—`, while a recorded zero remains zero. Missing identity or evidence
is displayed explicitly. A recipe's `data_policy.replay: false` prevents its
requests from appearing in Replay, including rejected requests.

### Configured-rate cost estimates

Insights uses recorded token usage and configured input, cached-input,
cache-write, and output rates. These are estimates, not invoices or GPU running
costs; infrastructure charges and billing adjustments are excluded.

| Field | Meaning |
| --- | --- |
| `actual_cost` | Estimated cost of the selected model for the recorded usage. |
| `baseline_cost` | Highest same-currency estimate in the selected recipe's model pool, using that same usage. |
| `baseline_model` | The model used for that comparison. Equal costs use model-name order. |
| `cost_savings` | The difference between the baseline and actual estimate. |
| `currency` | Currency of the recorded estimate; no exchange-rate conversion is applied. |

The baseline covers the recipe's models across all its decisions: model
references, explicit candidate-iteration models, and route destinations. A
permitted default-model fallback is included only for decisions that can use it;
auxiliary planners and judges do not enlarge the pool. Other recipes, unpriced
models, and different currencies are excluded. Direct requests without a recipe
compare against themselves. Alternative-model eligibility and tokenization are
not re-evaluated; this is a rate comparison, not another inference.

Missing usage, price, currency, or baseline stays unknown, with the reason shown
in Insights. Explicitly configured free rates remain zero. Cache hits have zero
additional model-inference cost; cache storage and lookup costs are excluded.
Existing records keep their captured prices and baseline. **Price not recorded**
means the historical record has no price; configuring rates today does not
backfill it.

Aggregates use `summary.by_currency`, a sorted array with `currency`,
`total_saved`, `baseline_spend`, `actual_spend`, and `cost_record_count` for each
currency. With one currency, the flat summary fields mirror that group. With
multiple currencies, flat `currency` is omitted and flat amounts are zero
placeholders: use `by_currency`, not those placeholders. There is no combined
cross-currency total.

`cost_record_count` counts complete estimates. `excluded_record_count` includes
non-completed requests and records missing the data needed for a complete
estimate; it does not imply that every excluded request lacks pricing.

When bearer authentication is enabled, replay callers need `replay.read`.
Prompt, response, tool, and other sensitive details remain redacted unless the
principal also has `replay.detail`. Treat replay storage as potentially
sensitive even when the API normally returns a redacted view.

Replay lifecycle values describe what the recorder observed:

- `in_progress`: no terminal response frame has been recorded yet.
- `completed`: the response finished normally.
- `failed`: routing or the upstream response failed.
- `aborted`: the stream ended without a valid terminal frame, for example
  after a disconnect or timeout.

An HTTP `200` response header alone does not make a streaming record
`completed`.

## Which port should I use?

| Task | Surface |
| --- | --- |
| Send model traffic | Configured Envoy listener; `8899` in the standard local stack |
| List public models | `GET /v1/models` on the inference listener |
| Check health or readiness | Management API on `8080` |
| Read or change configuration | Management API on `8080` |
| Inspect replay records | Management API on `8080` |
| Manage Router-owned files or vector stores | Management API on `8080` under `/api/v1/storage/*` |

Do not expose the management port as a substitute for the public inference
listener. Its endpoints can reveal configuration and operational data or make
state-changing requests.
