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
| `GET` | `/v1/models` | OpenAI Models | Lists models exposed by the active router configuration |

Other `/v1/*` paths fail closed. In particular, `/v1/files`,
`/v1/vector_stores`, and Router Replay paths are not available on a public
inference listener. Router-owned file and vector-store operations use
`/api/v1/storage/files` and `/api/v1/storage/vector-stores` on the management
listener.

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

Protocol translation is limited to fields the router supports. When a request
crosses protocols, inspect `x-vsr-client-protocol`,
`x-vsr-upstream-protocol`, and any `x-vsr-protocol-warnings` response header.

## Request budget errors

When `candidate_requirements.context` is `known_limits`, the Router checks the
estimated input plus the effective output limit against each candidate's configured
limits. If every candidate fails only this budget check, the request returns HTTP
`400` with `context_length_exceeded` or `max_output_tokens_exceeded`. Unknown model
limits, missing capabilities, unavailable selection evidence, and mixed failures
remain selection failures; they are not reported as a caller budget error.

Input accounting is an estimate, not the selected model's tokenizer. A backend
may still reject a request that passed this check. Valid backend errors retain
their HTTP status and meaningful message. vLLM's integer HTTP error codes are
normalized to strings; for example, its `BadRequestError` with `code: 400` becomes
an `invalid_request_error` with `code: "400"` in OpenAI-compatible output.

A request with `stream: true` that is rejected before generation receives the
same non-2xx JSON error rather than a successful SSE stream. When Replay is
enabled, these errors are retained as failed requests with the observed status;
Router budget rejections use the terminal reason `request_budget_exceeded`.
These checks do not themselves truncate the provider-bound conversation.

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

Insights estimates model costs from recorded token usage and the selected model's
configured input, cached-input, cache-write, and output rates. These estimates
exclude infrastructure charges and invoice adjustments. Missing usage or pricing
stays unknown; record cells explain whether usage, price, or a baseline was not
recorded. An explicitly configured free rate remains zero.

For each new routed record, the baseline is the highest estimated cost in the
selected recipe's complete model pool across all its decisions, including models
outside the matched decision. This includes decision model references, explicit
candidate-iteration models, and route-action destinations. The router default is
included only when a recipe decision with no model references permits that
fallback; strict candidate requirements, minimum-candidate constraints, immediate
responses, and route actions do not implicitly admit the default. Auxiliary
planner and judge models do not expand the pool.
The comparison uses the same recorded usage and currency.
Other recipes, unpriced models, and other currencies are ignored;
no exchange-rate conversion is performed. Equal-cost candidates use model-name
order for a stable baseline. A direct model request without a selected recipe compares
against itself. If the recipe has no priced model in the same currency, its baseline
remains unavailable. Candidate eligibility and tokenization on an alternative model
are not re-evaluated: this is a configured-rate comparison, not a second inference.
Cache hits record zero additional model-inference cost; storage and lookup costs
are outside this estimate. Existing records retain their captured baseline and
prices. Historical records without a captured price show **Price not recorded**;
adding or changing today's configured rates does not backfill their costs.

The aggregate response's `summary.by_currency` contains a sorted array of
`currency`, `total_saved`, `baseline_spend`, `actual_spend`, and `cost_record_count`
for each currency. Complete estimates are retained in their own group. With one
currency, the existing flat summary fields retain those same values. With several
currencies, flat `currency` is omitted and flat amounts are zero placeholders;
clients must use `by_currency` rather than display or combine those placeholders.
No cross-currency total is reported. `cost_record_count` counts complete estimates
across groups, while `excluded_record_count` counts non-completed requests and
records without complete usage, price, currency, or baseline data. Details
distinguish those unavailable-data reasons.

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
