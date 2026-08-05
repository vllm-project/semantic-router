# Router API Reference

The **Router** is the data-plane HTTP surface clients usually call through
**Envoy** (default public port **`8801`**).

Use this page when you want to:

- Send OpenAI-compatible chat or Responses requests through semantic routing
- Understand which frontend and backend protocols are supported
- Query **Router Replay** records after traffic has been captured

For control-plane helpers on port **`8080`** (health, classification, config,
outcomes), see [Router Apiserver API](./apiserver).

## Ports at a glance

| Surface | Default port | Purpose |
| --- | --- | --- |
| Envoy public ingress | `8801` | Client-facing routed HTTP APIs |
| ExtProc gRPC | `50051` | Internal Envoy external processing hook |
| Router apiserver | `8080` | Control and utility APIs (`/health`, `/config/router`, `/api/v1/classify/*`, …) |

## Frontend API

| API surface | Public path | Status | Notes |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Primary routed inference interface |
| OpenAI Responses API | `POST /v1/responses` | Supported | Internally translated to Chat Completions |
| OpenAI Responses API retrieval | `GET /v1/responses/{id}` | Supported | Requires Response API service/store |
| OpenAI Responses API delete | `DELETE /v1/responses/{id}` | Supported | Requires Response API service/store |
| OpenAI Responses API input items | `GET /v1/responses/{id}/input_items` | Supported | Requires Response API service/store |
| OpenAI Models API | `GET /v1/models` | Supported on apiserver | Served by `:8080`; can be re-exposed through Envoy |
| Router Replay list | `GET /v1/router_replay` | Supported when enabled | Served via ExtProc / Envoy path |
| Router Replay detail | `GET /v1/router_replay/{id}` | Supported when enabled | Full record including bodies when captured |
| Router Replay aggregate | `GET /v1/router_replay/aggregate` | Supported when enabled | Cost / decision / token summaries |
| Router Replay trajectory | `GET /v1/router_replay/trajectory` | Supported when enabled | Session message timeline |

## Quick start: chat completions

Send traffic to Envoy (`8801`), not the apiserver (`8080`).

Use model `auto` (or your recipe’s auto / MoM alias) so the router chooses a
backend from signals and decisions.

```bash
curl -sS http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to merge two sorted lists."
      }
    ]
  }'
```

Example OpenAI-compatible response (abbreviated; content and model id depend on
your backends):

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1722787200,
  "model": "qwen-coder",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "def merge(a, b):\n    ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 80,
    "total_tokens": 104
  }
}
```

Tips:

- Explicit model names still work when you want to pin a backend.
- With Router Replay enabled, look for replay correlation headers such as
  `x-vsr-replay-id` (exact header set depends on config) and then query
  `GET /v1/router_replay/{id}`.

## OpenAI Responses API

```bash
curl -sS http://localhost:8801/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "input": "Summarize the benefits of retrieval-augmented generation."
  }'
```

Example response (abbreviated):

```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "base-model",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "RAG improves factual grounding by retrieving relevant documents before generation."
        }
      ]
    }
  ]
}
```

Retrieval / delete / input-items paths require the Response API service/store:

- `GET /v1/responses/{id}`
- `DELETE /v1/responses/{id}`
- `GET /v1/responses/{id}/input_items`

## Router Replay

Router Replay is a durable **flight recorder** for routing decisions. When
enabled, each routed request can write a record (decision, selected model,
tokens, cost, optional bodies, learning diagnostics). Listing and reading those
records does **not** change live routing.

Enable the service (example):

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres   # or memory for local development
```

`global.services.router_replay.enabled` is the router-wide default. A decision can
opt out with a route-local `router_replay` plugin set to `enabled: false`.

Replay HTTP APIs are served on the **Envoy / ExtProc** path (typically
`http://localhost:8801`), not on apiserver `:8080`.

### List recent records

```bash
curl -sS 'http://localhost:8801/v1/router_replay?limit=20'
```

Useful query parameters:

| Param | Default | Notes |
| --- | --- | --- |
| `limit` | `20` | Max `100` |
| `offset` | `0` | Pagination |
| `decision` | | Filter by decision name |
| `model` | | Filter by selected model |
| `recipe` | | Filter by recipe |
| `session_id` | | Filter by session |
| `cache_status` | `all` | `cached` or `streamed` |
| `search` | | Free-text / id search |
| `showDetails` | `false` | Include large body fields (may fail if response is too large) |

Example list response (summary rows omit large bodies by default):

```json
{
  "object": "router_replay.list",
  "count": 1,
  "total": 1,
  "limit": 20,
  "offset": 0,
  "has_more": false,
  "data": [
    {
      "id": "replay_7f3a91",
      "timestamp": "2026-08-04T12:00:00Z",
      "session_id": "sess_alice_42",
      "recipe": "default",
      "decision": "code",
      "selected_model": "qwen-coder",
      "original_model": "auto",
      "from_cache": false,
      "streaming": false,
      "prompt_tokens": 24,
      "completion_tokens": 80,
      "total_tokens": 104
    }
  ]
}
```

### Fetch one full record

```bash
curl -sS http://localhost:8801/v1/router_replay/replay_7f3a91
```

Returns the full routing record, including captured request/response bodies when
those were stored.

### Aggregates for Insights-style dashboards

```bash
curl -sS 'http://localhost:8801/v1/router_replay/aggregate?decision=code'
```

Example response (abbreviated):

```json
{
  "object": "router_replay.aggregate",
  "record_count": 12,
  "summary": {
    "total_saved": 0.003,
    "baseline_spend": 0.01,
    "actual_spend": 0.007,
    "currency": "USD"
  },
  "model_selection": [
    { "name": "qwen-coder", "value": 8 },
    { "name": "base-model", "value": 4 }
  ],
  "decision_distribution": [
    { "name": "default/code", "value": 12 }
  ]
}
```

### Session trajectory

```bash
curl -sS 'http://localhost:8801/v1/router_replay/trajectory?session_id=sess_alice_42'
```

Returns an ordered message timeline reconstructed from replay records for that
session id.

### Link feedback to a replay id

After you have a replay id, submit learning outcomes on the **apiserver**:

```bash
curl -sS http://localhost:8080/v1/router/outcomes \
  -H 'Content-Type: application/json' \
  -d '{
    "replay_id": "replay_7f3a91",
    "source": "agent",
    "target": "model",
    "target_ref": "qwen-coder",
    "verdict": "good_fit",
    "score": 1.0
  }'
```

See [Router Apiserver API](./apiserver#submit-a-router-learning-outcome) for the
full outcome schema.

## Backend model API

These are upstream model protocols the router can target after routing. They are
backend-facing integrations, not necessarily public client ingress paths.

| Backend model API | Upstream path | Status | Notes |
| --- | --- | --- | --- |
| OpenAI-compatible Chat Completions | `/chat/completions` | Supported | Default family for OpenAI-compatible backends |
| Anthropic Messages API | `/v1/messages` | Supported | Converts OpenAI-style requests (including tools); host/path from `backend_refs` |
| vLLM Omni Chat Completions | `/chat/completions` | Supported | Omni / image-generation backends such as `vllm_omni` |

Provider families with OpenAI-compatible chat-completions defaults include
`openai`, `azure-openai`, `bedrock`, `gemini`, and `vertex-ai`.

## Frontend behavior notes

### OpenAI Chat Completions

- Public path: `POST /v1/chat/completions`
- Main router ingress for routed inference
- Works with explicit model names or auto-model aliases such as `MoM` / `auto`

### OpenAI Responses API

- Public paths listed in the frontend table above
- `POST /v1/responses` is translated to Chat Completions internally, then
  translated back to Responses format
- Retrieval and delete require the Response API service/store

## Backend behavior notes

### Anthropic API

- Target Anthropic-backed models with `api_format: anthropic`
- Client ingress remains OpenAI-style Chat Completions or Responses API, not a
  public `POST /v1/messages` requirement for callers
- The router converts upstream to Anthropic `POST /v1/messages` and converts
  responses back to OpenAI-compatible output
- Streaming (`stream: true`) is supported; Anthropic SSE is translated to OpenAI
  `chat.completion.chunk` SSE, including tool-call deltas

### vLLM Omni and multimodal / image generation

- Supported for omni models and image-generation backends such as `vllm_omni`
- When a modality decision resolves to an omni model:
  - Chat Completions requests return the raw omni Chat Completions response
  - Responses API requests are normalized into Responses output items, including
    `image_generation_call` items when images are produced

## Configuration linkage

```yaml
providers:
  models:
    - name: claude-sonnet
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 3.0
        cached_input_per_1m: 0.30
        cache_write_per_1m: 3.75
        completion_per_1m: 15.0
      backend_refs:
        - base_url: https://api.anthropic.com
          provider: anthropic
```

- Upstream targets live under `providers.models[].backend_refs[]`
- Optional cost-aware policies use `pricing:`
- Response API behavior is under `global.services.response_api`
- Replay storage and enablement are under `global.services.router_replay`
- Modality / image generation is configured through routing decisions and
  backends such as `vllm_omni`

## Related docs

- [Router Apiserver API](./apiserver) — classification, config, outcomes on `:8080`
- [Session identification](./session-identification) — how `SessionID` is derived
- [API and observability](../tutorials/global/api-and-observability) — metrics and replay backends
- [Memory and replay](../tutorials/learning/memory-and-replay) — learning + replay layering
