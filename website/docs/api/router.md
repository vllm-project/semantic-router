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

Other `/v1/*` paths fail closed. In particular, Router Replay paths are not
available on a public inference listener.

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
          type: vllm
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
curl -sS 'http://localhost:8080/v1/router_replay?limit=20' \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/router_replay` | List and filter records |
| `GET` | `/v1/router_replay/{id}` | Read one record |
| `GET` | `/v1/router_replay/aggregate` | Aggregate routing and cost metadata |
| `GET` | `/v1/router_replay/trajectory?session_id=...` | Reconstruct one session trajectory |

List and aggregate requests accept filters such as `recipe`, `decision`,
`model`, `session_id`, `cache_status`, and `search`. Pagination uses `limit`
and `offset`; `limit` is capped at 100. `showDetails=true` requests large body
fields, so use it only when those fields are needed.

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

Do not expose the management port as a substitute for the public inference
listener. Its endpoints can reveal configuration and operational data or make
state-changing requests.
