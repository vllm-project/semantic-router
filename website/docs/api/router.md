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
| `POST` | `/v1/responses` | OpenAI Responses | Routed generation; no object store is required |
| `GET` | `/v1/responses/{id}` | OpenAI Responses | Reads an object when optional response retention is enabled |
| `DELETE` | `/v1/responses/{id}` | OpenAI Responses | Deletes an object when optional response retention is enabled |
| `GET` | `/v1/responses/{id}/input_items` | OpenAI Responses | Reads retained input items when optional response retention is enabled |
| `POST` | `/v1/messages` | Anthropic Messages | Routed generation through the same neutral protocol runtime |
| `GET` | `/v1/models` | OpenAI Models | Lists models exposed by the active router configuration |
| `POST` | `/v1/router/outcomes` | Router outcome feedback | Authenticated, replay-bound post-response feedback |

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

The Router accepts concrete names from `providers.models[]` and virtual names
from an Entrypoint's `model_names`. Physical connection and invocation policy
stay under `providers`; semantic metadata stays in `routing.modelCards`. The
`provider_model_id` is sent to the upstream provider:

```yaml
providers:
  models:
    - name: local-small
      provider_model_id: served-model
      backend_refs:
        - provider: vllm
          base_url: http://model-server:8000/v1
      control:
        retry:
          count: 1
          on: [unavailable]
      pricing:
        input_cost_per_million_tokens: "0"
        output_cost_per_million_tokens: "0"
routing:
  modelCards:
    - name: local-small
      capabilities: [chat]
global:
  billing:
    currency: USD
```

Pricing is operator-supplied metadata, not a live quote. Keep it aligned with
the provider contract when cost-aware selection or replay accounting uses it.
All Model rates use `global.billing.currency`. When an empty Management store is
initialized, the Namespace adopts that currency and becomes authoritative.

### Responses API

```bash
curl -sS http://localhost:8899/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "input": "Summarize the trade-offs of retrieval-augmented generation."
  }'
```

`POST /v1/responses` always uses the routed generation path and does not depend on
response retention. The optional response-retention service controls only object
lookup, deletion, input-item history, and `previous_response_id` continuity. Those
object operations return `404` when retention is disabled.

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

Every supported client format is decoded to the same neutral request and every
selected backend is encoded from it. Unsupported or lossy semantics fail before
dispatch unless the configured fidelity policy explicitly permits the conversion;
permitted loss produces bounded diagnostics without exposing request content.

## Submit outcome feedback

Router-native access exposes outcome feedback on the public inference listener. Use
the same API key that made the inference request, or a delegated inference
session derived from that key. The Router binds feedback to the durable replay
and derives namespace, logical key, User, Team, and authentication source from
that credential; those fields are not accepted in the request body.

The inference response returns `x-vsr-replay-id`. For a Model outcome, send the
exact Model identity and revision that served that replay:

```bash
curl -sS http://localhost:8899/v1/router/outcomes \
  -H "Authorization: Bearer ${VSR_API_KEY}" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: feedback-123' \
  -d '{
    "replay_id": "replay_01J...",
    "target": "model",
    "target_ref": "model/served",
    "target_revision": 7,
    "verdict": "good_fit",
    "reason": "Matched the workload.",
    "score": 0.9
  }'
```

`replay_id`, `target`, and `verdict` are required. A Model target also requires
`target_ref` and a positive `target_revision`. Supported verdicts are
`good_fit`, `underpowered`, `overprovisioned`, and `failed`. Bodies, reasons,
metadata, and idempotency keys are bounded.

A new outcome returns `201`; retrying the identical body and idempotency key
returns the original receipt with `200`. Reusing that key for another body
returns `409`. Unknown replays, replays owned by another logical key, and Model
claims that do not match the served Model all return the same `404` response.
The endpoint has a separate global abuse limit and does not consume inference
request, token, or cost quota because it performs no Model dispatch.

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
| Submit outcome feedback | `POST /v1/router/outcomes` on the inference listener |
| Check health or readiness | Management API on `8080` |
| Read or change configuration | Management API on `8080` |
| Inspect replay records | Management API on `8080` |

Do not expose the management port as a substitute for the public inference
listener. Its endpoints can reveal configuration and operational data or make
state-changing requests.
