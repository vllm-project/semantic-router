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

## Client-facing protocols

| Client API | Inference endpoint | Buffered | Streaming | Availability |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Supported | Available on the public inference listener. |
| OpenAI Responses | `POST /v1/responses` | Supported | Supported | Requires `global.services.response_api` and its store to be available. Router-owned object operations are not forwarded to a model backend. |
| Anthropic Messages | `POST /v1/messages` | Supported | Supported | Send the Anthropic request shape and an appropriate `anthropic-version` header. Client authentication remains deployment-specific. |

The public listener also serves `GET /v1/models`. See the
[Router API](../api/router) for the complete method and path inventory,
Responses object operations, and request examples.

## Backend model protocols

Set `api_format` on each `providers.models[]` entry. It describes the wire
contract implemented by that model endpoint, not the provider brand.

| `api_format` | Backend request and response shape | Default upstream path | Notes |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | Default when `api_format` is omitted. `backend_refs[].chat_path` can override the Chat path. |
| `responses` | OpenAI Responses | `/v1/responses` | The backend itself must implement the Responses wire contract; enabling the Router's Responses service does not add that API to a backend. |
| `anthropic` | Anthropic Messages | `/v1/messages` | Configure provider authentication and required version headers on the backend ref. |

These fields are easy to confuse:

- model `api_format` selects the request, response, error, and streaming codec;
- backend-ref `protocol` selects HTTP or HTTPS transport; and
- backend-ref `provider` supplies provider-specific authentication and path
  defaults. It does not prove that the endpoint implements an API format.

For Responses and Messages backends, a path in `base_url` is retained and the
protocol path is appended. `chat_path` applies only to Chat Completions.

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
| Strict JSON Schema output | Supported | Supported | Supported |
| Buffered and streaming responses | Supported | Supported | Supported |
| Reasoning content and effort | Supported | Supported | Supported |
| JSON object mode without a schema | Supported | Supported | Not supported |
| Audio input | Supported | Not supported | Not supported |
| Hosted image-generation lifecycle | Not supported | Supported | Not supported |
| Multiple response candidates | Supported | Not supported | Not supported |
| Prompt-cache directives | Supported | Not supported | Supported |
| Reasoning token budget | Supported extension | Not supported | Supported |
| Seed and frequency or presence penalties | Supported | Not supported | Not supported |
| `top_k` sampling | Not supported | Not supported | Supported |
| Stop sequences | Supported | Not supported | Supported |
| Native response or conversation state fields | Not supported | Supported | Not supported |

This table describes codec representation, not model capability. For example,
an OpenAI-compatible server can accept the Chat request shape while rejecting
images or tools for a particular model. Qualify the actual endpoint and model
revision before adding them to a routing pool.

A Responses client can still use `previous_response_id` with a Chat
Completions or Messages backend. The Router retrieves and materializes the
retained history, removes Router-owned object controls, and then encodes the
stateless request in the selected backend format.

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
          base_url: https://api.anthropic.com
          provider: anthropic
          api_key_env: ANTHROPIC_API_KEY
          extra_headers:
            anthropic-version: "2023-06-01"
          weight: 100
```

Test the backend directly with its native path and a minimal request first.
Then send the same semantic request through the Router using the client API that
your application needs. A successful health check does not validate request
schema, streaming, tools, or error translation.

## Validation and failure behavior

- Public requests are decoded into the neutral contract even when client and
  backend formats match. Unknown or unsupported request fields fail closed.
- Cross-protocol requests preserve shared semantics. Target-specific features
  that cannot be represented return a typed protocol error.
- The response keeps the client protocol's JSON or SSE shape. Provider
  transport errors and incomplete streams are translated separately from
  successful model responses.
- `x-vsr-client-protocol`, `x-vsr-upstream-protocol`, and
  `x-vsr-protocol-warnings` expose translation details when applicable. See
  [VSR routing headers](../troubleshooting/vsr-headers).

The repository verifies all three protocols pairwise in codec tests, at the
Envoy ExtProc boundary, and in an 18-cell deployment matrix: three client
formats by three backend formats by buffered or streaming mode. See the
[implemented codec design](../proposals/multi-protocol-adaptor) for the full
verification and extension contract.
