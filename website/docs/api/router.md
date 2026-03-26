# Router API Reference

The router is the data-plane HTTP surface typically exposed through Envoy.

For control-plane endpoints such as health, config, and discovery, see [Router Apiserver API](./apiserver).

## Entry Points

| Surface | Default Port | Purpose |
| --- | --- | --- |
| Envoy public ingress | `8801` | Client-facing routed HTTP APIs |
| ExtProc gRPC | `50051` | Internal Envoy external processing hook |
| Router apiserver | `8080` | Control and utility APIs such as `/v1/models`, `/health`, and `/config/router` |

## Frontend API

| API surface | Public path | Status | Notes |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Primary routed inference interface |
| OpenAI Responses API | `POST /v1/responses` | Supported | Internally translated to Chat Completions |
| OpenAI Responses API retrieval | `GET /v1/responses/{id}` | Supported | Requires Response API service/store |
| OpenAI Responses API delete | `DELETE /v1/responses/{id}` | Supported | Requires Response API service/store |
| OpenAI Responses API input items | `GET /v1/responses/{id}/input_items` | Supported | Requires Response API service/store |
| OpenAI Models API | `GET /v1/models` | Supported on apiserver | Served by `:8080`; can be re-exposed through Envoy if desired |

## Backend Model API

These are upstream model protocols the router can target after routing. They are backend-facing integrations, not necessarily public client ingress paths.

| Backend model API | Upstream path | Status | Notes |
| --- | --- | --- | --- |
| OpenAI-compatible Chat Completions | `/chat/completions` | Supported | Default family used for OpenAI-compatible backends |
| Anthropic Messages API | `/v1/messages` | Supported | Router converts OpenAI-style requests to Anthropic format before forwarding |
| vLLM Omni Chat Completions | `/chat/completions` | Supported | Used for omni and image-generation backends such as `vllm_omni` |

Provider families with OpenAI-compatible chat-completions defaults include `openai`, `azure-openai`, `bedrock`, `gemini`, and `vertex-ai`.

## Frontend Behavior

### OpenAI Chat Completions

- Public request path: `POST /v1/chat/completions`
- This is the main router ingress for routed inference.
- Works with explicit model names or the router auto-model name such as `MoM` or `auto`.

Minimal request:

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": "What is the derivative of x^2?"
    }
  ]
}
```

### OpenAI Responses API

- Public request paths:
  - `POST /v1/responses`
  - `GET /v1/responses/{id}`
  - `DELETE /v1/responses/{id}`
  - `GET /v1/responses/{id}/input_items`
- The router translates `POST /v1/responses` into Chat Completions internally, then translates the backend response back into Responses API format.
- Retrieval and delete paths require the Response API service/store to be enabled.

Minimal request:

```json
{
  "model": "auto",
  "input": "Summarize the benefits of retrieval-augmented generation."
}
```

## Backend Behavior

### Anthropic API

- The router can target Anthropic-backed models when a model is configured with `api_format: anthropic`.
- Anthropic support lives in the backend model API layer.
- Client ingress is still OpenAI-style Chat Completions or Responses API, not `POST /v1/messages`.
- The router converts the upstream request to Anthropic `POST /v1/messages` and converts the response back to OpenAI-compatible output.
- Streaming is not supported for Anthropic-backed routing.

### vLLM Omni and Multimodal/Image Generation

- The router supports multimodal/image-generation routing with omni models and image-generation backends.
- `vllm_omni` is a supported image-generation backend type.
- When a modality decision resolves to an omni model:
  - Chat Completions requests return the raw omni Chat Completions response.
  - Responses API requests are normalized into Responses API output items, including `image_generation_call` items when images are produced.
- This is the path used for multimodal or image-generation decisions rather than a separate public protocol.

## Configuration Linkage

Upstream targets and provider-specific behavior come from the standard router config:

```yaml
providers:
  models:
    - name: claude-sonnet
      api_format: anthropic
      pricing:
        currency: USD
        prompt_per_1m: 3.0
        completion_per_1m: 15.0
      backend_refs:
        - base_url: https://api.anthropic.com
          provider: anthropic
```

- Upstream routing targets are configured under `providers.models[].backend_refs[]`.
- Optional cost-aware policies can use `pricing:`.
- Response API behavior is configured under `global.services.response_api`.
- Modality and image-generation behavior is configured through routing decisions and image-generation backends such as `vllm_omni`.
