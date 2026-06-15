# OpenAI API Type Contracts

Inventory of OpenAI-facing request/response types in Semantic Router, their SDK
alignment status, and documented differences. Policy rules live in
[architecture-guardrails.md](architecture-guardrails.md#api-type-contracts).

Related issue: [#1217](https://github.com/vllm-project/semantic-router/issues/1217).

## SDK baseline

- Go SDK: `github.com/openai/openai-go`
- Chat Completions, streaming chunks, and tool calling use SDK types end-to-end
  in the router hot path (`#1550`, `#1712`).

## Inventory

### Chat Completions (`/v1/chat/completions`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/extproc/utils.go` | parse/serialize via `openai.ChatCompletionNewParams` | SDK | Request ingress |
| `pkg/extproc/openai_compat_test.go` | round-trip tests | SDK | Regression guard |
| `pkg/cache/cache.go` | `openai.ChatCompletionNewParams` | SDK | Cache key extraction |
| `pkg/cache/sdk_compat_test.go` | compat tests | SDK | |
| `pkg/memory/extractor.go` | `openai.ChatCompletionMessageParamUnion` | SDK | Memory enrichment |
| `pkg/responseapi/translator.go` | `openai.ChatCompletionNewParams`, `openai.ChatCompletion` | SDK | Response API translation |
| `pkg/classification/vllm_client.go` | SDK + `extra_body` composition | SDK + extension | vLLM-specific field |
| `pkg/modelselection/benchmark_runner.go` | SDK + error envelope | SDK + extension | Provider error detection |
| `pkg/mcp/types.go` | SDK tool types | SDK | Tool conversion |
| `pkg/utils/http/response.go` | `openai.ChatCompletion`, `openai.ChatCompletionChunk` | SDK | Response normalization |
| `pkg/extproc/request_context.go` | `ChatCompletionMessage` | Internal | Role + plain text only; not wire format |
| `dashboard/backend/handlers/openclaw_*.go` | `openAIChat*` structs | Custom | Dashboard proxy; minimal subset |
| `internal/nlgen/` | `ChatCompletionRequest` | Custom | Standalone NL tool; avoids router import cycle |
| `e2e/pkg/fixtures/chat.go` | `ChatCompletionsRequest` | Custom | E2E isolation |

### Streaming

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/extproc/processor_res_body_streaming.go` | `openai.ChatCompletionChunk` | SDK | Chunk accumulation |
| `pkg/extproc/openai_compat_test.go` | chunk parsing tests | SDK | |
| `pkg/anthropic/compat_test.go` | OpenAI chunk to Anthropic | SDK source | Cross-protocol |

### Models list (`GET /v1/models`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/extproc/req_filter_models.go` | `OpenAIModel`, `OpenAIModelList` | Custom | Router-generated list |
| `pkg/extproc/models_compat_test.go` | wire-format tests | Partial | Core fields match `openai.Model` |
| `pkg/apiserver/config.go` | `OpenAIModel`, `OpenAIModelList` | Custom | Dashboard status mirror |

**Documented differences from `openai.Model`:**

- Extra fields: `description`, `logo_url` (Chat UI extensions)
- `owned_by` is always `vllm-semantic-router` for router-managed entries

### Response API (`/v1/responses`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/responseapi/types.go` | `ResponseAPIRequest`, `ResponseAPIResponse`, items | Custom | Stateful API; SDK `Response` not used yet |
| `pkg/responseapi/translator.go` | translates to/from Chat Completions SDK types | SDK at boundary | |
| `pkg/responseapi/*_test.go` | translation + round-trip | Custom + SDK boundary | |
| `pkg/responseapi/types_compat_test.go` | JSON round-trip | Custom | Wire stability guard |
| `e2e/pkg/fixtures/response_api.go` | fixture types | Custom | E2E isolation |

**Documented differences:**

- Router extension fields: `auto_store` (memory e2e)
- Translation layer uses `openai.ChatCompletionNewParams` / `openai.ChatCompletion`
  for backend calls; Response API wire types remain custom until SDK coverage is
  adopted end-to-end

### Image generation (`/v1/images/generations`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/imagegen/backend_openai.go` | `openAIImageRequest`, `openAIImageResponse` | Custom | Outbound client only |
| `pkg/imagegen/backend_openai_compat_test.go` | wire-format tests | Partial | Overlaps `openai.ImageGenerateParams`, `openai.ImagesResponse` |

**Documented differences:**

- Custom structs cover the subset used by the image-gen plugin (model, prompt,
  size, quality, style, url/b64_json). SDK types parse the same wire JSON for
  those fields.

### Files API (`/v1/files`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/openai/filestore.go` | `File`, `FileListResponse` | Custom | Outbound RAG client |
| `pkg/openai/wire_compat_test.go` | wire-format tests | Partial | Core fields match `openai.FileObject` |
| `pkg/openai/openai_validation_e2e_test.go` | live API validation | Custom | `openai_validation` build tag |

### Vector Stores API (`/v1/vector_stores`)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/openai/vectorstore.go` | `VectorStore`, list/create/search types | Custom | Outbound RAG client |
| `pkg/openai/wire_compat_test.go` | wire-format tests | Partial | Core fields match `openai.VectorStore` |
| `e2e/testing/09-openai-api-validation-test.py` | live API validation | Custom | Files + Vector Stores |

### Other (config / telemetry / tests only)

| Location | Types | SDK alignment | Notes |
|----------|-------|---------------|-------|
| `pkg/config/image_gen_plugin.go` | `OpenAIImageGenConfig` | Config schema | Not wire format |
| `pkg/config/rag_plugin_backends.go` | `OpenAIRAGConfig` | Config schema | Not wire format |
| `pkg/sessiontelemetry/telemetry.go` | `ResponseAPIInput` | Internal telemetry | Not client-facing |
| `pkg/classification/hallucination_detector_test.go` | test-only structs | Custom | Test simulation only |

## Regression tests

| Package | Test file | Coverage |
|---------|-----------|----------|
| `pkg/extproc` | `openai_compat_test.go` | Chat Completions request/response/streaming |
| `pkg/cache` | `sdk_compat_test.go` | Request extraction |
| `pkg/anthropic` | `compat_test.go` | OpenAI to Anthropic conversion |
| `pkg/responseapi` | `translator_test.go`, `roundtrip_test.go`, `types_compat_test.go` | Response API wire + translation |
| `pkg/extproc` | `models_compat_test.go` | Models list wire format |
| `pkg/imagegen` | `backend_openai_compat_test.go` | Image generation wire format |
| `pkg/openai` | `wire_compat_test.go` | Files + Vector Stores wire format |
| `pkg/openai` | `openai_validation_e2e_test.go` | Live API (optional, build tag) |
| `e2e/testing` | `09-openai-api-validation-test.py` | Live Files/Vector Stores |

## When to adopt SDK types

1. Hot-path request/response handling: always prefer SDK types.
2. Outbound clients with narrow field usage: custom minimal structs are OK if
   covered by `wire_compat_test.go` and listed here.
3. E2E fixtures and standalone tools: custom minimal types OK with a comment
   citing isolation.
