# Copilot Agent Instructions — vLLM Semantic Router

Purpose: help AI coding agents work effectively in this repo by knowing the architecture, conventions, and non-obvious workflows.

## Big picture

- This is a Mixture-of-Models router for LLM requests with: Envoy External Processing (gRPC) for request routing, classification (intent/PII/security), semantic similarity caching, and tool auto-selection.
- Primary implementation is Go with a Rust ML binding (HuggingFace Candle) via CGO for embeddings/similarity. A small HTTP Classification API is exposed alongside the gRPC extproc server.

## Core components (key files)

- Entry point: `src/semantic-router/cmd/main.go` (starts gRPC extproc, Classification API, and Prometheus metrics)
- Envoy ExtProc server: `src/semantic-router/pkg/extproc/` (stream handlers, routing logic, request/response transforms)
- Configuration: `config/config.yaml` (routing categories, model_config, reasoning families, semantic cache backend, vLLM endpoints, tools DB, classifiers)
- Classification API: `src/semantic-router/pkg/api/server.go` (e.g., POST `/api/v1/classify/intent|pii|security|batch`)
- Config loader/utilities: `src/semantic-router/pkg/config/` (hot-reload support, endpoint selection, policy helpers)
- Cache backends: `src/semantic-router/pkg/cache/` (in-memory or Milvus; compile-time tag `milvus`)
- Tools database: `src/semantic-router/pkg/tools/` (semantic tool selection)
- Candle Rust binding (CGO): `candle-binding/` (builds native lib used for similarity)
- Tests: Go unit/integration under `src/semantic-router/pkg/**`, e2e in `e2e-tests/`, research/bench suite in `bench/`

## How things talk to each other

1. Client → Envoy → gRPC ExtProc (`extproc.Server`) → Router selects model/tools/reasoning and edits OpenAI-compatible request → forwards to chosen vLLM endpoint.
2. Router uses Candle embeddings for similarity cache and tool selection.
3. Classification uses either legacy ModernBERT models or auto-discovered LoRA unified classifiers (services initialize a global ClassificationService).
4. Config changes are hot-reloaded (fsnotify) without restarting the gRPC server.

## Build / run workflows (non-obvious bits)

- Makefile orchestrates sub-makefiles under `tools/make/`
  - Build router (also builds Rust lib): `make build-router`
  - Run router with config: `CONFIG_FILE=config/config.yaml make run-router`
  - Run Envoy (installs func-e if missing): `make run-envoy`
  - Download local models from HF Hub: `make download-models` (uses `hf download` CLI)
- Dynamic library path on macOS: prefer `DYLD_LIBRARY_PATH` to point to `candle-binding/target/release`; Linux uses `LD_LIBRARY_PATH`. The Makefile sets `LD_LIBRARY_PATH`—on macOS set `DYLD_LIBRARY_PATH` in zsh if needed.
- Ports: gRPC extproc `:50051` (flag `-port`), Classification API `:8080` (`-api-port`), Prometheus `:9190` (`-metrics-port`).
- Docker: `docker-compose.yml` spins up router + Envoy (+ optional testing profile).

Example (zsh):

```sh
# Build native lib + router
make build-router

# If macOS, ensure Candle dylib is discoverable for CGO
export DYLD_LIBRARY_PATH="$PWD/candle-binding/target/release:$DYLD_LIBRARY_PATH"

# Run router with the default config and metrics
CONFIG_FILE=config/config.yaml make run-router

# Run Envoy (separate terminal)
make run-envoy
```

## Configuration patterns (edit `config/config.yaml`)

- `categories[]` with per-category `model_scores` and reasoning flags drive model selection; `default_model` is the fallback.
- `model_config` + `reasoning_families` normalize “reasoning mode” syntax across model families (e.g., deepseek, qwen3, gpt-oss). Use `GetModelReasoningFamily()` helpers, don’t hardcode.
- `semantic_cache`: `backend_type: memory|milvus`, `similarity_threshold`, `ttl_seconds`. For Milvus, run `make start-milvus` and test with `-tags=milvus`.
- `tools`: enable semantic tool selection via `tools_db_path` (JSON), `top_k`, and threshold (defaults to BERT threshold if unset).
- `classifier`: paths to ModernBERT/LoRA models and mapping jsons; batch endpoint requires unified classifier to be available.
- `vllm_endpoints[]`: list models per endpoint; selection respects per-model `preferred_endpoints` and weights.

## Testing

- Go vet and tidy: `make vet` and `make check-go-mod-tidy`
- Unit tests (Go): `make test-semantic-router` (set `SKIP_MILVUS_TESTS=false` to include Milvus) or `go test -v ./...` under `src/semantic-router`
- Milvus-specific: `make test-milvus-cache` or `make test-semantic-router-milvus` (uses `-tags=milvus`)
- E2E Python tests: see `e2e-tests/README.md` (requires router+envoy running)
- Quick cURL demos: `make test-auto-prompt-reasoning`, `test-pii`, `test-tools` (hits Envoy at `http://localhost:8801/v1/chat/completions` with `model: "auto"`)

## Conventions & tips for contributors (agents)

- Use config accessors from `pkg/config` (e.g., endpoint selection, PII policies). Avoid duplicating selection logic.
- Prefer `services.*ClassificationService` APIs for classification; a global service may be set by auto-discovery.
- Respect streaming in ExtProc handlers and record metrics via `pkg/metrics`.
- Keep hot-reload safe: re-create `OpenAIRouter` on config changes using `Server.watchConfigAndReload` pattern.
- When adding cache/tool logic, use existing interfaces: `cache.CacheBackend`, `tools.ToolsDatabase`.

References

- Router main: `src/semantic-router/cmd/main.go`
- ExtProc: `src/semantic-router/pkg/extproc/`
- Config: `config/config.yaml`, helpers in `src/semantic-router/pkg/config/`
- Candle binding: `candle-binding/`
- Bench: `bench/` (CLI and plots)
- Docs site: `website/` (Docusaurus)
