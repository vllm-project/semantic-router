# TD033: Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends

## Status

Closed

## Scope

`candle-binding/**`, `onnx-binding/**`, router-side classification/bootstrap seams, and build-tag or modfile selection paths that swap native backends

## Summary

The repository now exposes Candle and ONNX through an explicit typed backend contract instead of assuming backend parity from a nominally compatible Go API. Both binding packages advertise capabilities, supported embedding families, lifecycle semantics, and reload expectations through `CurrentBackendContract()`. Router-side classification and model-runtime bootstrap paths fail early when unified classification, LoRA batch inference, or unsupported embedding families are requested on a backend that does not implement them. The ONNX Go wrapper surface has been brought back into parity for the supported feature set, and Docker build paths now copy the contract sources into runtime images so local and packaged builds see the same backend metadata.

## Evidence

- [candle-binding/semantic-router.go](../../../candle-binding/semantic-router.go)
- [candle-binding/backend_contract.go](../../../candle-binding/backend_contract.go)
- [candle-binding/backend_contract_test.go](../../../candle-binding/backend_contract_test.go)
- [candle-binding/src/ffi/init.rs](../../../candle-binding/src/ffi/init.rs)
- [candle-binding/src/ffi/state_manager.rs](../../../candle-binding/src/ffi/state_manager.rs)
- [onnx-binding/semantic-router.go](../../../onnx-binding/semantic-router.go)
- [onnx-binding/backend_contract.go](../../../onnx-binding/backend_contract.go)
- [onnx-binding/backend_contract_test.go](../../../onnx-binding/backend_contract_test.go)
- [onnx-binding/src/ffi/unified.rs](../../../onnx-binding/src/ffi/unified.rs)
- [onnx-binding/src/ffi/embedding.rs](../../../onnx-binding/src/ffi/embedding.rs)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go)
- [src/semantic-router/pkg/classification/unified_classifier_onnx_contract_test.go](../../../src/semantic-router/pkg/classification/unified_classifier_onnx_contract_test.go)
- [src/semantic-router/pkg/modelruntime/router_runtime.go](../../../src/semantic-router/pkg/modelruntime/router_runtime.go)
- [src/semantic-router/pkg/modelruntime/router_runtime_onnx_contract_test.go](../../../src/semantic-router/pkg/modelruntime/router_runtime_onnx_contract_test.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [e2e/config/onnx-binding/config.onnx-binding-test.yaml](../../../e2e/config/onnx-binding/config.onnx-binding-test.yaml)
- [e2e/config/onnx-binding/config.onnx-classifiers-test.yaml](../../../e2e/config/onnx-binding/config.onnx-classifiers-test.yaml)
- [src/vllm-sr/Dockerfile](../../../src/vllm-sr/Dockerfile)
- [src/vllm-sr/Dockerfile.rocm](../../../src/vllm-sr/Dockerfile.rocm)
- [tools/docker/Dockerfile.extproc](../../../tools/docker/Dockerfile.extproc)
- [tools/docker/Dockerfile.extproc-rocm](../../../tools/docker/Dockerfile.extproc-rocm)

## Why It Matters

- Platform-specific runtime behavior can differ silently because the router currently selects a backend via build tags and modfile indirection while still assuming Candle-compatible semantics in higher layers.
- Process-wide singleton model state makes hot reload, partial shutdown, test isolation, and long-lived process memory control harder as more models and backends are added.
- Unsupported backend features are not surfaced through one capability contract, so callers can accidentally enable LoRA, unified batch, or batched embedding paths that only fail deep inside a backend-specific implementation.
- As the repo expands AMD and non-default deployment stories, binding parity gaps become product-surface gaps rather than an internal implementation detail.

## Desired End State

- Native backends expose a shared capability and lifecycle contract that higher layers can inspect before enabling backend-specific features.
- Candle and ONNX bindings either implement the same steady-state runtime features or explicitly advertise unsupported features through typed capability flags and documented fallbacks.
- Model initialization, cleanup, and reload behavior have one clear owner per backend rather than a mix of public singleton globals and partially adopted state-manager abstractions.
- Router bootstrap and classification code choose features based on backend capabilities instead of assuming parity from a nominally compatible Go API.

## Exit Criteria

- Satisfied on 2026-04-06: runtime code can query backend capabilities for embedding, unified classification, LoRA batch inference, multimodal support, and reload semantics without relying on build-tag knowledge.
- Satisfied on 2026-04-06: unsupported features fail early through explicit capability checks rather than late through stubbed FFI calls or backend-specific placeholder behavior.
- Satisfied on 2026-04-06: binding lifecycle and contract tests cover supported and unsupported init paths for both Candle and ONNX backends, including degraded-startup rejection on unsupported router/bootstrap seams.
- Satisfied on 2026-04-06: backend-selection docs, ONNX config comments, and packaged-image build paths make parity gaps explicit enough that deployment-specific surprises do not surface only at runtime.

## Resolution

- `candle-binding/backend_contract.go` and `onnx-binding/backend_contract.go` now publish a shared typed backend contract, including supported embedding families, feature flags, and reload semantics, with focused tests in the adjacent `backend_contract_test.go` files.
- `src/semantic-router/pkg/classification/unified_classifier.go`, `src/semantic-router/pkg/classification/model_discovery.go`, and `src/semantic-router/pkg/modelruntime/router_runtime.go` now gate unified classification, LoRA batch inference, and embedding-family startup through that contract instead of discovering unsupported paths only after FFI calls fail.
- `onnx-binding/semantic-router.go` now restores missing Go wrapper surface for the supported ONNX backend contract, including consistent init-state tracking, batch embedding access, and helper methods needed by router callers and tests.
- Docker build paths for router and extproc images now copy the backend-contract Go sources so packaged builds use the same capability metadata as local module builds.

## Validation

- `go test -run 'Test(CurrentBackendContract_|InitMmBertEmbeddingModel|InitEmbeddingModels|InitEmbeddingModelsBatched|GetEmbeddingsBatch|CalculateEmbeddingSimilarity|CalculateSimilarityBatch|GetMatryoshkaConfig)$' ./...`
  Run in `/Users/bitliu/vs/onnx-binding`
- `go test -run 'TestCurrentBackendContract_' ./...`
  Run in `/Users/bitliu/vs/candle-binding`
- `CGO_LDFLAGS='-L../../onnx-binding/target/release' go test -modfile=go.onnx.mod -tags=onnx ./pkg/classification ./pkg/modelruntime -run 'Test(InitializeLegacyUnifiedClassifierRejectsUnsupportedONNXBackend|InitializeLoRAUnifiedClassifierRejectsUnsupportedONNXBackend|UnifiedClassifierInitializeRejectsUnsupportedONNXBackend|UnifiedClassifierInitializeLoRABindingsRejectsUnsupportedONNXBackend|ValidateEmbeddingBackendContractRejectsUnsupportedONNXFamilies|ValidateEmbeddingBackendContractAllowsMMBertOnONNX)$'`
  Run in `/Users/bitliu/vs/src/semantic-router`
- `go test ./pkg/classification ./pkg/modelruntime`
  Run in `/Users/bitliu/vs/src/semantic-router`
- `pytest /Users/bitliu/vs/src/vllm-sr/tests/test_runtime_support.py`
- `pytest /Users/bitliu/vs/src/vllm-sr/tests/test_cli_main.py`
- `pytest /Users/bitliu/vs/src/vllm-sr/tests/test_openclaw_shared_network.py -k 'runtime_config or isolated_network'`
- `make agent-validate`
- `make test-semantic-router`
- `make test-binding-minimal`
- `make agent-lint AGENT_CHANGED_FILES_PATH=/tmp/vsr_td033_changed.txt`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td033_changed.txt`
- `make agent-feature-gate ENV=cpu AGENT_CHANGED_FILES_PATH=/tmp/vsr_td033_changed.txt`
