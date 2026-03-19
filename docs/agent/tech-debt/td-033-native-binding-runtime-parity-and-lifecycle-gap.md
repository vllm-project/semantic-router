# TD033: Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends

## Status

Open

## Scope

`candle-binding/**`, `onnx-binding/**`, router-side classification/bootstrap seams, and build-tag or modfile selection paths that swap native backends

## Summary

The repository exposes Candle and ONNX backends through a nominally compatible Go surface, but the actual runtime contract still diverges in capability, lifecycle, and failure semantics. The ONNX binding intentionally leaves parts of the unified LoRA classifier path as stubs, returns placeholder probability payloads in some classifier calls, and overloads Candle-compatible APIs such as `InitEmbeddingModelsBatched` with mmBERT-only semantics. The Candle binding, meanwhile, still relies on a large set of process-wide `OnceLock` singletons for model state while also carrying a separate `state_manager` abstraction that is not the primary owner of public initialization flows. Router bootstrap and classifier code assume a Candle-shaped API and backend parity, yet there is no typed capability registry or explicit lifecycle contract that tells the runtime which features are actually available, resettable, or safe to reload on a given backend.

## Evidence

- [candle-binding/semantic-router.go](../../../candle-binding/semantic-router.go)
- [candle-binding/src/ffi/init.rs](../../../candle-binding/src/ffi/init.rs)
- [candle-binding/src/ffi/state_manager.rs](../../../candle-binding/src/ffi/state_manager.rs)
- [onnx-binding/semantic-router.go](../../../onnx-binding/semantic-router.go)
- [onnx-binding/src/ffi/unified.rs](../../../onnx-binding/src/ffi/unified.rs)
- [onnx-binding/src/ffi/embedding.rs](../../../onnx-binding/src/ffi/embedding.rs)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [e2e/config/onnx-binding/config.onnx-binding-test.yaml](../../../e2e/config/onnx-binding/config.onnx-binding-test.yaml)

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

- Runtime code can query backend capabilities for embedding, unified classification, LoRA batch inference, multimodal support, and reload semantics without relying on build-tag knowledge.
- Unsupported features fail early through explicit capability checks rather than late through stubbed FFI calls or backend-specific placeholder behavior.
- Binding lifecycle tests cover init, repeated init, cleanup or reset, and degraded startup semantics for both Candle and ONNX backends.
- Backend-selection docs and validation paths make parity gaps explicit enough that deployment-specific surprises do not surface only at runtime.
