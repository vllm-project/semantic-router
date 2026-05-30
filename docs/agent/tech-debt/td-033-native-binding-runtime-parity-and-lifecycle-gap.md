# TD033: Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

`candle-binding/**`, `onnx-binding/**`, router-side native classification
bootstrap, backend capability selection, and release/version checks that publish
native artifacts.

## Summary

Candle and ONNX expose similar Go-facing binding surfaces, but their runtime
contracts are not yet equivalent. ONNX still has unsupported or placeholder
paths for parts of the unified classifier flow, while Candle still relies on
process-wide model state and a separate state-manager abstraction that is not
the clear lifecycle owner for public initialization. The router now has an
explicit native backend capability seam, but reset, reload, feature parity, and
live upgrade/rollback confidence remain incomplete.

## Evidence

- [candle-binding/semantic-router.go](../../../candle-binding/semantic-router.go)
- [candle-binding/src/ffi/init.rs](../../../candle-binding/src/ffi/init.rs)
- [candle-binding/src/ffi/state_manager.rs](../../../candle-binding/src/ffi/state_manager.rs)
- [candle-binding/src/ffi/generative_classifier.rs](../../../candle-binding/src/ffi/generative_classifier.rs)
- [candle-binding/src/ffi/generative_guard.rs](../../../candle-binding/src/ffi/generative_guard.rs)
- [onnx-binding/semantic-router.go](../../../onnx-binding/semantic-router.go)
- [onnx-binding/src/ffi/unified.rs](../../../onnx-binding/src/ffi/unified.rs)
- [onnx-binding/src/ffi/embedding.rs](../../../onnx-binding/src/ffi/embedding.rs)
- [src/semantic-router/pkg/classification/native_capabilities.go](../../../src/semantic-router/pkg/classification/native_capabilities.go)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [website/docs/installation/native-backends.md](../../../website/docs/installation/native-backends.md)
- [tools/release/check_version_contract.py](../../../tools/release/check_version_contract.py)
- [tools/release/release_contract_markers.py](../../../tools/release/release_contract_markers.py)

## Why It Matters

- Backend-specific behavior can differ silently unless capability, lifecycle,
  and failure semantics are explicit.
- Process-wide model state makes hot reload, partial shutdown, memory control,
  and test isolation harder.
- Release confidence for v0.3 depends on the native backend contract matching
  what installation docs and release checks claim.

## Desired End State

- Candle and ONNX advertise a clear capability matrix consumed by router
  startup and classifier paths.
- Unsupported backend features fail early with actionable errors.
- Native model lifecycle has explicit init, readiness, shutdown/reset, and
  reload semantics.
- Release checks cover native package, image, docs, and upgrade/rollback claims
  without relying on manual maintainer memory.

## Exit Criteria

- ONNX unsupported paths are either implemented or clearly blocked by capability
  checks and tests.
- Candle public initialization flows have one lifecycle owner and a documented
  reset/shutdown policy.
- Native backend smoke, release-version checks, and upgrade/rollback docs all
  describe the same supported v0.3 backend contract.
