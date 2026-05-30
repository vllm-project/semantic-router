# TD033: Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends

## Status

Open

## Scope

`candle-binding/**`, `onnx-binding/**`, router-side classification/bootstrap seams, and build-tag or modfile selection paths that swap native backends

## Summary

The repository exposes Candle and ONNX backends through a nominally compatible Go surface, but the actual runtime contract still diverges in capability, lifecycle, and failure semantics. The ONNX binding intentionally leaves parts of the unified LoRA classifier path as stubs, returns placeholder probability payloads in some classifier calls, and overloads Candle-compatible APIs such as `InitEmbeddingModelsBatched` with mmBERT-only semantics. The Candle binding, meanwhile, still relies on a large set of process-wide `OnceLock` singletons for model state while also carrying a separate `state_manager` abstraction that is not the primary owner of public initialization flows.

Current-source recheck on 2026-05-24 found the router-side capability portion
narrower than the original debt text. Router classification now exposes
`NativeBackendCapabilities` for the selected build backend, with Candle, ONNX,
and non-CGO stub capability declarations. Unified and LoRA classifier entrypoints
fail early when the backend does not advertise support, and classifier stats
include the current capability contract. A later 2026-05-24 build-hygiene pass
also removed the repeated native binding compiler warnings for unsupported MLP
Metal cfg branches without introducing a broken Metal feature. A later
2026-05-25 native hygiene pass removed the remaining generative Candle binding
unused-import and rand deprecation warnings from the local CPU native build,
made the generative raw-pointer C FFI functions explicitly `unsafe extern "C"`
with documented pointer ownership requirements without changing the C ABI
surface, and split the touched Guard FFI plus Qwen3Guard loading/generation/
sampling hotspots into sibling modules so the changed files stay below the
structure-rule error threshold. A later release
contract pass found the version-consistency scope narrower than the original
text: the release workflow already validates the Python package and Candle
crate against `v*` tags, but the local `make release` helper only updated the
Python package. The helper now updates `src/vllm-sr/pyproject.toml`,
`candle-binding/Cargo.toml`, and the Candle lockfile package entry together, and
the current Candle crate version is aligned to `0.3.0`. A follow-up release
contract pass added one repo-level checker shared by `make release-check`,
`make release`, and the tag validation workflow; that checker validates
Python/Candle/Candle-lockfile version alignment, Helm release packaging
semantics, and Docker release image coverage in the unified release notes. A
later Operator release-image pass found that `operator-ci.yml` already publishes
`semantic-router/operator` and `semantic-router/operator-bundle` on `v*` tags,
but the shared checker and unified release body did not include those images;
the checker now parses the Operator workflow and release notes list both
Operator images. A later upgrade-runbook pass found that the runbook only
documented a partial Docker image set; the shared checker now requires
`website/docs/installation/upgrade-rollback.md` to mention every versioned
release image parsed from Docker and Operator release workflows, and the runbook
lists the full image set. A later simulator release-ownership pass found that
`pypi-publish-vllm-sr-sim.yml` already owned an independent
`vllm-sr-sim-v*` tag stream with package/tag version checks and installation
snippets, but the shared checker and upgrade runbook did not enforce that
contract; the checker now validates those workflow markers plus simulator
release notes and runbook coverage. A later backend-docs pass found that the
router already exposes a typed capability seam, but user-facing installation
docs did not explain Candle, ONNX, or non-CGO support boundaries; the native
backend guide now documents build selection, runtime capabilities, early-failure
semantics, and the current lack of explicit reset. A later upgrade-runbook
fixture pass tightened the shared checker so `make release-check`,
`make release`, and release tag validation now require full tagged release image
references plus pinned Helm/Docker/Python upgrade and rollback fixture markers in
the runbook. A later Candle crate workflow ownership pass found that
`publish-crate.yml` already owns the crate version guard, CPU-only smoke/check/
build, publish dry-run, crates.io publish, and GitHub Release native artifact
attachment, but the shared checker did not enforce those markers; it now does.
The static marker lists live in `tools/release/release_contract_markers.py` so
the checker remains a small orchestrator instead of growing a new release
hotspot.
The debt remains open because real lifecycle or reset ownership, ONNX feature
parity, and live environment upgrade/rollback execution are still unresolved.

## Evidence

- [candle-binding/semantic-router.go](../../../candle-binding/semantic-router.go)
- [candle-binding/Cargo.toml](../../../candle-binding/Cargo.toml)
- [candle-binding/Cargo.lock](../../../candle-binding/Cargo.lock)
- [candle-binding/src/ffi/init.rs](../../../candle-binding/src/ffi/init.rs)
- [candle-binding/src/ffi/state_manager.rs](../../../candle-binding/src/ffi/state_manager.rs)
- [candle-binding/src/ffi/mlp.rs](../../../candle-binding/src/ffi/mlp.rs)
- [candle-binding/src/ffi/generative_classifier.rs](../../../candle-binding/src/ffi/generative_classifier.rs)
- [candle-binding/src/ffi/generative_guard.rs](../../../candle-binding/src/ffi/generative_guard.rs)
- [candle-binding/src/model_architectures/generative/qwen3_guard.rs](../../../candle-binding/src/model_architectures/generative/qwen3_guard.rs)
- [candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_generation.rs](../../../candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_generation.rs)
- [candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_loading.rs](../../../candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_loading.rs)
- [candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_sampling.rs](../../../candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_sampling.rs)
- [onnx-binding/semantic-router.go](../../../onnx-binding/semantic-router.go)
- [onnx-binding/src/ffi/unified.rs](../../../onnx-binding/src/ffi/unified.rs)
- [onnx-binding/src/ffi/embedding.rs](../../../onnx-binding/src/ffi/embedding.rs)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/native_capabilities.go](../../../src/semantic-router/pkg/classification/native_capabilities.go)
- [src/semantic-router/pkg/classification/native_capabilities_test.go](../../../src/semantic-router/pkg/classification/native_capabilities_test.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_onnx.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [website/docs/installation/native-backends.md](../../../website/docs/installation/native-backends.md)
- [src/vllm-sr/pyproject.toml](../../../src/vllm-sr/pyproject.toml)
- [src/fleet-sim/pyproject.toml](../../../src/fleet-sim/pyproject.toml)
- [src/vllm-sr/scripts/release.sh](../../../src/vllm-sr/scripts/release.sh)
- [tools/release/check_version_contract.py](../../../tools/release/check_version_contract.py)
- [tools/release/release_contract_markers.py](../../../tools/release/release_contract_markers.py)
- [tools/make/release.mk](../../../tools/make/release.mk)
- [website/docs/installation/upgrade-rollback.md](../../../website/docs/installation/upgrade-rollback.md)
- [.github/workflows/release.yml](../../../.github/workflows/release.yml)
- [.github/workflows/pypi-publish-vllm-sr-sim.yml](../../../.github/workflows/pypi-publish-vllm-sr-sim.yml)
- [.github/workflows/helm-publish.yml](../../../.github/workflows/helm-publish.yml)
- [.github/workflows/docker-release.yml](../../../.github/workflows/docker-release.yml)
- [.github/workflows/operator-ci.yml](../../../.github/workflows/operator-ci.yml)
- [e2e/config/onnx-binding/config.onnx-binding-test.yaml](../../../e2e/config/onnx-binding/config.onnx-binding-test.yaml)

## Why It Matters

- Platform-specific runtime behavior can differ silently because the router currently selects a backend via build tags and modfile indirection while still assuming Candle-compatible semantics in higher layers.
- Process-wide singleton model state makes hot reload, partial shutdown, test isolation, and long-lived process memory control harder as more models and backends are added.
- Unsupported backend features are now surfaced for the router classification runtime, but the wider backend lifecycle and release story still depends on backend-specific implementation details and docs.
- The MLP FFI `device_type: 2` path is now explicitly a CPU fallback; adding a true Metal feature still needs a dependency and lockfile decision rather than dead `cfg(feature = "metal")` branches.
- The standard CPU native binding build no longer emits the stale generative Candle binding unused-import or rand deprecation warnings, changed-file clippy now treats the generative raw-pointer FFI boundary as an explicit unsafe contract instead of an implicit safe Rust API, and the touched native files no longer trip structure-rule errors.
- The local release helper now writes the same Python and Candle crate version files that the release workflow validates, so maintainers no longer need a manual crate bump to keep the `v*` tag guard green.
- `make release-check`, `make release`, and `.github/workflows/release.yml`
  now use the same version-contract checker instead of separate grep/sed
  parsers for Python, Candle, Helm packaging semantics, and Docker release-note
  image coverage; the checker now includes Operator and Operator bundle images
  that are published by `operator-ci.yml` on release tags, verifies the
  upgrade/rollback runbook contains full tagged image references plus pinned
  upgrade/rollback fixture markers, and checks the independent `vllm-sr-sim-v*`
  package workflow plus runbook contract and the Candle crate publish/native
  artifact workflow.
- `website/docs/installation/native-backends.md` now documents the runtime
  capability differences between Candle, ONNX, and non-CGO builds, including
  the current no-explicit-reset lifecycle contract.
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
- Backend-selection docs and validation paths make parity gaps explicit enough
  that deployment-specific surprises do not surface only at runtime.
- The local release helper and tag validation workflow keep versioned Python
  and Rust artifacts aligned before publish workflows run, and the shared
  release-contract checker covers Helm release packaging plus Docker and
  Operator image release-note and upgrade-runbook coverage, static runbook
  fixture markers, Candle crate publish/native artifact markers, plus the
  independent simulator package release stream.
