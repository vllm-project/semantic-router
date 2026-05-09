# TD040: Kubernetes Reconcile Path Skips File-Config Family Validators

## Status

Open

## Scope

`pkg/config` validator dispatch and `pkg/k8s` reconciler integration

## Summary

`validateConfigStructure` in `src/semantic-router/pkg/config/validator.go` early-returns when `cfg.ConfigSource == ConfigSourceKubernetes`, skipping every per-family validator the file-config path runs. The early-return was originally added because `decisions` and `model_config` are loaded from CRDs in K8s mode and the file-config-shape assertions did not apply to them. The unintended consequence is that contract validators which DO apply equally to both sources are also skipped on the K8s reconcile path. CRD-loaded configs can therefore be admitted and marked Ready when the same configuration loaded from file would be rejected.

PR #1880's review surfaced one slice of this gap (embedding-modality contracts). The follow-up PR closes the embedding slice by exporting `ValidateEmbeddingContracts` from `pkg/config` and calling it explicitly from `Reconciler.validateAndUpdate` after `ParseYAMLBytes`. The remaining family validators that the early-return still bypasses on the K8s path are tracked here.

## Evidence

- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go) lines 89-119: `validateConfigStructure` early-return at lines 93-95; the validator dispatch table at lines 100-112 enumerates every family validator skipped when the early-return fires.
- [src/semantic-router/pkg/k8s/reconciler.go](../../../src/semantic-router/pkg/k8s/reconciler.go): `validateAndUpdate` invokes `config.ParseYAMLBytes` (which calls `validateConfigStructure`) on the K8s reconcile path; the embedding-modality slice is now applied directly after `ParseYAMLBytes` returns.
- [src/semantic-router/pkg/config/validator_embedding.go](../../../src/semantic-router/pkg/config/validator_embedding.go): the embedding-modality validator and the exported `ValidateEmbeddingContracts` wrapper the embedding-slice PR added.
- [src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go](../../../src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go): the reconcile-path test that exercises the embedding slice through a real `Reconciler.validateAndUpdate` call.

## Why It Matters

- The K8s reconcile path is the operator-mode entrypoint. Every CRD that contradicts a runtime contract the validator can detect should be marked `Ready=False` with a clear message, not admitted silently.
- The validators currently skipped each protect a real invariant (domain coverage, projection consistency, knowledge-base wiring, conversation rules, decision graph integrity, modality contracts, model-selection consistency, tool-filtering consistency). A contributor who ships any of these contracts via CRDs gets the same false-Ready failure mode rootfs identified for embedding modality on PR #1880.
- One-off mirroring per validator (the pattern the embedding-slice PR uses) does not scale. As file-config validators evolve, the K8s reconcile path will keep drifting unless the dispatch is unified.

## Desired End State

A single dispatch surface that runs every K8s-safe family validator on both code paths. Two design directions are plausible; either retires the gap:

- **Selective early-return**: split `validateConfigStructure` into a Kubernetes-safe subset (everything except `decisions` and `model_config` shape checks) and a file-only subset. K8s reconcile calls the safe subset; file-config calls both.
- **Per-validator opt-in**: tag each family validator with a `KubernetesSafe bool` (or move them into separate dispatch lists). `validateConfigStructure` runs the right list based on `ConfigSource`.

Either way, the embedding-modality wrapper exported by the slice PR remains valid as a public seam for callers that want to validate a single contract without dispatching the whole set.

## Exit Criteria

- The K8s reconcile path runs the same family-validator surface that file-config does, except for the explicitly-K8s-incompatible cases (`decisions`, `model_config` shape) which are documented and tagged.
- `Reconciler.validateAndUpdate` no longer needs to call individual exported validators (or, equivalently, the exported validators become internal again because the dispatch covers them).
- The test pattern in `reconciler_embedding_modality_test.go` is replaced by a single dispatch-level test that asserts every K8s-safe contract fires on the reconcile path, with the per-family unit tests remaining the source of truth for branch coverage.
- Validators currently skipped on the K8s path that need to be re-enabled: `validateDomainContracts`, `validateStructureContracts`, `validateReaskContracts`, `validateProjectionContracts`, `validateKnowledgeBaseContracts`, `validateConversationContracts`, `validateDecisionContracts` (insofar as it applies to non-CRD-shape concerns), `validateModalityContracts`, `validateModelSelectionConfig`, `validateAdvancedToolFilteringConfig`. Per-family applicability should be confirmed during the unification design pass.

## Recipe for closing another slice (template established by the embedding slice)

The unification path described in Desired End State above is the preferred end state. Until that lands, slices can be closed incrementally with the mirror-per-slice pattern this debt entry's first slice established. To close another family validator's slice on the K8s reconcile path:

1. In `pkg/config/validator_<family>.go`, export a wrapper that mirrors the private validator 1:1: `ValidateXContracts(cfg *RouterConfig) error`. Match the `validate(cfg)` shape used by every other family validator in `validator.go`'s dispatch table so the K8s call site reads the same as the file-config dispatch.
2. Call it from `Reconciler.validateAndUpdate` in `pkg/k8s/reconciler.go` after `ParseYAMLBytes` returns and `ConfigSource = ConfigSourceKubernetes` is set, before `onConfigUpdate`.
3. On validation failure, mark pool and route `Ready=False, Reason=ValidationFailed` mirroring the existing pattern in `validateAndUpdate` and return a `fmt.Errorf %w`-wrapped error so callers can `errors.Unwrap` to retrieve the underlying validator error.
4. Add a reconciler-level integration test mirroring `pkg/k8s/reconciler_embedding_modality_test.go`. Use controller-runtime's `fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(...).WithObjects(pool, route).Build()` to drive a real `Reconciler.validateAndUpdate` call, then assert both the returned error AND the persisted Ready conditions on both refetched CRs. Verify the load-bearing property by removing the new validation block and confirming the rejection cases fail.
5. Update this entry's "Validators currently skipped" list above to remove the slice you closed.
6. When all listed validators are closed via this pattern (or the unification path lands), close TD040 per `tech-debt/README.md`'s retirement policy.

The mirror-per-slice path is acceptable for incremental closure but is not the desired end state. If a unification PR (selective early-return or per-validator opt-in) is feasible, that is preferred over N more mirror-PRs.
