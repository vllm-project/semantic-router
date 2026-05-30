# TD040: Kubernetes Reconcile Path Skips File-Config Family Validators

## Status

Resolved

## Scope

`pkg/config` validator dispatch and `pkg/k8s` reconciler integration

## Summary

Resolved on 2026-05-24. The initial Kubernetes static-config parse still
short-circuits because `decisions` and `model_config` are loaded from CRDs, but
the reconciler no longer mirrors individual validators one by one. After
`IntelligentPool` and `IntelligentRoute` are converted into canonical runtime
config, `Reconciler.validateAndUpdate` now calls the single exported
`config.ValidateKubernetesConfigContracts` seam.

That seam shares the same `sharedConfigContractValidators` dispatch table used
by file-config validation, so every currently shared family validator runs for
both file-loaded config and CRD-loaded config after conversion.

## Evidence

- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go): `sharedConfigContractValidators` is now the single dispatch table for file and K8s post-conversion validation; `ValidateKubernetesConfigContracts` exposes the K8s-safe dispatch seam.
- [src/semantic-router/pkg/k8s/reconciler.go](../../../src/semantic-router/pkg/k8s/reconciler.go): `validateAndUpdate` calls `config.ValidateKubernetesConfigContracts` after CRD conversion instead of calling a single family validator.
- [src/semantic-router/pkg/config/validator_test.go](../../../src/semantic-router/pkg/config/validator_test.go): covers the shared dispatch table and verifies K8s post-conversion validation rejects a decision contract that initial K8s static parsing intentionally skips.
- [src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go](../../../src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go): still covers the original embedding-modality bug and now adds a second reconciler-level dispatch canary for `tools.advanced_filtering`.

## Why It Matters

- The K8s reconcile path is the operator-mode entrypoint. Every CRD that contradicts a runtime contract the validator can detect should be marked `Ready=False` with a clear message, not admitted silently.
- The validators now share a dispatch surface, so new family validators must explicitly declare their file and K8s scopes instead of silently drifting away from reconcile validation.
- The prior one-off mirroring pattern was retired from `Reconciler.validateAndUpdate`; `ValidateEmbeddingContracts` remains only as a narrow exported helper for callers that need the embedding slice by itself.

## Desired End State

Achieved with a scoped dispatch table:

- `validateConfigStructure` continues to tolerate the initial Kubernetes static-config parse.
- File config uses `validateConfigContracts(..., configValidationScopeFile)`.
- K8s reconcile uses `ValidateKubernetesConfigContracts` after CRD conversion, which calls `validateConfigContracts(..., configValidationScopeKubernetes)`.
- Every validator in the current shared table is wired for both scopes and covered by a table-wiring regression test.

## Exit Criteria

- [x] The K8s reconcile path runs the same shared family-validator surface that file-config does after CRD conversion.
- [x] `Reconciler.validateAndUpdate` no longer calls individual exported validators for routine reconcile validation.
- [x] Dispatch wiring is covered in `pkg/config`, and the reconciler path has load-bearing status/error tests for the original embedding-modality case plus a non-embedding dispatch canary.
- [x] The previously skipped validators are re-enabled through the shared K8s dispatch: `validateDomainContracts`, `validateStructureContracts`, `validateReaskContracts`, `validateProjectionContracts`, `validateKnowledgeBaseContracts`, `validateConversationContracts`, `validateDecisionContracts`, `validateEmbeddingContracts`, `validateModalityContracts`, `validateModelSelectionConfig`, and `validateAdvancedToolFilteringConfig`.

## Validation

- `go test ./pkg/config ./pkg/k8s`
- `make test-semantic-router`
