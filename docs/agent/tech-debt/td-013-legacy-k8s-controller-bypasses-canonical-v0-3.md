# TD013: Legacy IntelligentPool and IntelligentRoute Controller Bypasses Canonical v0.3 Config

## Status

Closed

## Scope

legacy Kubernetes CRD controller path

## Summary

The primary router, CLI, dashboard, Helm, operator, and maintained config assets now converge on the canonical v0.3 `version/listeners/providers/routing/global` contract. The last older Kubernetes path that previously bypassed that contract, the in-process `pkg/k8s` controller watching `IntelligentPool` and `IntelligentRoute`, now emits canonical v0.3 config and normalizes it through the same parser as the rest of the system.

## Evidence

- [src/semantic-router/cmd/main.go](../../../src/semantic-router/cmd/main.go)
- [src/semantic-router/pkg/k8s/controller.go](../../../src/semantic-router/pkg/k8s/controller.go)
- [src/semantic-router/pkg/k8s/reconciler.go](../../../src/semantic-router/pkg/k8s/reconciler.go)
- [src/semantic-router/pkg/k8s/converter.go](../../../src/semantic-router/pkg/k8s/converter.go)
- [src/semantic-router/pkg/config/canonical_export.go](../../../src/semantic-router/pkg/config/canonical_export.go)
- [src/semantic-router/pkg/config/canonical_global.go](../../../src/semantic-router/pkg/config/canonical_global.go)
- [src/semantic-router/pkg/k8s/testdata/base-config.yaml](../../../src/semantic-router/pkg/k8s/testdata/base-config.yaml)
- [src/semantic-router/pkg/k8s/testdata/output/01-basic.yaml](../../../src/semantic-router/pkg/k8s/testdata/output/01-basic.yaml)
- [src/semantic-router/pkg/k8s/testdata/output/04-domain-only.yaml](../../../src/semantic-router/pkg/k8s/testdata/output/04-domain-only.yaml)

## Why It Matters

- This controller path is still active in the router binary, so it had to converge on the same parser and public contract instead of silently keeping a second runtime-only shape alive.
- The contract now has one documented switch for that path, `global.router.config_source: kubernetes`, instead of an implicit legacy seam.

## Desired End State

- The in-process CRD controller emits canonical v0.3 config and normalizes through the same parser as the rest of the system.
- Test fixtures and docs for this path no longer rely on legacy runtime YAML as the steady-state artifact.

## Exit Criteria

- `src/semantic-router/pkg/k8s/converter.go` no longer produces legacy `BackendModels` and `IntelligentRouting` as its public output shape.
- The active router startup path in `src/semantic-router/cmd/main.go` re-enters canonical config normalization for `IntelligentPool` / `IntelligentRoute` instead of merging legacy runtime structs directly.
- `pkg/k8s` test fixtures validate canonical config output, and the enablement path is documented as `global.router.config_source: kubernetes`.

## Retirement Notes

- `pkg/k8s` now converts `IntelligentPool` and `IntelligentRoute` into canonical `version/listeners/providers/routing/global` config and then normalizes through `config.ParseYAMLBytes`.
- `config.CanonicalStaticConfigFromRouterConfig` provides the static canonical base for this path, so the reconciler no longer merges CRD output directly into legacy runtime-only structs.
- `global.router.config_source` is the public contract switch for Kubernetes-backed reconciliation, with `file` remaining the default for YAML-first workflows.
- `pkg/k8s/testdata/base-config.yaml` and generated output fixtures now use canonical v0.3 YAML instead of legacy runtime layout.
