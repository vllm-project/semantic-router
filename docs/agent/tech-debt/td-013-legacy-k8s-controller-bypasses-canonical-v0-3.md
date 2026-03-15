# TD013: Legacy IntelligentPool and IntelligentRoute Controller Bypasses Canonical v0.3 Config

## Status

Open

## Scope

legacy Kubernetes CRD controller path

## Summary

The primary router, CLI, dashboard, Helm, operator, and maintained config assets now converge on the canonical v0.3 `version/listeners/providers/routing/global` contract. One older Kubernetes path still bypasses that contract: the in-process `pkg/k8s` controller that watches `IntelligentPool` and `IntelligentRoute` CRDs and converts them directly into legacy `config.BackendModels` and `config.IntelligentRouting` runtime structs.

## Evidence

- [src/semantic-router/cmd/main.go](../../../src/semantic-router/cmd/main.go)
- [src/semantic-router/pkg/k8s/controller.go](../../../src/semantic-router/pkg/k8s/controller.go)
- [src/semantic-router/pkg/k8s/reconciler.go](../../../src/semantic-router/pkg/k8s/reconciler.go)
- [src/semantic-router/pkg/k8s/converter.go](../../../src/semantic-router/pkg/k8s/converter.go)
- [src/semantic-router/pkg/k8s/testdata/output/intelligent-pool-converted.yaml](../../../src/semantic-router/pkg/k8s/testdata/output/intelligent-pool-converted.yaml)
- [src/semantic-router/pkg/k8s/testdata/output/intelligent-route-converted.yaml](../../../src/semantic-router/pkg/k8s/testdata/output/intelligent-route-converted.yaml)

## Why It Matters

- This controller path is still active in the router binary, so it remains a real behavior surface instead of dead compatibility code.
- It reconstructs legacy runtime-only fields such as `default_model`, `model_config`, `vllm_endpoints`, and `categories`, which means one Kubernetes workflow still reasons about a different contract than the public v0.3 schema.
- Leaving the gap undocumented would make the closed status of the main rollout look more complete than the code actually is.

## Desired End State

- The legacy in-process CRD controller either emits canonical v0.3 config and normalizes through the same parser as the rest of the system, or it is explicitly retired from the active router runtime.
- Test fixtures and docs for this path no longer rely on legacy runtime YAML as the steady-state artifact.

## Exit Criteria

- `src/semantic-router/pkg/k8s/converter.go` no longer produces legacy `BackendModels` and `IntelligentRouting` as its public output shape.
- The active router startup path in `src/semantic-router/cmd/main.go` no longer needs a legacy-only config conversion seam for `IntelligentPool` / `IntelligentRoute`.
- `pkg/k8s` test fixtures either validate canonical config output or the legacy controller path is removed and its fixtures retired.
