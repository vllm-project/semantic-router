# TD018: Skill Surface Taxonomy Has Drifted Away from Active Module Boundaries

## Status

Closed

## Scope

agent skill registry, change-surface catalog, and primary-skill routing coverage for active repository subsystems

## Summary

The current skill taxonomy is still centered on older surface groupings such as classifier post-training, plugin runtime, dashboard config UI, and operator-only Kubernetes changes. The repository has since grown several active modules and deployment surfaces that have dedicated code, local rules, and E2E coverage, but no dedicated primary skill or surface ownership in the harness. As a result, important changes under active paths such as memory, response API, authz, apiserver, fleet-sim, and non-CRD deployment manifests routinely fall back to `cross-stack-bugfix`, while some existing skill names no longer describe the code they actually cover. Active execution is tracked in [../plans/pl-0005-skill-taxonomy-refresh.md](../plans/pl-0005-skill-taxonomy-refresh.md).

## Evidence

- [tools/agent/skill-registry.yaml](../../../tools/agent/skill-registry.yaml)
- [docs/agent/change-surfaces.md](../../../docs/agent/change-surfaces.md)
- [tools/agent/repo-manifest.yaml](../../../tools/agent/repo-manifest.yaml)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [src/fleet-sim/AGENTS.md](../../../src/fleet-sim/AGENTS.md)
- [src/semantic-router/pkg/apiserver/route_memory.go](../../../src/semantic-router/pkg/apiserver/route_memory.go)
- [src/semantic-router/pkg/authz/chain.go](../../../src/semantic-router/pkg/authz/chain.go)
- [src/semantic-router/pkg/memory/extractor.go](../../../src/semantic-router/pkg/memory/extractor.go)
- [src/semantic-router/pkg/responseapi/store.go](../../../src/semantic-router/pkg/responseapi/store.go)
- [deploy/kubernetes/response-api/redis.yaml](../../../deploy/kubernetes/response-api/redis.yaml)

## Why It Matters

- `make agent-report` produces a generic fallback skill for several high-traffic subsystems that now have distinct code and test contracts.
- The fallback context pack pulls in broad fragments that are unrelated to the changed path, which raises noise and hides the real validation intent.
- Existing skill names such as classifier-focused training or plugin-runtime imply scopes that no longer match the latest module layout, which makes the catalog harder to trust.
- Dedicated E2E and workflow coverage already exists for some of these areas, so the missing skill layer creates an avoidable mismatch between executable validation and the human-readable harness.

## Desired End State

- Active subsystems with distinct contracts have dedicated primary-skill routing or an intentional shared surface with current naming.
- Surface descriptions and typical paths in [docs/agent/change-surfaces.md](../../../docs/agent/change-surfaces.md) match the current package layout instead of legacy path examples.
- Outdated skill names are either retired or renamed so the skill catalog reflects current module responsibilities.
- Routing validation fixtures cover representative paths for the newer subsystems instead of only the older core surfaces.

## Exit Criteria

- Satisfied on 2026-03-18: `tools/agent/skill-registry.yaml` no longer routes representative changes under `src/fleet-sim/**`, `src/semantic-router/pkg/memory/**`, `src/semantic-router/pkg/responseapi/**`, `src/semantic-router/pkg/authz/**`, `src/semantic-router/pkg/apiserver/**`, and `deploy/kubernetes/response-api/**` to `cross-stack-bugfix` by default.
- Satisfied on 2026-03-18: the harness now defines dedicated or intentionally shared surfaces for the latest active modules and deployment areas, including router service-platform code, fleet-sim, deployment profiles, and the broader training stack.
- Satisfied on 2026-03-18: `docs/agent/change-surfaces.md` no longer describes `plugin_runtime` using stale `src/semantic-router/pkg/plugins/**` path examples.
- Satisfied on 2026-03-18: the training skill family no longer describes the whole `src/training/**` tree as classifier-only work when model-selection, embeddings, evaluation, and experiment subtrees are active.
- Satisfied on 2026-03-18: routing validation fixtures fail if representative subsystem paths regress back to fallback-only routing.

## Resolution

- `tools/agent/skill-registry.yaml` now adds first-class routing for router service-platform modules, fleet-sim, and non-operator deployment profiles, and renames the training skill family around the actual `src/training/**` scope.
- `docs/agent/change-surfaces.md`, `tools/agent/context-map.yaml`, and `docs/agent/skill-catalog.md` now describe the current surfaces and names instead of the older classifier-only and stale plugin-path framing.
- `tools/agent/routing-fixtures.yaml` now covers representative apiserver, memory, authz, fleet-sim, deployment-profile, and training-stack paths so the new routing model is mechanically enforced.
- The old classifier-only skill files were retired in favor of `training-stack-change` and `training-stack-runtime`.
