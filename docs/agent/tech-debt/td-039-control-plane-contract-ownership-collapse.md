# TD039: Control Planes Still Depend on Router Runtime Internals Instead of Contract-Owned Seams

## Status

Closed

## Scope

`src/semantic-router/pkg/config/**`, `src/semantic-router/pkg/apiserver/**`, `dashboard/backend/**`, `deploy/operator/**`, `src/vllm-sr/cli/**`, and the shared harness docs that define their cross-stack boundary rules

## Summary

The repository now exposes explicit control-plane seams for router config and authoring flows. `src/semantic-router/pkg/routercontract/` re-exports the normalized canonical config contract and routing-domain helpers used by dashboard backend and operator code, while `src/semantic-router/pkg/routerauthoring/` exposes the DSL validation, parse, compile, format, and decompile helpers used by dashboard authoring paths. Dashboard backend and operator controllers no longer import `pkg/config` or `pkg/dsl` directly, `pkg/routercontract/control_plane_import_test.go` now ratchets that dependency direction, and the dashboard backend Docker build copies the new seam packages into packaged builds.

This umbrella debt is now closed because the remaining cross-stack control-plane gap is gone, while the earlier local debts that fed into it were already retired:

- TD022 closed the CLI-side config-contract collapse
- TD028 closed the operator-internal contract-ownership collapse
- TD031 closed the runtime bootstrap and global-state composition gap
- TD003 closed the remaining CLI package-topology umbrella around local runtime orchestration

## Evidence

- [src/semantic-router/pkg/routercontract/config.go](../../../src/semantic-router/pkg/routercontract/config.go)
- [src/semantic-router/pkg/routercontract/control_plane_import_test.go](../../../src/semantic-router/pkg/routercontract/control_plane_import_test.go)
- [src/semantic-router/pkg/routerauthoring/dsl.go](../../../src/semantic-router/pkg/routerauthoring/dsl.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/canonical_transport.go](../../../dashboard/backend/handlers/canonical_transport.go)
- [dashboard/backend/handlers/builder_nl_validation.go](../../../dashboard/backend/handlers/builder_nl_validation.go)
- [dashboard/backend/handlers/setup.go](../../../dashboard/backend/handlers/setup.go)
- [dashboard/backend/handlers/topology_response.go](../../../dashboard/backend/handlers/topology_response.go)
- [dashboard/backend/handlers/deploy.go](../../../dashboard/backend/handlers/deploy.go)
- [dashboard/backend/handlers/deploy_test.go](../../../dashboard/backend/handlers/deploy_test.go)
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
- [deploy/operator/controllers/backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
- [dashboard/backend/Dockerfile](../../../dashboard/backend/Dockerfile)
- [src/vllm-sr/tests/test_dashboard_dockerfile_surface.py](../../../src/vllm-sr/tests/test_dashboard_dockerfile_surface.py)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [dashboard/backend/handlers/AGENTS.md](../../../dashboard/backend/handlers/AGENTS.md)
- [deploy/operator/controllers/AGENTS.md](../../../deploy/operator/controllers/AGENTS.md)
- [src/semantic-router/pkg/config/AGENTS.md](../../../src/semantic-router/pkg/config/AGENTS.md)
- [docs/agent/adr/adr-0006-platform-kernel-and-contract-first-control-planes.md](../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md)
- [docs/agent/tech-debt/td-003-package-topology-hotspot-layout-debt.md](td-003-package-topology-hotspot-layout-debt.md)
- [docs/agent/tech-debt/td-022-cli-config-contract-boundary-collapse.md](td-022-cli-config-contract-boundary-collapse.md)
- [docs/agent/tech-debt/td-028-operator-config-contract-boundary-collapse.md](td-028-operator-config-contract-boundary-collapse.md)
- [docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)

## Why It Matters

- Stable kernel seams are forced to evolve with change-heavy edge modules because control planes and runtime internals share the same implementation-level contracts.
- Canonical config evolution becomes a repo-wide synchronization task across runtime, CLI, dashboard, and operator instead of a narrow contract change.
- Public versus private boundaries remain unclear. Other modules can reach into router internals because the repo does not present one explicit contract-first path.
- Onboarding and refactor cost both rise because a contributor has to understand cross-product runtime internals to make what should be a control-plane-only change.
- Existing subsystem debt can be reduced locally while the cross-stack dependency problem still survives. Without an umbrella debt item, the repo can appear healthier than it is.

## Desired End State

- The repo has an explicit contract-first seam between the router kernel and each control plane.
- Dashboard backend, operator, and CLI consume versioned contracts or public runtime-service APIs instead of deep router-runtime implementation packages whenever the dependency crosses a product boundary.
- Runtime bootstrap and API-server steady-state wiring rely on explicit construction and ownership rather than package-global service publication.
- Local hotspot rules and shared harness docs reflect the real orchestration files and the intended dependency direction.
- Subsystem-specific debt remains focused on local implementation collapse, while this entry tracks the cross-stack boundary until the dependency direction itself is healthier.

## Exit Criteria

- Satisfied on 2026-04-06: dashboard backend config, deploy, topology, and authoring flows now consume `pkg/routercontract` and `pkg/routerauthoring`, and `pkg/routercontract/control_plane_import_test.go` blocks new direct `pkg/config` or `pkg/dsl` imports from dashboard/backend and operator/controller packages.
- Satisfied on 2026-04-06: operator controller-side canonical translation and backend discovery now depend on `pkg/routercontract` aliases instead of importing router canonical config implementation types directly.
- Satisfied on 2026-04-06: shared harness docs and nearest local rules now point cross-stack control-plane work at `pkg/routercontract` and `pkg/routerauthoring` as the contract-first seams.
- Satisfied before umbrella closure and carried forward on 2026-04-06: CLI contract ownership and local runtime-orchestration hotspots were already ratcheted under TD022 and TD003.
- Satisfied before umbrella closure and carried forward on 2026-04-06: runtime bootstrap and API-server global-state cleanup were already ratcheted under TD031.

## Resolution

- Added `src/semantic-router/pkg/routercontract/` as the public router-config seam for control-plane consumers, with canonical config aliases, parsing helpers, routing-domain helpers, and an import guard test that blocks new direct `pkg/config` and `pkg/dsl` imports from dashboard backend and operator controller packages.
- Added `src/semantic-router/pkg/routerauthoring/` as the public DSL seam for control-plane authoring flows, with validate/parse/compile/format/decompile helpers that return the canonical public contract instead of runtime-internal config types.
- Repointed dashboard backend config, deploy, setup, topology, transport, and builder-NL code plus the affected tests onto those public seams, and updated the dashboard backend Dockerfile so packaged builds copy the same seam packages used in local development.
- Repointed operator controller canonical translation and backend discovery code onto `pkg/routercontract`, then updated shared module-boundary docs and nearest local `AGENTS.md` files so the dependency direction is part of the durable harness guidance instead of only living in code.

## Validation

- `go test ./pkg/routercontract ./pkg/routerauthoring`
  Run in `/Users/bitliu/vs/src/semantic-router`
- `go test -run 'Test(UpdateConfigHandler|GlobalConfigYAMLHandler_ReturnsEffectiveGlobalConfig|GlobalConfigYAMLHandler_ReturnsDefaultsWhenGlobalMissing|MergeDeployPayload_RoundTripsMaintainedCanonicalConfig|BuildEvaluatedRule_EmptyConditionsSerializeAsEmptyArray|BuildBuilderNLTaskContextIncludesSharedHints)$'`
  Run in `/Users/bitliu/vs/dashboard/backend/handlers`
- `go test ./controllers`
  Run in `/Users/bitliu/vs/deploy/operator`
- `pytest /Users/bitliu/vs/src/vllm-sr/tests/test_dashboard_dockerfile_surface.py`
