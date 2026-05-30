# TD039: Control Planes Still Depend on Router Runtime Internals Instead of Contract-Owned Seams

## Status

Open

## Scope

`src/semantic-router/pkg/config/**`, `src/semantic-router/pkg/apiserver/**`, `dashboard/backend/**`, `deploy/operator/**`, `src/vllm-sr/cli/**`, and the shared harness docs that define their cross-stack boundary rules

## Summary

The repository's top-level product split is coherent, but the cross-stack dependency direction is still too porous. Dashboard backend, operator, and CLI surfaces continue to co-own router-runtime contract knowledge instead of consuming a narrow contract or public-service seam. Dashboard backend imports router config and authoring internals directly. The operator's controller-side canonical translation depends directly on router canonical config types. The CLI still carries a large share of canonical config knowledge while its local hotspot rules had drifted behind the real runtime orchestration files. At the same time, router runtime bootstrap and API-server startup still publish shared dependencies through process-wide globals.

Current-source recheck on 2026-05-24 found this debt still valid, but not all
original evidence has the same weight. Two Dashboard backend slices were narrowed:
`/api/tools-db` path resolution now goes through
`dashboard/backend/routercontract.ReadToolSelection` instead of importing router
config directly from `dashboard/backend/router/core_routes.go`, and OpenClaw
model-gateway discovery now goes through
`dashboard/backend/routercontract.ReadFirstListenerEndpoint` instead of parsing
router listener YAML inside `dashboard/backend/handlers/openclaw.go`. A
2026-05-25 Dashboard security-policy slice also narrowed a live apply-time
validation gap: `applySecurityFragment` now validates merged generated config
through `dashboard/backend/handlers/security_policy_config_validation.go` before
the config is written or applied to runtime, with
`dashboard/backend/handlers/security_policy_apply.go` owning the write/hot-reload
application path so the transport handler no longer grows past the structure
warn threshold. The debt remains open because
Dashboard config/deploy/setup handlers, Operator canonical
translation, CLI runtime/config ownership, and router bootstrap globals still
carry the broader cross-stack ownership problem.

Validation for the OpenClaw listener adapter slice passed with
`go test ./handlers ./routercontract -run 'TestResolveOpenClawModelBaseURL|TestDefaultOpenClawModelBaseURL|TestReadFirstListenerEndpoint|TestOpenClawModelGatewayContainerName'`
from `dashboard/backend`, followed by changed-set `make agent-ci-gate
CHANGED_FILES="..."`. Feature integration and local smoke remain blocked by the
desktop Docker socket, not by this code path.

This is broader than the existing subsystem debts:

- TD022 focused on CLI-internal config ownership collapse
- TD028 focused on operator-internal config ownership collapse
- TD031 focused on router bootstrap and global-state runtime composition
- TD030 focused on dashboard frontend boundary collapse

Those entries remain valid, but they do not by themselves define the cross-stack problem: the repo still lacks a durable contract-owned boundary between the router kernel and its control planes.

## Evidence

- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/security_policy.go](../../../dashboard/backend/handlers/security_policy.go)
- [dashboard/backend/handlers/security_policy_apply.go](../../../dashboard/backend/handlers/security_policy_apply.go)
- [dashboard/backend/handlers/security_policy_config_validation.go](../../../dashboard/backend/handlers/security_policy_config_validation.go)
- [dashboard/backend/routercontract/tools.go](../../../dashboard/backend/routercontract/tools.go)
- [dashboard/backend/routercontract/listeners.go](../../../dashboard/backend/routercontract/listeners.go)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/canonical_transport.go](../../../dashboard/backend/handlers/canonical_transport.go)
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
- [deploy/operator/controllers/backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/validator.py](../../../src/vllm-sr/cli/validator.py)
- [src/vllm-sr/cli/AGENTS.md](../../../src/vllm-sr/cli/AGENTS.md)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/adr/adr-0006-platform-kernel-and-contract-first-control-planes.md](../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md)
- [docs/agent/tech-debt/td-022-cli-config-contract-boundary-collapse.md](td-022-cli-config-contract-boundary-collapse.md)
- [docs/agent/tech-debt/td-028-operator-config-contract-boundary-collapse.md](td-028-operator-config-contract-boundary-collapse.md)
- [docs/agent/tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md](td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
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

- Dashboard backend no longer depends on router-runtime implementation packages for config authoring, deploy/apply control, or transport shaping when a contract or public service seam can own that dependency.
- Operator controller-side canonical translation no longer depends directly on router canonical config implementation types for routine feature evolution.
- CLI contract ownership and runtime-orchestration hotspots are aligned with the current code shape, and new control-plane work does not default to deep router-runtime imports.
- Runtime bootstrap and API-server startup paths materially reduce their reliance on process-wide shared service globals.
- Shared harness docs, local rules, and follow-up execution plans all point at the same contract-first control-plane target rather than separate local interpretations.
