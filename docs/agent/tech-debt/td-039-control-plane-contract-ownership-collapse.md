# TD039: Control Planes Still Depend on Router Runtime Internals Instead of Contract-Owned Seams

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

Router config/API packages, dashboard backend, Kubernetes operator, Python CLI,
and shared harness docs that define cross-stack dependency direction.

## Summary

The repo has a coherent product split: router kernel, CLI, dashboard, and
operator. The dependency direction is still too porous. Dashboard backend,
operator, and CLI surfaces continue to know too much about router runtime
implementation details instead of consuming narrow contracts or public service
seams. Some dashboard backend slices now use `routercontract`, but config,
deploy/setup, operator canonical translation, CLI runtime/config ownership, and
router bootstrap globals still keep the cross-stack boundary too broad.

## Evidence

- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/routercontract/tools.go](../../../dashboard/backend/routercontract/tools.go)
- [dashboard/backend/routercontract/listeners.go](../../../dashboard/backend/routercontract/listeners.go)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/deploy.go](../../../dashboard/backend/handlers/deploy.go)
- [dashboard/backend/handlers/security_policy.go](../../../dashboard/backend/handlers/security_policy.go)
- [deploy/operator/controllers/canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)
- [deploy/operator/controllers/backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/tech-debt/td-028-operator-config-contract-boundary-collapse.md](td-028-operator-config-contract-boundary-collapse.md)
- [docs/agent/tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md](td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)

## Why It Matters

- Contract changes become repo-wide synchronization work when control planes and
  runtime internals share implementation-level types.
- Public/private boundaries stay unclear, so new code can reach into router
  internals by default.
- Local subsystem cleanups can make files smaller while the cross-stack
  dependency direction remains wrong.

## Desired End State

- Router kernel internals stay inward-facing.
- CLI, dashboard backend, and operator consume versioned contracts or public
  runtime-service APIs when dependencies cross product boundaries.
- Runtime bootstrap and API startup use explicit construction and ownership
  rather than process-wide service publication.
- Local rules and shared harness docs point at the same contract-first target.

## Exit Criteria

- Dashboard backend config/deploy/setup paths no longer depend on router
  implementation packages where a contract or public service seam can own the
  dependency.
- Operator canonical translation does not require routine direct edits against
  router canonical implementation types.
- CLI runtime/config ownership aligns with current orchestration files and does
  not become the default owner for router kernel behavior.
- Runtime bootstrap materially reduces process-wide shared service globals.
