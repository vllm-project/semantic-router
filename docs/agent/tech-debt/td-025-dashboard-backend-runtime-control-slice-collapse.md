# TD025: Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection

## Status

Closed

## Scope

`dashboard/backend/handlers/{config.go,deploy.go,status.go,status_modes.go}` and adjacent dashboard backend route wiring

## Summary

The dashboard backend now owns config editing, deploy/rollback, and runtime status flows, but the control slice had been concentrated in a few handler hotspots. `config.go` mixed HTTP transport, canonical config read/write, endpoint validation, rollback handling, and runtime propagation. `deploy.go` mixed deploy/preview/rollback endpoints with merge logic, backup/version management, write/apply semantics, and response shaping. `status.go` and `status_modes.go` combined handler transport, Docker or supervisor probing, log parsing, router-runtime synthesis, and model readiness fetches. Existing debt already tracked broader dashboard foundations and weak typing, but it did not describe this backend runtime-control boundary collapse explicitly.

`runtime_config_apply.go`, `config_backups.go`, `status_probes.go`, and `status_collectors.go` now absorb the extracted persistence and status logic, focused handler tests and harness validation are green, and the CPU-local smoke path now confirms the same narrower seams in the local runtime stack.

## Evidence

- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/deploy.go](../../../dashboard/backend/handlers/deploy.go)
- [dashboard/backend/handlers/status.go](../../../dashboard/backend/handlers/status.go)
- [dashboard/backend/handlers/status_modes.go](../../../dashboard/backend/handlers/status_modes.go)
- [dashboard/backend/router/router.go](../../../dashboard/backend/router/router.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/router/proxy_routes.go](../../../dashboard/backend/router/proxy_routes.go)
- [dashboard/backend/handlers/AGENTS.md](../../../dashboard/backend/handlers/AGENTS.md)
- [docs/agent/tech-debt/td-005-dashboard-enterprise-console-foundations.md](td-005-dashboard-enterprise-console-foundations.md)
- [docs/agent/tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md](td-015-weakly-typed-config-and-dsl-contracts.md)
- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)

## Why It Matters

- Narrow backend behavior changes reopen the same handler files even when the work only targets config persistence, deploy control, or runtime status.
- HTTP transport code is harder to test independently when it shares ownership with file writes, runtime apply semantics, and host/container probing.
- Status collection logic is coupled to response shaping and deployment-mode dispatch, which makes regressions harder to isolate.
- The repo previously lacked a nearest local `AGENTS.md` for this backend hotspot tree, so extraction expectations were implicit instead of local and explicit.

## Desired End State

- Dashboard backend handlers own HTTP method guards, request decoding, response encoding, and delegation only.
- Config edit, deploy/rollback, backup/version inventory, and runtime propagation move behind narrower support modules or services.
- Runtime status mode selection, container or supervisor probing, log parsing, router-runtime synthesis, and model readiness fetches live behind focused status collectors.
- Dashboard backend local rules explicitly name the config/deploy/status hotspots and the extraction-first boundaries expected when they are touched.

## Exit Criteria

- Config edit, deploy/rollback, and runtime status work no longer requires reopening the same handler hotspots for unrelated responsibilities.
- `config.go` and `deploy.go` delegate file mutation, backup/version, and runtime-apply behavior to narrower helpers or services.
- `status.go` and `status_modes.go` no longer combine transport, mode selection, probing, and status synthesis in the same primary seam.
- Dashboard backend local `AGENTS.md` and harness docs stay aligned with the active hotspot boundaries.

## Resolution

- Dashboard backend config/deploy handlers now delegate persistence, backup/version inventory, and runtime apply behavior through `runtime_config_apply.go` and `config_backups.go` instead of continuing to own those side effects inline.
- Status handling now delegates probing and synthesis through `status_probes.go` and `status_collectors.go`, leaving `status.go` as the transport shell and `status_modes.go` as the mode-dispatch seam.
- The nearest local rule file at `dashboard/backend/handlers/AGENTS.md` and the shared architecture docs now explicitly call out this backend hotspot boundary so future edits inherit the extraction-first path.

## Validation

- `go test ./handlers -run 'Test(UpdateConfigHandler|UpdateRouterDefaultsHandler|DeployHandler_(SuccessfulDeploy|DeepMergePreservesExistingFields|UsesImportedCanonicalBaseConfig|NoDSLSource)|RollbackHandler_|ConfigVersionsHandler_|CleanupBackups|DeployAndRollback_Integration|DetectRouterRuntimeStatus|ConfiguredRuntimeConfigPathUsesEnvOverride|SyncRuntimeConfigLocallyWritesInternalRuntimeConfig|RuntimeSyncPythonBinaryRejectsNonPythonOverride)'` from `dashboard/backend`
- `make agent-validate`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-serve-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
- `make agent-smoke-local AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
- `make agent-stop-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
