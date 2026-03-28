# Dashboard Backend and Go Config Boundary Ratchet Execution Plan

## Goal

- Turn the latest repository design audit into executable, resumable cleanup work for the dashboard backend runtime-control slice and the Go router config contract.
- Reduce responsibility collapse in dashboard backend config/deploy/status handlers and in the router config package hotspots.
- Retire TD025 and TD026 only after the code, local rules, and harness evidence converge on narrower seams.

## Scope

- `dashboard/backend/handlers/**`
- `dashboard/backend/router/{router.go,core_routes.go,proxy_routes.go}`
- `src/semantic-router/pkg/config/**`
- `docs/agent/architecture-guardrails.md`
- `docs/agent/module-boundaries.md`
- local `AGENTS.md` files for the touched hotspot trees

## Exit Criteria

- TD025 and TD026 are both closed with concrete code and validation evidence.
- Dashboard backend handler changes no longer collapse HTTP transport, config persistence, deploy/rollback control, and runtime status collection into the same hotspot seams.
- Router config contract changes have clearer ownership boundaries between schema families, canonical conversion/export, plugin-family contracts, and semantic validation.
- Shared and local AGENT rules stay aligned with the active hotspot boundaries so future work inherits the narrower design target.

## Task List

- [x] `S001` Record the newly-audited dashboard backend and Go config hotspot debts in canonical TD entries and create this execution plan.
- [x] `S002` Tighten shared and local AGENT rules so dashboard backend handlers and Go config hotspots are explicit before code extraction starts.
- [x] `S003` Extract dashboard backend config/deploy seams so HTTP handlers delegate config persistence, backup/version inventory, and runtime propagation to narrower support modules.
- [x] `S004` Split dashboard backend status collection so mode detection, container or supervisor probing, log parsing, and router-runtime synthesis stop sharing one handler seam.
- [x] `S005` Split Go config contract ownership so schema families, canonical normalization/export, plugin-family contracts, and semantic validation stop collapsing into `config.go`, `validator.go`, and `rag_plugin.go`.
- [x] `S006` Run the required harness and package validation for the remaining work and close TD025 and TD026 when the slice boundaries catch up.

## Current Loop

- 2026-03-19: the latest repository design audit identified new boundary debt in dashboard backend runtime-control handlers and the Go router config contract.
- 2026-03-19: TD025 and TD026 were added so those hotspots are no longer only implicit structural debt.
- 2026-03-19: shared and local AGENT rules were updated so future edits discover the new backend handler hotspot tree and the active Go config contract hotspots.
- 2026-03-19: `S003` extracted runtime write/apply helpers and backup/version inventory into dedicated dashboard backend support modules, so `config.go` and `deploy.go` now delegate more of their persistence and runtime side effects.
- 2026-03-19: `S004` split dashboard status handling into dedicated probe and collector support files, leaving `status.go` as the transport shell and `status_modes.go` as the mode-dispatch seam.
- 2026-03-19: `S005` split config validation across `validator.go`, `validator_decision.go`, `validator_modality.go`, and `validator_tool_filtering.go`, and split the RAG plugin family into `rag_plugin.go`, `rag_plugin_backends.go`, `rag_plugin_validation.go`, and `rag_plugin_test.go`, so `validator.go` and `rag_plugin.go` now keep thinner entrypoint ownership.
- 2026-03-19: package and harness validation passed for the config refactor: `go test ./pkg/config/...`, `go test ./pkg/extproc/...`, `make agent-validate`, `make agent-lint CHANGED_FILES="..."`, `make test-semantic-router`, and `make agent-ci-gate CHANGED_FILES="..."`.
- 2026-03-19: CPU-local smoke passed on an isolated local stack via `make agent-serve-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`, `make agent-smoke-local AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`, and `make agent-stop-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`; container, router, envoy, dashboard, and fleet-sim all reported healthy.
- Completed on 2026-03-19: `S006` closed the final runtime evidence gap, so TD025 and TD026 were retired in the same loop as the earlier package and harness validation.

## Decision Log

- Prefer repo-local AGENT rules and execution plans over a new repo-local skill when the guidance is tightly coupled to one subtree's ownership and hotspots.
- Treat TD006 as a structural ratchet, not as a substitute for subsystem-specific debt when the repo needs a clearer architecture target for active modules.
- Keep dashboard backend config/deploy extraction and status-collector extraction as separate tasks so the work can progress incrementally without reopening all handlers at once.
- Keep the Go config contract cleanup in the same plan because schema, canonical conversion, validation, and plugin-family boundaries share the same package-level ownership problem.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
- [../tech-debt/td-026-go-config-contract-boundary-collapse.md](../tech-debt/td-026-go-config-contract-boundary-collapse.md)
- [../adr/README.md](../adr/README.md) (no dedicated ADR yet)
