# Knowledge Base Seed and Steady-State Convergence Loop

## Goal

- Converge built-in and user-managed knowledge-base storage on one steady-state runtime root instead of keeping built-in seeds, managed assets, and runtime materialization as separate long-lived contracts.
- Make built-in seeds ship only from `/app/config/knowledge_bases/*`, while the router steady-state reads only from `/app/.vllm-sr/knowledge_bases/*`.
- Remove the legacy custom KB root, built-in source-path aliasing, and per-KB runtime source-path rewrite so built-in and user-created KBs are managed through the same persistent runtime directory model after bootstrap.

## Scope

- `docs/agent/plans/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/cmd/**`
- `src/vllm-sr/cli/**`
- `dashboard/backend/**`
- `dashboard/frontend/**`
- `config/**`
- `deploy/recipes/**`
- `tools/docker/**`
- targeted tests and repo-native validation for config, runtime, API, dashboard, and packaging surfaces
- nearest local rules for `pkg/config`, `vllm-sr/cli`, `dashboard/backend/handlers`, and `dashboard/frontend/src`

## Exit Criteria

- Built-in KB seeds ship only from `config/knowledge_bases/<seed-name>/` in repo sources and `/app/config/knowledge_bases/<seed-name>/` in runtime images.
- One steady-state runtime KB root exists at `.vllm-sr/knowledge_bases/<source-dir>/`, while canonical `source.path` values stay on `knowledge_bases/<source-dir>/`.
- Built-in seed bootstrap is explicit, idempotent, and runtime-owned; it no longer depends on `start-router.sh` copy logic, per-KB source-path rewrites to `<kb-name>`, or built-in source-path aliasing.
- Dashboard and router apiserver KB create, update, list, and delete flows operate directly on the steady-state root instead of any legacy custom KB root.
- Built-in and user-created KBs share one persistence model after bootstrap, including deletion behavior across restart.
- Canonical defaults, maintained recipes, Docker packaging, and docs all describe the same seed-vs-steady-state contract.
- Applicable repo-native validation is rerun until the active changed-set gates pass or a durable blocker is recorded.

## Task List

- [x] `KBS001` Create and index the durable execution plan for KB seed and steady-state convergence.
- [x] `KBS002` Inventory the current KB ownership seams across config defaults, runtime sync, apiserver CRUD, dashboard proxy flows, and image packaging, then lock the migration boundary for this workstream.
- [x] `KBS003` Define the exact steady-state KB contract: canonical config path shape, built-in seed bootstrap rules, and restart/delete persistence semantics without retaining a legacy custom-KB compatibility layer.
- [x] `KBS004` Introduce an explicit runtime-owned KB bootstrap or store seam that seeds built-ins into `.vllm-sr/knowledge_bases/*` without relying on shell-script-only copy behavior.
- [x] `KBS005` Move router config/runtime loading onto steady-state-only KB reads and remove per-KB runtime source-path rewrite, built-in source-path aliasing, and any remaining source-path canonicalization shims.
- [x] `KBS006` Move apiserver KB CRUD and persistence fully onto the steady-state root so built-in and custom KBs share one runtime store without legacy custom-KB handling.
- [x] `KBS007` Align dashboard backend and frontend KB-management flows with the steady-state store so the UI no longer assumes separate built-in and custom storage roots.
- [x] `KBS008` Update images, maintained recipes, docs, and validation/test surfaces for the new seed-vs-steady-state contract, including any maintained examples or harness docs that still mention the old storage model.
- [x] `KBS009` Run the repo-native validation ladder for the affected surfaces, capture the results here, and promote any unresolved architecture gap into indexed debt or ADR records only if the current loop cannot close it cleanly.

## Current Loop

- Loop status: completed on 2026-04-03; the storage contract is converged in code and the active validation loop passed.
- Current implementation snapshot:
  - runtime sync seeds or imports KB assets into `.vllm-sr/knowledge_bases/<source-dir>/`, preserves canonical `source.path: knowledge_bases/<source-dir>/`, and records one-time bootstrap state so deleted KB directories are not silently recreated
  - router config loading now prefers steady-state `.vllm-sr/knowledge_bases/*` when present, while source-config validation still falls back to bundled seed assets for built-ins
  - apiserver KB CRUD persists to the steady-state store and keeps built-in source-dir names stable on update without carrying a legacy custom-KB compatibility layer
- Next loop focus:
  - no further follow-up is required for this loop unless a new KB contract change reopens the storage model
- Validation results:
  - `python3 -m pytest src/vllm-sr/tests/test_runtime_support.py src/vllm-sr/tests/test_dashboard_dockerfile_surface.py -q` passed (`21 passed`)
  - `go test ./pkg/config ./pkg/apiserver` passed
  - `go test ./handlers -run 'TestSyncRuntimeConfigLocallySeedsKnowledgeBaseAssetsIntoRuntimeStore|TestSyncRuntimeConfigLocallyWritesInternalRuntimeConfig|TestSyncRuntimeConfigInManagedContainerUsesDashboardVenvPythonForSplitRuntime'` passed
  - `make agent-validate` passed

## Decision Log

- 2026-04-03: built-in KB directories under `/app/config/knowledge_bases/*` are seed inputs, not the steady-state runtime read path.
- 2026-04-03: after bootstrap, built-in and user-created KBs should be managed through one persistent runtime store so they no longer diverge by origin.
- 2026-04-03: the legacy custom KB root, built-in source-path aliases, and per-KB runtime source-path rewrite are transitional seams to retire rather than long-term public contracts.
- 2026-04-03: bootstrap and migration should be owned by an explicit runtime seam rather than by ad hoc entrypoint-shell behavior.
- 2026-04-03: the canonical public `source.path` contract remains `knowledge_bases/<source-dir>/`; local/dev runtime achieves steady-state-only reads by always running the router against a `.vllm-sr` runtime config whose base dir points at the shared KB store.
- 2026-04-03: built-in deletion persists across restart through per-path bootstrap state instead of unconditional reseeding.

## Follow-up Debt / ADR Links

- Predecessor KB context: [pl-0018-generic-embedding-kb-workstream.md](pl-0018-generic-embedding-kb-workstream.md)
- Related runtime ownership context: [pl-0027-router-runtime-composition-root-convergence-loop.md](pl-0027-router-runtime-composition-root-convergence-loop.md)
- [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [TD025 Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
