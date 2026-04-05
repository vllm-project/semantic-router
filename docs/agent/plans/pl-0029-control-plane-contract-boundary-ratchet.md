# Control-Plane Contract Boundary Ratchet Execution Plan

## Goal

- Turn the 2026-04 full-repository modular architecture audit into canonical, resumable governance artifacts instead of leaving the findings in chat-only form.
- Lock the repository onto one durable dependency direction: platform kernel inward, contract-first control planes outward.
- Finish a first complete loop that records the new ADR and debt, aligns shared harness docs, and corrects stale CLI hotspot guidance.

## Scope

- `docs/agent/adr/**`
- `docs/agent/tech-debt/**`
- `docs/agent/plans/**`
- `docs/agent/architecture-guardrails.md`
- `docs/agent/module-boundaries.md`
- `docs/agent/local-rules.md`
- `src/vllm-sr/cli/AGENTS.md`
- `tools/agent/repo-manifest.yaml`

## Exit Criteria

- ADR0006 and TD039 exist, are indexed, and reflect the audit's cross-stack findings.
- Shared harness docs state the contract-first control-plane boundary explicitly instead of leaving it implied across subsystem debts.
- CLI-local rules and harness docs point at the real runtime-orchestration hotspots (`core.py`, `commands/runtime.py`) and keep `docker_cli.py` as a thin compatibility seam.
- Repo-manifest inventories stay aligned with the new governance artifacts and the current CLI execution shape.
- `make agent-validate` and the changed-set `make agent-ci-gate CHANGED_FILES="..."` pass for the loop.

## Task List

- [x] `CPR001` Promote the audit findings into one durable ADR and one cross-stack technical debt entry.
- [x] `CPR002` Create and index this execution plan so future loops can resume from the repository alone.
- [x] `CPR003` Update shared harness docs with the contract-first control-plane boundary and the corrected CLI hotspot ownership.
- [x] `CPR004` Align local CLI rules and repo-manifest metadata with the current orchestration entrypoints.
- [x] `CPR005` Run harness validation and the changed-set CI gate, then record the follow-up workstreams that remain open after this governance loop.
- [x] `CPR006` Align overlapping boundary-ratchet execution plans with ADR0006 so resumed work no longer points at a missing dedicated ADR.
- [x] `CPR007` Align the shared repo map with the current CLI orchestration ownership so canonical entrypoint guidance no longer treats `docker_cli.py` as the main runtime wiring owner.

## Current Loop

- Loop opened on 2026-04-03 from a full-repository modular architecture audit covering router runtime, CLI, dashboard backend and frontend, operator, fleet-sim, native bindings, E2E, and the agent harness.
- Completed in this loop:
  - created ADR0006 to formalize the platform-kernel plus contract-first control-plane target
  - created TD039 to track the cross-stack boundary gap between router internals and the CLI, dashboard backend, and operator control planes
  - updated shared harness docs so the dependency rule and CLI hotspot ownership are discoverable from canonical entrypoints
  - corrected the CLI local rule drift so `core.py` and `commands/runtime.py` are treated as the active orchestration seams while `docker_cli.py` remains thin
  - aligned the repo manifest inventories and entrypoints with the new docs and the current CLI execution shape
  - validated the governance loop with `make agent-validate` and `make agent-ci-gate CHANGED_FILES="docs/agent/adr/adr-0006-platform-kernel-and-contract-first-control-planes.md,docs/agent/adr/README.md,docs/agent/architecture-guardrails.md,docs/agent/local-rules.md,docs/agent/module-boundaries.md,docs/agent/plans/pl-0029-control-plane-contract-boundary-ratchet.md,docs/agent/plans/README.md,docs/agent/tech-debt/td-039-control-plane-contract-ownership-collapse.md,docs/agent/tech-debt/README.md,src/vllm-sr/cli/AGENTS.md,tools/agent/repo-manifest.yaml"`
  - aligned the overlapping boundary-ratchet plans with ADR0006 so future resume loops no longer point at a missing dedicated ADR for this audit family
  - aligned `docs/agent/repo-map.md` with the current CLI execution shape so the main entrypoint map now points at `commands/runtime.py` and `core.py` before `docker_cli.py`
  - revalidated the expanded changed set after the plan-link and repo-map alignment with `make agent-validate` and `make agent-ci-gate CHANGED_FILES="docs/agent/adr/adr-0006-platform-kernel-and-contract-first-control-planes.md,docs/agent/adr/README.md,docs/agent/architecture-guardrails.md,docs/agent/local-rules.md,docs/agent/module-boundaries.md,docs/agent/plans/pl-0007-hotspot-boundary-ratchet.md,docs/agent/plans/pl-0008-dashboard-backend-and-go-config-boundary-ratchet.md,docs/agent/plans/pl-0009-fleet-sim-optimizer-and-operator-config-boundary-ratchet.md,docs/agent/plans/pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md,docs/agent/plans/pl-0029-control-plane-contract-boundary-ratchet.md,docs/agent/plans/README.md,docs/agent/repo-map.md,docs/agent/tech-debt/td-039-control-plane-contract-ownership-collapse.md,docs/agent/tech-debt/README.md,src/vllm-sr/cli/AGENTS.md,tools/agent/repo-manifest.yaml"`
- Completed on 2026-04-03: CPR001 through CPR007 are all complete, and this governance plan has no remaining open tasks.
- Follow-up workstreams remain open in the existing subsystem plans and debt items; this plan intentionally stops after the audit-capture and harness-alignment loop instead of pretending to complete the downstream code refactors in one pass.

## Decision Log

- 2026-04-03: use one cross-stack ADR instead of duplicating the same dependency-direction guidance into multiple subsystem plans.
- 2026-04-03: open one umbrella debt entry for cross-control-plane contract ownership while keeping subsystem-specific debts focused on their local implementation collapse.
- 2026-04-03: treat `src/vllm-sr/cli/core.py` and `src/vllm-sr/cli/commands/runtime.py` as the current CLI orchestration hotspots; keep `docker_cli.py` as a compatibility seam, not the default runtime-flow owner.
- 2026-04-03: scope this execution plan to audit capture and harness alignment so the full loop can complete in one change; follow-on refactors continue in existing implementation plans.

## Follow-up Debt / ADR Links

- [../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md](../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md)
- [../tech-debt/td-039-control-plane-contract-ownership-collapse.md](../tech-debt/td-039-control-plane-contract-ownership-collapse.md)
- [../tech-debt/td-028-operator-config-contract-boundary-collapse.md](../tech-debt/td-028-operator-config-contract-boundary-collapse.md)
- [../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [pl-0009-fleet-sim-optimizer-and-operator-config-boundary-ratchet.md](pl-0009-fleet-sim-optimizer-and-operator-config-boundary-ratchet.md)
- [pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md](pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md)
