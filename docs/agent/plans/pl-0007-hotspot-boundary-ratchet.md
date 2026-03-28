# Hotspot Boundary Ratchet Execution Plan

## Goal

- Turn the latest cross-module design audit into executable, resumable boundary cleanup work.
- Reduce responsibility collapse in the CLI config contract, extproc request pipeline, and OpenClaw feature slice.
- Retire TD022, TD023, and TD024 only after the code and local rules converge on narrower seams.

## Scope

- `src/vllm-sr/cli/**`
- `src/semantic-router/pkg/extproc/**`
- `dashboard/frontend/src/pages/**`
- `dashboard/frontend/src/components/**`
- `dashboard/backend/{handlers,router}/openclaw*.go`
- `docs/agent/architecture-guardrails.md`
- local `AGENTS.md` files for the touched hotspot trees

## Exit Criteria

- TD022, TD023, and TD024 are all closed with concrete code and validation evidence.
- CLI config-contract changes have clearer ownership boundaries between schema, compatibility migration, and semantic validation.
- Extproc request-time phases are split enough that transport translation, semantic evaluation, and memory/query enrichment do not collapse into the same hotspot files.
- OpenClaw feature work no longer depends on one large page/component/handler seam for unrelated concerns.

## Task List

- [x] `S001` Record the newly-audited hotspot debts in canonical TD entries and create this execution plan.
- [x] `S002` Tighten shared and local AGENT rules so future edits inherit the new hotspot-boundary expectations.
- [x] `S003` Refactor the CLI config-contract path so schema inventories, migration helpers, and semantic validation stop sharing broad duplicated ownership.
- [x] `S004` Split extproc request-time phases into narrower transport, evaluation, and enrichment seams.
- [x] `S005` Extract OpenClaw page/component/backend helpers so realtime transport, page orchestration, and backend proxy/control do not keep collapsing into the same files.
- [x] `S006` Run the required harness validation for the remaining OpenClaw work and close TD024 when that slice catches up to the desired boundaries.

## Current Loop

- 2026-03-18: the audit added new debt records for the CLI config contract, extproc request pipeline, and OpenClaw feature slice.
- 2026-03-18: shared and local AGENT rules were updated so the current hotspots are explicit before code extraction starts.
- 2026-03-18: `S003` landed a shared CLI config-contract inventory so parser, migration, and validator reuse one owner for canonical routing signal and legacy provider tables.
- 2026-03-19: `S004` split extproc memory and signal phases across dedicated search/rewrite/context and signal-input/context/runtime helpers, reducing phase collapse in `req_filter_memory.go` and `req_filter_classification.go`.
- 2026-03-19: `S005` extracted the OpenClaw route shell into `OpenClawPage.tsx`, split tab and wizard ownership across `OpenClawPageTabs.tsx`, `OpenClawArchitectureTab.tsx`, `OpenClawTeamTab.tsx`, `OpenClawWorkerTab.tsx`, and `OpenClawWorkerProvisionWizard.tsx`, moved chat transport and message support into `useClawRoomTransport.ts` and `clawRoomChatSupport.ts`, and split backend worker-chat and automation helpers into `openclaw_room_worker_chat.go` and `openclaw_room_automation.go`.
- 2026-03-19: `S006` passed focused backend tests, `make dashboard-check`, and CPU-local smoke via `make agent-serve-local ENV=cpu AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`, `make agent-smoke-local AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`, and `make agent-stop-local ENV=cpu AGENT_STACK_NAME=pl0007-openclaw AGENT_PORT_OFFSET=600`, closing the remaining OpenClaw validation loop.
- Completed on 2026-03-19: TD022, TD023, and TD024 are all closed, so this execution plan has no remaining open tasks.

## Decision Log

- Prefer repo-local AGENT rules over new repo-local skills when the guidance is tightly coupled to one tree's ownership and hotspots.
- Treat TD006 as structural ratchet debt, not as a substitute for subsystem-specific boundary debt when the feature slice needs a clearer architecture target.
- Use one execution plan for this audit's three related hotspot-boundary workstreams so future loops can resume from the repo alone.
- Close hotspot-specific TDs as soon as their code and harness evidence converge; do not keep TD022/TD023 open just because TD024 still remains in the same execution plan.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-022-cli-config-contract-boundary-collapse.md](../tech-debt/td-022-cli-config-contract-boundary-collapse.md)
- [../tech-debt/td-023-extproc-request-pipeline-phase-collapse.md](../tech-debt/td-023-extproc-request-pipeline-phase-collapse.md)
- [../tech-debt/td-024-openclaw-feature-slice-boundary-collapse.md](../tech-debt/td-024-openclaw-feature-slice-boundary-collapse.md)
- [../adr/README.md](../adr/README.md) (no dedicated ADR yet)
