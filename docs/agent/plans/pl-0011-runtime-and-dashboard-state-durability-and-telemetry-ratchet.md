# Runtime And Dashboard State Durability And Telemetry Ratchet Execution Plan

## Goal

- Turn the latest repository architecture review into a resumable cleanup loop for router and dashboard state ownership.
- Converge on an explicit contract for durability, recovery, and control-plane telemetry across runtime stores, dashboard workflows, and CLI-mounted local state.
- Retire the newest cross-stack debt only after router, dashboard, CLI, docs, and validation evidence all agree on the same state model.

## Scope

- `src/semantic-router/pkg/{config,responsestore,routerreplay,vectorstore,startupstatus}/**`
- `dashboard/backend/{auth,evaluation,mlpipeline,modelresearch,handlers}/**`
- `dashboard/frontend/src/{hooks,components,utils}/**`
- `src/vllm-sr/cli/**`
- `docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md`
- shared harness docs or validation assets needed to index and enforce the workstream

## Exit Criteria

- Router and dashboard contributors can discover one canonical state taxonomy with clear ownership, durability, and restart expectations.
- At least the highest-risk user-visible surfaces stop depending on process memory alone for restart-sensitive state.
- Progress and status reporting for long-running workflows become typed and restart-aware enough that dashboard recovery does not depend on log scraping or browser-local state.
- The resulting architecture is expressed through updated docs, validation, and issue-sized follow-up changes rather than only review text.

## Task List

- [x] `S001` Seed the workstream from the architecture review by creating the indexed debt entry and this execution plan.
- [ ] `S002` Inventory state surfaces and classify each one by owner, backend, durability level, restart expectations, and telemetry source.
- [ ] `S003` Harden router-side state seams so response storage, replay, vector-store metadata, file metadata, and runtime status stop depending on process memory or temp-only state where restart-safe behavior is required.
- [ ] `S004` Unify dashboard and CLI workspace-state ownership so workflow jobs, campaign state, OpenClaw state, chat/session behavior, and local workspace mounts follow one server-owned persistence model with explicit local-dev adapters.
- [ ] `S005` Replace ad hoc health and progress reporting with typed restart-aware status seams for at least one long-running dashboard workflow and one router runtime surface.
- [ ] `S006` Split the work into maintainer-sized GitHub issues with priority, area labels, validation expectations, and debt cross-links.

## Current Loop

- 2026-03-20: architecture review completed for router core, dashboard frontend/backend, Python CLI, training integrations, native bindings, and shared contracts. The review found an uncovered cross-stack gap around state durability, recovery, and telemetry semantics, so TD034 and this plan were created as the durable execution anchor.
- Next focus: `S002` should produce the first explicit state inventory and identify which surfaces are intentionally ephemeral versus mistakenly memory-backed by default.

## Decision Log

- 2026-03-20: keep this workstream separate from TD021. Shared Milvus lifecycle reuse is a backend-adapter problem; this plan is broader and covers user-visible state ownership, restart contracts, and telemetry semantics across router, dashboard, and CLI.
- 2026-03-20: keep this workstream separate from TD005. Dashboard auth and browser session hardening remain their own product slice, but they now need to align with the same cross-stack state taxonomy.
- 2026-03-20: keep this workstream separate from TD031. Global bootstrap state and single-consumer config reload are part of runtime composition debt; this plan focuses on what state exists, where it lives, and how it survives restart.

## Follow-up Debt / ADR Links

- [TD005 Dashboard Lacks Enterprise Console Foundations](../tech-debt/td-005-dashboard-enterprise-console-foundations.md)
- [TD021 Milvus Lifecycle Logic Is Duplicated Across Runtime Stores](../tech-debt/td-021-milvus-adapter-duplication-across-runtime-stores.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD032 Training and Evaluation Artifact Contracts Still Drift Across Dashboard, Runtime, and Scripts](../tech-debt/td-032-training-evaluation-artifact-contract-drift.md)
- [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
