# Extproc Response and Dashboard Frontend Boundary Ratchet Execution Plan

## Goal

- Turn the latest design audit into executable, resumable cleanup work for the extproc response pipeline and the dashboard frontend route/config/interaction surfaces.
- Reduce responsibility collapse in `processor_res_body*.go`, response-side filters, `App.tsx`, dashboard overview or config pages, and large frontend interaction containers.
- Retire TD029 and TD030 only after code, local rules, and validation evidence converge on narrower seams.

## Scope

- `src/semantic-router/pkg/extproc/**`
- `dashboard/frontend/src/App.tsx`
- `dashboard/frontend/src/pages/**`
- `dashboard/frontend/src/components/**`
- shared harness docs and local `AGENTS.md` files for the touched hotspot trees

## Exit Criteria

- TD029 and TD030 are both closed with concrete code and validation evidence.
- Extproc response changes no longer collapse provider normalization, streaming finalization, replay/cache persistence, and response-side warning shaping into one hotspot seam.
- Dashboard frontend changes no longer collapse route shell/auth gating, page orchestration, and large interaction-container rendering into the same hotspots.
- Shared and local AGENT rules stay aligned with the active hotspot boundaries so future work inherits the narrower design target.

## Task List

- [x] `S001` Record the extproc response-pipeline and dashboard frontend boundary debts in canonical TD entries and create this execution plan.
- [x] `S002` Tighten shared and local AGENT rules so the response-pipeline and dashboard frontend hotspot trees are explicit before code extraction starts.
- [x] `S003` Split extproc response ownership so provider normalization, non-streaming response shaping, replay/cache persistence, streaming finalization, and response warnings stop accumulating in the current response seams.
- [ ] `S004` Narrow the dashboard frontend route shell so auth/setup gating, config-section routing, and shared layout composition stop growing in `App.tsx`.
- [ ] `S005` Split dashboard overview/config and large interaction containers so page orchestration, helper shaping, and display fragments stop collapsing into `DashboardPage.tsx`, `ConfigPage.tsx`, `SetupWizardPage.tsx`, `ChatComponent.tsx`, and `ExpressionBuilder.tsx`.
- [ ] `S006` Run the required harness and subsystem validation for the remaining work and close TD029 and TD030 when the narrowed seams are proven.

## Current Loop

- 2026-03-19: the latest design audit identified two additional subsystem-specific gaps not yet captured by TD023, TD024, or TD015: extproc response-phase collapse and dashboard frontend route/config/interaction collapse.
- 2026-03-19: TD029 and TD030 were added so those active hotspots are no longer only implicit structural debt.
- 2026-03-19: shared and local AGENT rules were updated so future edits discover the response-pipeline seam and dashboard frontend app-shell/page/component boundaries before widening them further.
- 2026-03-19: `S003` completed: usage accounting, cache persistence, and memory scheduling extracted into `processor_res_usage.go`, `processor_res_cache.go`, and `processor_res_memory.go` with targeted test suites; pipeline and streaming orchestrators slimmed to thin dispatchers; extproc-local `AGENTS.md` updated with primary phase owners; TD029 closed.
- 2026-04-03: ADR0006 now provides the shared cross-stack dependency direction for the remaining dashboard-side work in this plan; frontend and dashboard follow-up should continue to narrow interface surfaces instead of recreating control-plane logic in route or container hotspots.
- Next loop target: `S004` dashboard frontend route shell narrowing.

## Decision Log

- Prefer repo-local AGENT rules and execution plans over a new repo-local skill when the guidance is tightly coupled to one subtree's ownership and hotspots.
- Treat TD006 as a structural ratchet, not as a substitute for subsystem-specific debt when the repo needs a clearer architecture target for active modules.
- Keep extproc response-pipeline cleanup and dashboard frontend shell/page cleanup in the same plan because both emerged from the same latest design audit and both need durable governance plus follow-up extraction work.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-029-extproc-response-pipeline-phase-collapse.md](../tech-debt/td-029-extproc-response-pipeline-phase-collapse.md)
- [../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md](../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md)
