# TD030: Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers

## Status

Open

## Scope

`dashboard/frontend/src/App.tsx`, `dashboard/frontend/src/stores/dslStore.ts`, dashboard config or overview pages, and large interaction containers under `dashboard/frontend/src/components/**`

## Summary

The dashboard frontend now has local rules for pages and shared components, but its most active config and interaction surfaces still concentrate too much behavior into a few orchestration hotspots. `App.tsx` owns the top-level route shell, auth/setup gating, config-section routing, and repeated layout composition. `DashboardPage.tsx` combines data fetching, overview state, flow-diagram rendering, quick-action cards, and runtime inventory shaping in one page. `ConfigPage.tsx`, `SetupWizardPage.tsx`, and `BuilderPage.tsx` remain broad route-level hotspots, while `ChatComponent.tsx` and `ExpressionBuilder.tsx` still mix transport or editor orchestration with large rendering trees and helper logic. The editor-side store in `dslStore.ts` has also grown into a client-side control plane that owns WASM bootstrap, compile/validate/decompile flows, deploy-preview fetches, deploy/rollback calls, runtime health polling, and cross-page refresh events in the same module as local editor state. TD015 tracks weak typing in config and DSL helpers, and TD024 tracks OpenClaw-specific slice collapse, but the repo still lacks a dashboard-frontend-specific debt item for the non-OpenClaw route shell, editor-control, config page, and interaction-container boundaries.

## Evidence

- [dashboard/frontend/src/App.tsx](../../../dashboard/frontend/src/App.tsx)
- [dashboard/frontend/src/stores/dslStore.ts](../../../dashboard/frontend/src/stores/dslStore.ts)
- [dashboard/frontend/src/pages/DashboardPage.tsx](../../../dashboard/frontend/src/pages/DashboardPage.tsx)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/SetupWizardPage.tsx](../../../dashboard/frontend/src/pages/SetupWizardPage.tsx)
- [dashboard/frontend/src/pages/BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)
- [dashboard/frontend/src/components/ExpressionBuilder.tsx](../../../dashboard/frontend/src/components/ExpressionBuilder.tsx)
- [dashboard/frontend/src/pages/AGENTS.md](../../../dashboard/frontend/src/pages/AGENTS.md)
- [dashboard/frontend/src/components/AGENTS.md](../../../dashboard/frontend/src/components/AGENTS.md)
- [dashboard/frontend/src/AGENTS.md](../../../dashboard/frontend/src/AGENTS.md)
- [docs/agent/tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md](td-015-weakly-typed-config-and-dsl-contracts.md)
- [docs/agent/tech-debt/td-024-openclaw-feature-slice-boundary-collapse.md](td-024-openclaw-feature-slice-boundary-collapse.md)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- Narrow dashboard changes still reopen the same route shell, page, store, and container hotspots even when the work only targets config navigation, overview projection, deploy UX, or chat/editor display behavior.
- Auth/setup gating, route-level layout composition, async page orchestration, editor/WASM lifecycle, and large UI containers are harder to test or evolve independently when they remain coupled in the same files.
- The editor store currently mixes client-side state with backend deploy orchestration and runtime polling, which makes retry/error handling and future multi-tab or long-running workflows harder to reason about.
- Existing debts explain weak typing or OpenClaw-specific boundaries, but they do not give contributors a clear architecture target for the general dashboard frontend shell and config/interaction surfaces.

## Desired End State

- The dashboard frontend route shell keeps auth/setup gating, routing, and shared layout composition on a narrower seam than page-specific logic.
- Editor stores keep local drafting state, while deploy/rollback orchestration, runtime polling, and side-effectful control-plane calls move behind narrower service or hook seams.
- Overview/config pages orchestrate sibling support modules instead of carrying large helper tables, fetch shaping, and render fragments inline.
- Large interaction containers such as `ChatComponent.tsx` and `ExpressionBuilder.tsx` keep transport or editor orchestration separate from display fragments, parsing helpers, and repeated UI branches.
- Local frontend `AGENTS.md` files explicitly name the app-shell, page, and container hotspots and the extraction-first rules expected when they are touched.

## Exit Criteria

- New dashboard frontend features no longer require reopening `App.tsx`, `dslStore.ts`, plus one large page or interaction container for unrelated responsibilities.
- Overview/config page support code, deploy-control orchestration, and interaction-container display logic have narrower primary owners than the current hotspots.
- Dashboard-frontend local `AGENTS.md` coverage explicitly includes the route shell alongside the existing pages and components guidance.
