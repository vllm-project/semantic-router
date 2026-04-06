# TD030: Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers

## Status

Closed

## Scope

`dashboard/frontend/src/App.tsx`, `dashboard/frontend/src/stores/dslStore.ts`, dashboard config or overview pages, and large interaction containers under `dashboard/frontend/src/components/**`

## Summary

The dashboard frontend route shell, editor control plane, and overview page now sit on narrower sibling-owned seams instead of collapsing back into the same hotspot files. `App.tsx` keeps route registration plus auth/setup gating, while repeated layout wrappers, config-section routing helpers, knowledge-base route adapters, and legacy redirects now live in `appShellRouteSupport.tsx`. `dslStore.ts` keeps editor state, WASM bootstrap, and compile or validate flows, while deploy preview, deploy or rollback mutations, runtime-health polling, and version fetches now live in `dslStoreDeploySupport.ts`. `DashboardPage.tsx` keeps fetch cadence, polling, and navigation orchestration, while stats shaping, decision categorization, flow-diagram rendering, right-column actions, and lower-panel rendering moved into adjacent page modules. Together with the existing `configPage*`, `setupWizardSupport.ts`, `ChatComponent*`, and `ExpressionBuilder*` sibling modules plus the route-shell guidance in `dashboard/frontend/src/AGENTS.md`, the frontend now has a durable boundary contract for the surfaces this debt covered.

## Evidence

- [dashboard/frontend/src/App.tsx](../../../dashboard/frontend/src/App.tsx)
- [dashboard/frontend/src/appShellRouteSupport.tsx](../../../dashboard/frontend/src/appShellRouteSupport.tsx)
- [dashboard/frontend/src/stores/dslStore.ts](../../../dashboard/frontend/src/stores/dslStore.ts)
- [dashboard/frontend/src/stores/dslStoreSupport.ts](../../../dashboard/frontend/src/stores/dslStoreSupport.ts)
- [dashboard/frontend/src/stores/dslStoreDeploySupport.ts](../../../dashboard/frontend/src/stores/dslStoreDeploySupport.ts)
- [dashboard/frontend/src/pages/DashboardPage.tsx](../../../dashboard/frontend/src/pages/DashboardPage.tsx)
- [dashboard/frontend/src/pages/dashboardPageSupport.ts](../../../dashboard/frontend/src/pages/dashboardPageSupport.ts)
- [dashboard/frontend/src/pages/DashboardStatsCards.tsx](../../../dashboard/frontend/src/pages/DashboardStatsCards.tsx)
- [dashboard/frontend/src/pages/DashboardMiniFlowDiagram.tsx](../../../dashboard/frontend/src/pages/DashboardMiniFlowDiagram.tsx)
- [dashboard/frontend/src/pages/DashboardRightColumn.tsx](../../../dashboard/frontend/src/pages/DashboardRightColumn.tsx)
- [dashboard/frontend/src/pages/DashboardBottomPanels.tsx](../../../dashboard/frontend/src/pages/DashboardBottomPanels.tsx)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/configPageSupport.ts](../../../dashboard/frontend/src/pages/configPageSupport.ts)
- [dashboard/frontend/src/pages/SetupWizardPage.tsx](../../../dashboard/frontend/src/pages/SetupWizardPage.tsx)
- [dashboard/frontend/src/pages/setupWizardSupport.ts](../../../dashboard/frontend/src/pages/setupWizardSupport.ts)
- [dashboard/frontend/src/pages/BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)
- [dashboard/frontend/src/components/ChatComponentConversationViewport.tsx](../../../dashboard/frontend/src/components/ChatComponentConversationViewport.tsx)
- [dashboard/frontend/src/components/ChatComponentControls.tsx](../../../dashboard/frontend/src/components/ChatComponentControls.tsx)
- [dashboard/frontend/src/components/ExpressionBuilder.tsx](../../../dashboard/frontend/src/components/ExpressionBuilder.tsx)
- [dashboard/frontend/src/components/ExpressionBuilderSupport.ts](../../../dashboard/frontend/src/components/ExpressionBuilderSupport.ts)
- [dashboard/frontend/src/components/ExpressionBuilderFlow.ts](../../../dashboard/frontend/src/components/ExpressionBuilderFlow.ts)
- [dashboard/frontend/src/components/ExpressionBuilderDialogs.tsx](../../../dashboard/frontend/src/components/ExpressionBuilderDialogs.tsx)
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

- Satisfied on 2026-04-06: new dashboard frontend work no longer needs to reopen `App.tsx`, `dslStore.ts`, and `DashboardPage.tsx` for unrelated responsibilities because route-shell wrappers, deploy-control actions, and overview render fragments now live in sibling support modules.
- Satisfied on 2026-04-06: overview/config page support code, deploy-control orchestration, and interaction-container display logic have narrower primary owners across `dashboardPageSupport.ts`, `dslStoreDeploySupport.ts`, the existing `configPage*` and `setupWizardSupport.ts` helpers, and the extracted `ChatComponent*` or `ExpressionBuilder*` modules.
- Satisfied on 2026-04-06: dashboard-frontend local `AGENTS.md` coverage explicitly includes the route shell alongside the existing pages and components guidance.

## Retirement Notes

- `dashboard/frontend/src/App.tsx` now stays focused on route registration and auth/setup gating, while `dashboard/frontend/src/appShellRouteSupport.tsx` owns repeated shell wrappers, setup status UI, config-section routing adapters, and legacy redirects.
- `dashboard/frontend/src/stores/dslStore.ts` now centers on editor-local state plus WASM, compile, validate, and decompile actions, while `dashboard/frontend/src/stores/dslStoreDeploySupport.ts` owns deploy preview, deploy or rollback mutations, runtime-health polling, and config-version fetches.
- `dashboard/frontend/src/pages/DashboardPage.tsx` now orchestrates fetch cadence, polling, and navigation, while `dashboard/frontend/src/pages/dashboardPageSupport.ts`, `DashboardStatsCards.tsx`, `DashboardMiniFlowDiagram.tsx`, `DashboardRightColumn.tsx`, and `DashboardBottomPanels.tsx` own the extracted overview helpers and large render sections.
- The rest of the debt scope now follows the same local boundary pattern through existing sibling modules such as `configPageSupport.ts`, `setupWizardSupport.ts`, `ChatComponentConversationViewport.tsx`, `ChatComponentControls.tsx`, `ExpressionBuilderSupport.ts`, `ExpressionBuilderFlow.ts`, and `ExpressionBuilderDialogs.tsx`.

## Validation

- `cd /Users/bitliu/vs/dashboard/frontend && npm run type-check`
- `cd /Users/bitliu/vs/dashboard/frontend && npm run lint`
- `make agent-validate`
- `make agent-lint AGENT_CHANGED_FILES_PATH=/tmp/vsr_td030_changed.txt`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td030_changed.txt`
