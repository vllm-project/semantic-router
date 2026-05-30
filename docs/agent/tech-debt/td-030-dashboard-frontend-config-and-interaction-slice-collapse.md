# TD030: Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers

## Status

Open

## Scope

`dashboard/frontend/src/App.tsx`, `dashboard/frontend/src/stores/dslStore.ts`, dashboard config or overview pages, and large interaction containers under `dashboard/frontend/src/components/**`

## Summary

The dashboard frontend now has local rules for pages and shared components, but its most active config and interaction surfaces still concentrate too much behavior into a few orchestration hotspots. Current-source recheck on 2026-05-24 found the older `App.tsx` portion of this debt partly stale: `App.tsx` is already a narrow provider and iframe-guard shell. The live app-shell hotspot had shifted to `dashboard/frontend/src/app/AppRouter.tsx`, where setup/auth orchestration, public routes, protected route registration, config-section routing, protected redirects, and repeated layout composition still lived together. This loop narrowed that seam by moving authenticated route registration and repeated shell wrapping into `AuthenticatedAppRoutes.tsx`, dropping `AppRouter.tsx` from 305 to 75 lines while preserving route behavior. A follow-up slice moved route metadata into a serializable `routeManifest.ts`, added the first source-level Vitest unit test under `dashboard/frontend/src`, and wired that unit test into `make dashboard-check`.

The broader TD030 debt remains open. A second follow-up moved `DashboardPage.tsx` signal-breakdown and decision-preview row shaping into `dashboardPageOverview.ts` with source-level unit tests, but `DashboardPage.tsx` still combines data fetching, overview state, flow-diagram rendering, quick-action cards, runtime inventory shaping, and large section rendering in one page. A third follow-up stabilized `ConfigPageProjectionsSection.tsx` fallback arrays, and a fourth scoped the Fast Refresh lint rule to real component boundaries plus fixed `ColorBends` prop synchronization so the Dashboard frontend lint backlog dropped from 58 warnings to 0. `ConfigPage.tsx`, `SetupWizardPage.tsx`, and `BuilderPage.tsx` remain broad route-level hotspots, while `ChatComponent.tsx` and `ExpressionBuilder.tsx` still mix transport or editor orchestration with large rendering trees and helper logic. The editor-side store in `dslStore.ts` has also grown into a client-side control plane that owns WASM bootstrap, compile/validate/decompile flows, deploy-preview fetches, deploy/rollback calls, runtime health polling, and cross-page refresh events in the same module as local editor state. TD015 tracks weak typing in config and DSL helpers, and TD024 tracks OpenClaw-specific slice collapse, but dashboard frontend still needs a clearer architecture target for editor-control, config page, and interaction-container boundaries.

## Evidence

- [dashboard/frontend/src/App.tsx](../../../dashboard/frontend/src/App.tsx)
- [dashboard/frontend/src/app/AppRouter.tsx](../../../dashboard/frontend/src/app/AppRouter.tsx)
- [dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx](../../../dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx)
- [dashboard/frontend/src/app/routeManifest.ts](../../../dashboard/frontend/src/app/routeManifest.ts)
- [dashboard/frontend/src/app/routeManifest.test.ts](../../../dashboard/frontend/src/app/routeManifest.test.ts)
- [dashboard/frontend/package.json](../../../dashboard/frontend/package.json)
- [dashboard/frontend/eslint.config.js](../../../dashboard/frontend/eslint.config.js)
- [tools/make/dashboard.mk](../../../tools/make/dashboard.mk)
- [dashboard/frontend/src/components/ColorBends.tsx](../../../dashboard/frontend/src/components/ColorBends.tsx)
- [dashboard/frontend/src/pages/dashboardPageOverview.ts](../../../dashboard/frontend/src/pages/dashboardPageOverview.ts)
- [dashboard/frontend/src/pages/dashboardPageOverview.test.ts](../../../dashboard/frontend/src/pages/dashboardPageOverview.test.ts)
- [dashboard/frontend/src/pages/ConfigPageProjectionsSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageProjectionsSection.tsx)
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
- Validation passed for the app-shell route-registry slice: `npm run type-check`; `npm run lint`; `make dashboard-check`.
- Validation passed for the route-manifest frontend unit-test slice: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
- Validation passed for the dashboard overview row-model slice: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
- Validation passed for the projections dependency-stability lint slice: `npm run lint`; `make dashboard-check`.
- Validation passed for the frontend lint boundary cleanup slice: `npm run lint`; `npm run test:unit`; `npm run type-check`; `make dashboard-check`.
- `npm run lint` now exits with 0 warnings for `dashboard/frontend`.

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
- Source-level frontend unit tests cover stable shell, route, editor-control, and interaction-helper seams in addition to Playwright user-flow coverage.
- Local frontend `AGENTS.md` files explicitly name the app-shell, page, and container hotspots and the extraction-first rules expected when they are touched.

## Exit Criteria

- New dashboard frontend features no longer require reopening `App.tsx`, `dslStore.ts`, plus one large page or interaction container for unrelated responsibilities.
- Overview/config page support code, deploy-control orchestration, and interaction-container display logic have narrower primary owners than the current hotspots.
- Dashboard-frontend local `AGENTS.md` coverage explicitly includes the route shell alongside the existing pages and components guidance.
- `make dashboard-check` continues to include frontend unit tests, and new route/page/helper seams have focused source-level coverage rather than only Playwright coverage.
