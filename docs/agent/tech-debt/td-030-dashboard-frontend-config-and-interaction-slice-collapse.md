# TD030: Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

Dashboard frontend route shell, config/editor store, overview/config/setup pages,
and large interaction containers under `dashboard/frontend/src/**`.

## Summary

The dashboard frontend has improved route and component guidance, and several
shell/helper slices have already been extracted. The remaining debt is that the
most active user-facing surfaces still concentrate too much behavior:

- page shells combine data fetching, state shaping, layout, and large render
  fragments
- `dslStore.ts` mixes local editor state with WASM lifecycle, deploy/rollback
  calls, runtime health polling, and cross-page refresh events
- large interaction containers still combine transport/editor orchestration with
  display branches and helper logic

## Evidence

- [dashboard/frontend/src/app/AppRouter.tsx](../../../dashboard/frontend/src/app/AppRouter.tsx)
- [dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx](../../../dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx)
- [dashboard/frontend/src/app/routeManifest.ts](../../../dashboard/frontend/src/app/routeManifest.ts)
- [dashboard/frontend/src/stores/dslStore.ts](../../../dashboard/frontend/src/stores/dslStore.ts)
- [dashboard/frontend/src/pages/DashboardPage.tsx](../../../dashboard/frontend/src/pages/DashboardPage.tsx)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/SetupWizardPage.tsx](../../../dashboard/frontend/src/pages/SetupWizardPage.tsx)
- [dashboard/frontend/src/pages/BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)
- [dashboard/frontend/src/components/ExpressionBuilder.tsx](../../../dashboard/frontend/src/components/ExpressionBuilder.tsx)
- [dashboard/frontend/src/pages/AGENTS.md](../../../dashboard/frontend/src/pages/AGENTS.md)
- [dashboard/frontend/src/components/AGENTS.md](../../../dashboard/frontend/src/components/AGENTS.md)
- [docs/agent/tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md](td-015-weakly-typed-config-and-dsl-contracts.md)
- [docs/agent/tech-debt/td-006-structural-rule-exceptions.md](td-006-structural-rule-exceptions.md)

## Why It Matters

- Narrow dashboard changes still reopen route, page, store, and container
  hotspots.
- Editor control-plane side effects are harder to test when they live inside the
  same store as local draft state.
- v0.3 dashboard stability depends on predictable route/page ownership and
  focused source-level tests.

## Desired End State

- Route shell owns routing, auth/setup gating, and shared layout composition
  only.
- Editor stores own local drafting state, while deploy/rollback, runtime
  polling, and WASM lifecycle live behind narrower hooks or services.
- Overview/config/setup/builder pages orchestrate sibling support modules rather
  than carrying large helper tables and render fragments inline.
- Large interaction containers keep orchestration separate from display and
  parsing helpers.

## Exit Criteria

- New dashboard frontend features do not require editing the app shell, editor
  store, and large page/container for unrelated responsibilities.
- `make dashboard-check` includes focused source-level coverage for route,
  page/helper, and editor-control seams.
- Dashboard local rules and implementation shape agree on extraction-first
  ownership for pages and large components.
