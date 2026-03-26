# WizMap Knowledge Map Integration Execution Plan

## Goal

- Integrate `WizMap` as a self-hosted static application inside the dashboard platform instead of relying on the public demo or a fragile third-party iframe.
- Add a dedicated `Knowledge Map` route that opens a per-base embedding map for one knowledge base at a time.
- Close the workstream only after data export, dashboard routing, backend hosting, image packaging, auth behavior, and repo-native validation all agree on one supported integration model.

## Scope

- `docs/agent/plans/**`
- `dashboard/frontend/src/**`
- `dashboard/backend/**`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/config/**`
- `tools/docker/**`
- `tools/make/**`
- dashboard image or static-asset packaging needed to ship the self-hosted WizMap app
- targeted tests and repo-native validation for touched dashboard, backend, router API, and packaging surfaces
- nearest local rules for `dashboard/frontend/src/**`, `dashboard/frontend/src/pages/**`, `dashboard/frontend/src/components/**`, `dashboard/backend/**`, `src/semantic-router/pkg/apiserver/**`, and `src/semantic-router/pkg/classification/**`

## Exit Criteria

- The repository defines one supported self-hosted `WizMap` integration path rather than depending on `poloclub.github.io`.
- Dashboard exposes a dedicated `Knowledge Map` route that opens a per-base embedding map from the Knowledge surface.
- Router or dashboard backend exposes a typed export path for the data artifacts required by WizMap for a selected knowledge base.
- The exported data contract is stable enough that one KB can be visualized without ad hoc local notebooks or manual file uploads.
- Auth and same-origin behavior are explicit: the hosted map works inside the dashboard without cross-origin hacks.
- Dashboard and image packaging include the WizMap static app so local and packaged environments ship the same feature.
- Applicable repo-native validation passes, or durable blockers are recorded in this plan before the loop stops.

## Target Integration Model

### Hosted app shape

- `WizMap` is treated as a self-hosted static app owned by the dashboard platform.
- The dashboard serves the WizMap frontend from a repo-controlled static path such as `/embedded/wizmap/`.
- The dashboard frontend adds a first-class route such as `/knowledge-bases/:name/map` that opens the hosted app for one selected knowledge base.

### Data contract

- The backend provides one typed way to request WizMap data for a selected KB.
- The preferred model is:
  - metadata endpoint for the selected KB map
  - data endpoint for WizMap point records
  - grid endpoint for WizMap summaries
- The hosted map loads those endpoints through same-origin URLs rather than public external URLs.

### UX model

- `Knowledge Bases` remains the catalog and CRUD surface.
- Each base gets an `Open Map` action that navigates to the dedicated map route.
- `Knowledge Map` is full-page or near full-page, not embedded inside the existing base detail summary.
- The page owns a small shell around the hosted map:
  - base name
  - back link to `Knowledge`
  - loading/error state
  - optional map metadata or rebuild controls

## Task List

- [x] `WZM001` Create the indexed execution plan, register `pl-0019`, and link it to the KB platform and generic-KB predecessor workstreams.
- [x] `WZM002` Lock the integration contract for hosted `WizMap`: self-hosted static app path, dashboard route shape, data export endpoints, and auth or same-origin behavior.
- [x] `WZM003` Implement the backend data-export surface for one selected KB so WizMap can load the required point and grid artifacts from typed repo-owned endpoints.
- [x] `WZM004` Add the hosted WizMap static-app packaging path to the dashboard build or image flow so local and packaged environments ship the same app.
- [x] `WZM005` Add the dashboard `Knowledge Map` route, base-level entry actions, and page shell around the hosted map.
- [x] `WZM006` Validate route behavior, loading states, and map navigation against large or empty KBs without regressing the existing Knowledge manager UX.
- [x] `WZM007` Update docs, operator-facing notes, and packaging guidance so contributors know how the hosted map is built, served, and loaded with KB data.
- [x] `WZM008` Run the validation ladder, record results and blockers in this plan, and add indexed debt only for gaps that remain after the workstream closes.

## Current Loop

- Date: 2026-03-26
- Current task: `WZM008` completed
- Branch: `vsr/pr-1644-analysis`
- Completed work:
  - shipped the router KB map export surface at `/config/kbs/{name}/map/{metadata,data.ndjson,grid.json,topic.json}` with cached projection artifacts
  - added the dashboard `Knowledge Map` route, base-level `Open Map` actions, and the self-hosted `/embedded/wizmap/` static shell
  - wired dashboard, backend, and image packaging so the vendored WizMap app builds into `dashboard/frontend/dist/embedded/wizmap` and ships in the dashboard and `vllm-sr` images
  - adapted the vendored WizMap app to read repo-owned query params and same-origin KB map data instead of the public demo datasets
  - tightened the map page UX into a compact manager-style summary bar so the map viewport remains the dominant surface
  - fixed the embedded auth chain so KB map data URLs carry `authToken` explicitly and `/embedded/wizmap/assets/*` no longer relies on the dashboard cookie path to load the static bundle
  - follow-up ratchet: removed the outer summary bar entirely so `/knowledge-bases/:name/map` renders as a near-full-screen WizMap shell with only a minimal back affordance
  - follow-up ratchet: changed KB-mode WizMap defaults to a points-first view (`point` on, `contour` off, `label` off) and removed KB grouped overlays from the exported grid contract so the initial map is directly explorable
  - follow-up ratchet: simplified the embedded footer count path to show the loaded KB point count consistently instead of stale subset text, and added a Playwright route test to prevent the old summary bar layout from returning
  - follow-up ratchet: made `/knowledge-bases/:name/map` a standalone full-screen route outside the dashboard layout chrome so the hosted map owns the viewport
  - follow-up ratchet: refactored the embedded top controls into valid button/menu structure, removed KB-mode topic tile boxes from label rendering, and added a fixed selected-point card so click-on-point matches the hosted product expectation
- Validation:
  - `make agent-validate`
  - `cd dashboard/frontend && npm run type-check`
  - `cd dashboard/frontend && npm run build`
  - `cd dashboard/backend && PATH=/usr/local/go/bin:$PATH GOCACHE=/Users/bitliu/.codex/worktrees/eede/vs/.codex-go-cache go test ./auth -run 'Test(RequiresAuthentication|RequiredPermission|ExtractAccessToken)' -count=1`
  - `cd dashboard/backend && PATH=/usr/local/go/bin:$PATH GOCACHE=/Users/bitliu/.codex/worktrees/eede/vs/.codex-go-cache go test ./handlers -run 'TestWizMapStaticHandler' -count=1`
  - `cd src/semantic-router && PATH=/usr/local/go/bin:$PATH GOCACHE=/Users/bitliu/.codex/worktrees/eede/vs/.codex-go-cache go test ./pkg/apiserver -run 'TestHandleKnowledgeBaseMap(Endpoints|MissingKnowledgeBase)' -count=1`
  - `cd dashboard/wizmap && npm run build:embedded`
  - `cd dashboard/frontend && npx playwright test e2e/knowledge-map.spec.ts`
  - `make dashboard-check`
- Recorded blocker:
  - none

## Decision Log

- 2026-03-26: The supported product direction is self-hosted `WizMap`, not an iframe of the public demo.
- 2026-03-26: `Knowledge Map` is a dedicated route, not a small widget embedded into the existing base detail summary.
- 2026-03-26: The integration should use same-origin repo-owned endpoints for KB map data rather than public cross-origin URLs.
- 2026-03-26: This is a new execution workstream because it spans dashboard frontend, dashboard backend, router data export, and image packaging.
- 2026-03-26: The embedded WizMap shell must not depend on dashboard cookies for its static bundle; KB data endpoints carry explicit `authToken` query params and `/embedded/wizmap/assets/*` is treated as static transport rather than a protected product page.
- 2026-03-26: The `Knowledge Map` page should keep the base summary compact and reserve most of the vertical space for the map viewport.
- 2026-03-26: The final route should drop dashboard header chrome entirely and behave like a standalone full-screen WizMap surface with only a local back affordance.
- 2026-03-26: KB-mode label overlays should prefer plain labels and selected-point cards over dense tile rectangles because PCA-style KB projections cluster much more tightly than the public demo datasets.

## Follow-up Debt / ADR Links

- Predecessor workstream: [pl-0017-taxonomy-classifier-platform-loop.md](pl-0017-taxonomy-classifier-platform-loop.md)
- Predecessor workstream: [pl-0018-generic-embedding-kb-workstream.md](pl-0018-generic-embedding-kb-workstream.md)
- Related dashboard/frontend ratchet: [pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md](pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md)
- Reuse existing debt first if the integration still leaves major hotspots or split contracts:
  - [TD025 Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
  - [TD030 Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
