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
- [ ] `WZM002` Lock the integration contract for hosted `WizMap`: self-hosted static app path, dashboard route shape, data export endpoints, and auth or same-origin behavior.
- [ ] `WZM003` Implement the backend data-export surface for one selected KB so WizMap can load the required point and grid artifacts from typed repo-owned endpoints.
- [ ] `WZM004` Add the hosted WizMap static-app packaging path to the dashboard build or image flow so local and packaged environments ship the same app.
- [ ] `WZM005` Add the dashboard `Knowledge Map` route, base-level entry actions, and page shell around the hosted map.
- [ ] `WZM006` Validate route behavior, loading states, and map navigation against large or empty KBs without regressing the existing Knowledge manager UX.
- [ ] `WZM007` Update docs, operator-facing notes, and packaging guidance so contributors know how the hosted map is built, served, and loaded with KB data.
- [ ] `WZM008` Run the validation ladder, record results and blockers in this plan, and add indexed debt only for gaps that remain after the workstream closes.

## Current Loop

- Date: 2026-03-26
- Current task: `WZM001` completed
- Branch: `vsr/pr-1644-analysis`
- Planned loop order:
  - `L1` lock the hosted-app and data-export contract
  - `L2` land backend export endpoints and KB-to-map data shaping
  - `L3` land dashboard route, navigation, and page shell
  - `L4` land static-app packaging and environment integration
  - `L5` close docs, smoke coverage, and validation
- Initial discovery:
  - reviewed the current Knowledge Base dashboard routes and manager surfaces
  - reviewed current dashboard embedded-app patterns such as Grafana under `/embedded/grafana/`
  - inspected the current KB API surfaces under `src/semantic-router/pkg/apiserver/route_kbs.go`
  - researched `poloclub/wizmap` and confirmed the primary distribution is a standalone frontend app plus precomputed data files, not a React component library
  - confirmed the public WizMap demo can load data by URL, but the desired repo direction is self-hosted same-origin integration
  - registered this execution plan in the canonical plan index and repo manifest so the workstream can resume from the repository alone

## Decision Log

- 2026-03-26: The supported product direction is self-hosted `WizMap`, not an iframe of the public demo.
- 2026-03-26: `Knowledge Map` is a dedicated route, not a small widget embedded into the existing base detail summary.
- 2026-03-26: The integration should use same-origin repo-owned endpoints for KB map data rather than public cross-origin URLs.
- 2026-03-26: This is a new execution workstream because it spans dashboard frontend, dashboard backend, router data export, and image packaging.

## Follow-up Debt / ADR Links

- Predecessor workstream: [pl-0017-taxonomy-classifier-platform-loop.md](pl-0017-taxonomy-classifier-platform-loop.md)
- Predecessor workstream: [pl-0018-generic-embedding-kb-workstream.md](pl-0018-generic-embedding-kb-workstream.md)
- Related dashboard/frontend ratchet: [pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md](pl-0010-extproc-response-and-dashboard-frontend-boundary-ratchet.md)
- Reuse existing debt first if the integration still leaves major hotspots or split contracts:
  - [TD025 Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
  - [TD030 Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
