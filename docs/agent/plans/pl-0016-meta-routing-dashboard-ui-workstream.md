# Meta-Routing Dashboard UI Workstream

## Goal

- Turn the existing `routing.meta` runtime and feedback seam into a complete dashboard product workflow.
- Let dashboard users author meta-routing policy visually, observe rollout health, inspect one request in detail, and compare base versus refined routing outcomes without leaving the product.
- Deliver this without growing existing dashboard hotspots or introducing a second persistence owner for meta-routing records.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `dashboard/frontend/src/App.tsx`
- `dashboard/frontend/src/pages/**`
- `dashboard/frontend/src/components/**`
- `dashboard/backend/router/**`
- `dashboard/backend/handlers/**`
- new dashboard backend support modules for meta-routing record transport or aggregation
- `src/semantic-router/pkg/extproc/**`
- `website/docs/**`
- `config/README.md`
- targeted unit, integration, and E2E coverage for authoring and runtime inspection flows
- nearest local `AGENTS.md` files for touched dashboard frontend hotspots and any router runtime or backend seams touched by the workstream

## Exit Criteria

- Dashboard Config includes a dedicated visual section that can fully author `routing.meta` without requiring users to hand-edit raw YAML or DSL.
- The router exposes a stable HTTP query surface for persisted meta-routing feedback records with list, aggregate, and detail endpoints.
- Dashboard backend proxies the meta-routing feedback APIs through `/api/router/v1/meta_routing_feedback*` using the same transport posture as other router-owned APIs.
- Dashboard exposes a dedicated `/meta-routing` route with:
  - rollout summary cards
  - aggregate charts
  - searchable and filterable activity table
  - record detail inspector
- The record inspector shows base pass, refinement pass, assessment triggers, root causes, planned actions, executed actions, and final outcome in one place.
- The dashboard experience supports the full operator workflow:
  - configure policy
  - save or deploy config
  - observe recent meta-routing behavior
  - inspect one request
  - decide whether to promote the rollout mode
- Existing dashboard hotspots do not absorb a second responsibility collapse:
  - `ConfigPage.tsx` only wires a dedicated section module
  - `InsightsPage.tsx` stays replay-centric
  - `BuilderPage.tsx` is not made the primary visual meta-routing editor
- Public docs and tutorials describe the same user flow and API ownership model as the implemented surfaces.
- Applicable backend tests, frontend tests, E2E coverage, and governance validation pass for the changed-file set.

## Target Design

### User Workflow

The intended dashboard workflow is:

1. Open `Config -> Meta Routing` and author `routing.meta`.
2. Save or deploy the config through the existing canonical config flow.
3. Send traffic through the router while the router persists `FeedbackRecord`.
4. Open `/meta-routing` to see rollout-level aggregates and recent activity.
5. Open one record to inspect pass traces, triggers, root causes, action planning, and final outcome.
6. Use the resulting evidence to keep the current mode or move from `observe` to `shadow` to `active`.

### Page Map

- `ConfigPage`
  - add a new `ConfigPageMetaRoutingSection`
  - keep page orchestration in the page file
  - move field builders, edit shapes, and save transforms into adjacent support modules
- `MetaRoutingPage`
  - new dedicated route and nav entry
  - owns fetch cadence, filter state, pagination, and inspector modal state
- `InsightsPage`
  - keep replay economics ownership
  - add only lightweight cross-links when `router_replay_id` is present

### Config Cards

The Config authoring surface should expose:

- `Mode & Budget`
- `Trigger Policy`
- `Required Families`
- `Family Disagreements`
- `Allowed Actions`
- `Rollout Notes`

These are edits over the existing canonical config write path. No special write API is introduced.

### Operations Cards and Panels

The `/meta-routing` page should expose:

- summary cards:
  - total requests
  - planned refinement rate
  - executed refinement rate
  - overturn rate
  - average latency delta
  - p95 latency delta
  - top trigger
  - top root cause
- charts:
  - mode distribution
  - trigger distribution
  - root-cause distribution
  - action-type distribution
  - refined-family distribution
  - decision-change distribution
  - latency-delta histogram
- activity table:
  - mode
  - planned or executed
  - overturned or not
  - decision
  - model
  - triggers
  - root causes
  - actions
  - latency delta
  - response status
  - record timestamp
- detail inspector:
  - request and response summary
  - base-pass panel
  - refinement-pass panel
  - side-by-side compare section
  - trace-quality metrics
  - assessment and plan timeline
  - downstream weak labels
  - router replay cross-link when available

### API Contract

The data source for dashboard inspection is the router-owned feedback recorder.

Router-side APIs:

- `GET /v1/meta_routing_feedback`
- `GET /v1/meta_routing_feedback/aggregate`
- `GET /v1/meta_routing_feedback/{id}`

Dashboard proxy APIs:

- `GET /api/router/v1/meta_routing_feedback`
- `GET /api/router/v1/meta_routing_feedback/aggregate`
- `GET /api/router/v1/meta_routing_feedback/{id}`

Query filters:

- `limit`
- `offset`
- `search`
- `mode`
- `trigger`
- `root_cause`
- `action_type`
- `signal_family`
- `overturned`
- `decision`
- `model`
- `response_status`

List responses should stay summary-oriented. Detail responses should return the full `FeedbackRecord`.

### Boundary Notes

- Router runtime owns record storage and query semantics.
- Dashboard backend owns proxy registration and any thin HTTP adaptation required by the dashboard.
- Dashboard frontend owns operator workflow presentation.
- Builder visual CRUD for `META` is explicitly deferred until Config authoring and runtime inspection are finished.

## Task List

- [x] `MRDUI001` Create an indexed ADR and execution plan for the dashboard-facing meta-routing product surface, and register both in governance inventories before implementation starts.
- [x] `MRDUI002` Define the router-side `meta_routing_feedback` HTTP contract, including list, aggregate, detail, filters, object names, and summary-versus-detail payload boundaries.
- [x] `MRDUI003` Implement router-side list, aggregate, and detail handlers over persisted `FeedbackRecord` data without introducing a second storage owner.
- [x] `MRDUI004` Register and proxy the new router APIs through dashboard backend transport, and add typed frontend fetch helpers for the new payload shapes.
- [x] `MRDUI005` Add a dedicated Config authoring section for `routing.meta` with support modules for fields, edit transforms, and save flow wiring.
- [x] `MRDUI006` Add the new `/meta-routing` route, navigation entry, and summary-card or chart support modules without growing route-shell hotspots beyond local rules.
- [x] `MRDUI007` Add the activity table, filters, aggregate panels, and record detail inspector with base-versus-refined comparison and replay cross-links.
- [x] `MRDUI008` Add lightweight contextual links from nearby surfaces such as `Insights`, plus onboarding or docs affordances that explain the observe-to-shadow-to-active workflow.
- [x] `MRDUI009` Add focused backend tests, frontend tests, and E2E scenarios for authoring, list or aggregate inspection, and detail inspection flows.
- [x] `MRDUI010` Update public docs, run the applicable validation ladder, and record any remaining product or hotspot gaps as indexed debt if they are not retired in this workstream.

## Current Loop

- Date: 2026-03-24
- Current task: `MRDUI010` completed
- Branch: `main`
- Planned loop order:
  - `L1` lock the dashboard product boundary and execution record
  - `L2` land router-side feedback query contract and handlers
  - `L3` land dashboard backend proxy seam and typed frontend client
  - `L4` land Config authoring surface
  - `L5` land the dedicated meta-routing operations page
  - `L6` land inspector depth and cross-links
  - `L7` land tests, docs, and validation close-out
- Commands run:
  - `make agent-report ENV=cpu CHANGED_FILES="docs/agent/adr/adr-0004-meta-routing-dashboard-ui-surface.md,docs/agent/plans/pl-0016-meta-routing-dashboard-ui-workstream.md,docs/agent/adr/README.md,docs/agent/plans/README.md,tools/agent/repo-manifest.yaml"`
  - broad `codebase-retrieval` across governance docs, existing meta-routing docs, dashboard frontend or backend surfaces, insights, config, builder, and router replay transport seams
  - targeted source reads for `dashboard/frontend/src/pages/AGENTS.md`, `docs/agent/README.md`, `docs/agent/governance.md`, `dashboard/frontend/src/pages/InsightsPage.tsx`, `dashboard/backend/router/core_routes.go`, `src/semantic-router/pkg/extproc/meta_routing_types.go`, and nearby router replay or feedback code
  - `go test ./src/semantic-router/pkg/extproc -run 'TestHandleMetaRoutingFeedback|TestCollectMetaRoutingFeedbackRecordsReturnsParsedPayloads'`
  - `make dashboard-check`
  - `make agent-e2e-affected CHANGED_FILES="dashboard/backend/router/proxy_routes.go dashboard/frontend/src/App.tsx dashboard/frontend/src/components/ConfigNav.tsx dashboard/frontend/src/components/LayoutNavSupport.ts dashboard/frontend/src/pages/ConfigPage.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.module.css dashboard/frontend/src/pages/InsightsPage.tsx dashboard/frontend/src/pages/insightsPageSupport.tsx dashboard/frontend/src/pages/MetaRoutingPage.tsx dashboard/frontend/src/pages/MetaRoutingPage.module.css dashboard/frontend/src/pages/metaRoutingPageSupport.tsx dashboard/frontend/src/pages/metaRoutingPageTypes.ts dashboard/frontend/e2e/layout-nav.spec.ts src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_api.go src/semantic-router/pkg/extproc/meta_routing_feedback_aggregate.go src/semantic-router/pkg/extproc/meta_routing_feedback_api_test.go src/semantic-router/pkg/extproc/processor_req_header.go src/semantic-router/pkg/extproc/processor_req_header_validation.go docs/agent/adr/adr-0004-meta-routing-dashboard-ui-surface.md docs/agent/plans/pl-0016-meta-routing-dashboard-ui-workstream.md"`
  - `cd dashboard/frontend && npx playwright test e2e/layout-nav.spec.ts --reporter=line`
  - `make agent-validate`
  - `make agent-lint CHANGED_FILES="dashboard/backend/router/proxy_routes.go dashboard/frontend/src/App.tsx dashboard/frontend/src/components/ConfigNav.tsx dashboard/frontend/src/components/LayoutNavSupport.ts dashboard/frontend/src/pages/ConfigPage.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.module.css dashboard/frontend/src/pages/configPageMetaRoutingSupport.tsx dashboard/frontend/src/pages/InsightsPage.tsx dashboard/frontend/src/pages/insightsPageSupport.tsx dashboard/frontend/src/pages/MetaRoutingPage.tsx dashboard/frontend/src/pages/MetaRoutingPage.module.css dashboard/frontend/src/pages/metaRoutingPageSupport.tsx dashboard/frontend/src/pages/metaRoutingPageTypes.ts dashboard/frontend/e2e/layout-nav.spec.ts src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_api.go src/semantic-router/pkg/extproc/meta_routing_feedback_aggregate.go src/semantic-router/pkg/extproc/meta_routing_feedback_api_test.go src/semantic-router/pkg/extproc/processor_req_header.go src/semantic-router/pkg/extproc/processor_req_header_validation.go docs/agent/adr/adr-0004-meta-routing-dashboard-ui-surface.md docs/agent/plans/pl-0016-meta-routing-dashboard-ui-workstream.md config/README.md website/docs/installation/configuration.md"`
  - `make agent-ci-gate CHANGED_FILES="dashboard/backend/router/proxy_routes.go dashboard/frontend/src/App.tsx dashboard/frontend/src/components/ConfigNav.tsx dashboard/frontend/src/components/LayoutNavSupport.ts dashboard/frontend/src/pages/ConfigPage.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.module.css dashboard/frontend/src/pages/configPageMetaRoutingSupport.tsx dashboard/frontend/src/pages/InsightsPage.tsx dashboard/frontend/src/pages/insightsPageSupport.tsx dashboard/frontend/src/pages/MetaRoutingPage.tsx dashboard/frontend/src/pages/MetaRoutingPage.module.css dashboard/frontend/src/pages/metaRoutingPageSupport.tsx dashboard/frontend/src/pages/metaRoutingPageTypes.ts dashboard/frontend/e2e/layout-nav.spec.ts src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_api.go src/semantic-router/pkg/extproc/meta_routing_feedback_aggregate.go src/semantic-router/pkg/extproc/meta_routing_feedback_api_test.go src/semantic-router/pkg/extproc/processor_req_header.go src/semantic-router/pkg/extproc/processor_req_header_validation.go docs/agent/adr/adr-0004-meta-routing-dashboard-ui-surface.md docs/agent/plans/pl-0016-meta-routing-dashboard-ui-workstream.md config/README.md website/docs/installation/configuration.md"`
  - `make agent-feature-gate ENV=cpu CHANGED_FILES="dashboard/backend/router/proxy_routes.go dashboard/frontend/src/App.tsx dashboard/frontend/src/components/ConfigNav.tsx dashboard/frontend/src/components/LayoutNavSupport.ts dashboard/frontend/src/pages/ConfigPage.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.tsx dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.module.css dashboard/frontend/src/pages/configPageMetaRoutingSupport.tsx dashboard/frontend/src/pages/InsightsPage.tsx dashboard/frontend/src/pages/insightsPageSupport.tsx dashboard/frontend/src/pages/MetaRoutingPage.tsx dashboard/frontend/src/pages/MetaRoutingPage.module.css dashboard/frontend/src/pages/metaRoutingPageSupport.tsx dashboard/frontend/src/pages/metaRoutingPageTypes.ts dashboard/frontend/e2e/layout-nav.spec.ts src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_api.go src/semantic-router/pkg/extproc/meta_routing_feedback_aggregate.go src/semantic-router/pkg/extproc/meta_routing_feedback_api_test.go src/semantic-router/pkg/extproc/processor_req_header.go src/semantic-router/pkg/extproc/processor_req_header_validation.go docs/agent/adr/adr-0004-meta-routing-dashboard-ui-surface.md docs/agent/plans/pl-0016-meta-routing-dashboard-ui-workstream.md config/README.md website/docs/installation/configuration.md"`
  - Completed on 2026-03-24: router feedback APIs, dashboard authoring and operations UI, Insights cross-links, focused Playwright coverage, and the full validation ladder all passed. No additional dashboard-specific debt item was needed in this workstream.

## Decision Log

- 2026-03-24: A usable dashboard meta-routing experience requires a dedicated operations page. The existing replay Insights surface is not the primary owner for meta-routing analytics.
- 2026-03-24: Meta-routing authoring should land first in the Config page, not the Builder visual mode. Builder raw DSL remains an escape hatch, but not the primary product path.
- 2026-03-24: The router remains the source of truth for persisted `FeedbackRecord` storage and query semantics. Dashboard backend should proxy the router API instead of creating a second record store.
- 2026-03-24: The first dashboard workstream should ship list, aggregate, and detail inspection before any interactive probe or playground-style experimentation surface.
- 2026-03-24: Cross-links into replay are desirable, but replay economics and meta-routing quality remain separate page ownership boundaries.

## Follow-up Debt / ADR Links

- [ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [ADR 0004: Expose Meta-Routing Through Dedicated Dashboard Authoring and Operations Surfaces](../adr/adr-0004-meta-routing-dashboard-ui-surface.md)
- [pl-0015-meta-routing-refinement-workstream.md](pl-0015-meta-routing-refinement-workstream.md)
