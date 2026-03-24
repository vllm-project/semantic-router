# ADR 0004: Expose Meta-Routing Through Dedicated Dashboard Authoring and Operations Surfaces

## Status

Accepted

## Context

The repository now has a real `routing.meta` contract, request-phase meta-routing runtime behavior, pass-level trace artifacts, and durable `FeedbackRecord` persistence. That closes the backend routing seam, but it does not yet give dashboard users a complete product workflow.

The current dashboard state is incomplete for meta-routing:

- the config and DSL type layers understand `routing.meta`
- the Config page does not expose a dedicated visual section for meta-routing authoring
- the Builder page does not provide first-class visual CRUD for `META`
- the Insights page is built around `router_replay`, not `FeedbackRecord`
- there is no dedicated route for rollout health, trace inspection, or base-versus-refined comparison
- there is no router or dashboard HTTP surface for listing, aggregating, and inspecting persisted meta-routing feedback records

That leaves users with a fragmented workflow:

- configure `routing.meta` by hand in YAML or raw DSL
- send traffic through the router
- inspect logs or offline scripts to understand what happened

That is not a usable dashboard product surface. The repository needs one durable decision for where meta-routing authoring lives, where runtime inspection lives, which API owns the record query surface, and how to avoid growing existing dashboard hotspots into another mixed-responsibility slice.

The decision must also stay compatible with existing local rules:

- `ConfigPage.tsx` is already a legacy hotspot and should not absorb more schema or rendering logic directly
- `InsightsPage.tsx` is a replay-oriented page and should not become the owner of a second observability domain with different semantics
- `BuilderPage.tsx` is a ratcheted hotspot and should not become the first or only visual owner of meta-routing authoring
- dashboard backend route registration can grow narrowly, but handler transport, record querying, and aggregation logic need separate seams

## Decision

Expose meta-routing to dashboard users through two primary surfaces plus one lightweight cross-link pattern:

- a dedicated Config authoring section for `routing.meta`
- a dedicated operations page for `FeedbackRecord` inspection and rollout health
- lightweight cross-links from nearby dashboard surfaces instead of merging ownership into them

The durable architecture decision is:

- the primary authoring surface for meta-routing is the dashboard Config flow, not the Builder visual mode
- the primary runtime inspection surface is a new dedicated dashboard route, `/meta-routing`
- the source of truth for runtime inspection is the router-owned persisted `FeedbackRecord`, not a second dashboard-owned state store
- dashboard backend will proxy router-owned meta-routing feedback APIs the same way it already proxies router replay APIs
- Insights remains replay-centric; it may link to meta-routing records, but it does not become the primary owner of meta-routing analytics
- Builder raw DSL remains an escape hatch, but first-class visual `META` editing is deferred until the dedicated Config and operations surfaces are complete

### Dashboard Page Model

The complete user workflow is split across these dashboard surfaces:

- `Config -> Meta Routing`
  - visually author `routing.meta.mode`, `max_passes`, `trigger_policy`, and `allowed_actions`
  - review rollout notes and persistence prerequisites before enabling `shadow` or `active`
- `Meta Routing`
  - inspect rollout health across recent requests
  - filter, search, and compare feedback records
  - open one record and understand base pass, refinement pass, triggers, root causes, planned actions, executed actions, and final outcome
- `Insights`
  - stays focused on replay cost and route selection economics
  - may show contextual links into matching meta-routing records when `router_replay_id` is present

The new `Meta Routing` page is the primary user-visible runtime surface. It is composed from narrow support modules, not a large page-local monolith.

### Config Authoring Surface

The dashboard must add a dedicated Config section for meta-routing with support modules adjacent to `ConfigPage`, not inline expansion inside the hotspot page.

The visual authoring surface should provide these cards:

- `Mode & Budget`
  - mode selector for `observe`, `shadow`, `active`
  - `max_passes`
  - concise behavioral summary for the selected mode
- `Trigger Policy`
  - decision-margin threshold
  - projection-boundary threshold
  - partition-conflict toggle
- `Required Families`
  - editable required family rows with `type`, `min_confidence`, and `min_matches`
- `Family Disagreements`
  - editable cheap-versus-expensive family pairs
- `Allowed Actions`
  - `disable_compression`
  - `rerun_signal_families` plus signal-family selector
- `Rollout Notes`
  - recorder and persistence prerequisites
  - observe-to-shadow-to-active operator guidance

This section edits the existing canonical config contract through the existing config write path. It does not create a second special-purpose write API.

### Operations Surface

The new `/meta-routing` route is the operational home for this feature. It must be distinct from `Insights` because the source data, aggregates, and operator questions are different.

The page should include:

- overview hero with current mode, record count, refinement rate, overturn rate, and latency overhead summary
- summary cards for:
  - total observed requests
  - planned refinement rate
  - executed refinement rate
  - overturned decision rate
  - average and p95 latency delta
  - top trigger
  - top root cause
- aggregate charts for:
  - mode distribution
  - trigger distribution
  - root-cause distribution
  - action-type distribution
  - refined-family distribution
  - decision-change or model-change distribution
  - latency-delta histogram
- activity table with filters for:
  - mode
  - trigger
  - root cause
  - action type
  - refined family
  - overturned or not
  - decision
  - model
  - response status
  - free-text search over request identifiers, model, decision, and query text when available
- detail drawer or modal for one record with:
  - request and final outcome summary
  - base pass card
  - refinement pass card
  - side-by-side compare view
  - trace-quality metrics
  - assessment triggers and root causes
  - refinement plan and executed actions
  - downstream weak labels
  - link to matching replay record when `router_replay_id` exists

### API Ownership and Contract

The record query APIs are router-owned because the router owns persistence for `FeedbackRecord`.

The durable API ownership model is:

- router exposes:
  - `GET /v1/meta_routing_feedback`
  - `GET /v1/meta_routing_feedback/aggregate`
  - `GET /v1/meta_routing_feedback/{id}`
- dashboard backend proxies:
  - `GET /api/router/v1/meta_routing_feedback`
  - `GET /api/router/v1/meta_routing_feedback/aggregate`
  - `GET /api/router/v1/meta_routing_feedback/{id}`
- frontend fetches only through the dashboard proxy surface

The initial query contract should mirror the router replay ergonomics:

- pagination: `limit`, `offset`
- search: `search`
- filters:
  - `mode`
  - `trigger`
  - `root_cause`
  - `action_type`
  - `signal_family`
  - `overturned`
  - `decision`
  - `model`
  - `response_status`

The list endpoint returns summary records, not full trace payloads, so the table stays lightweight.

The aggregate endpoint returns:

- record count
- planned and executed refinement rates
- overturn rate
- average and p95 latency delta
- top trigger and top root cause
- chart-ready distributions
- available filter options

The detail endpoint returns one full `FeedbackRecord` plus recorder metadata such as record id and timestamp.

### Composition and Boundary Rules

To stay compatible with current local rules:

- `ConfigPage.tsx` only wires a new `ConfigPageMetaRoutingSection`
- `MetaRoutingPage.tsx` owns route-level fetch cadence, filters, and modal state only
- charts, table column shaping, summary-card shaping, and detail-section rendering live in adjacent support modules
- dashboard backend route registration stays thin in `core_routes.go`
- handler transport, record loading, filtering, and aggregation live in dedicated support modules instead of growing existing config or status handlers

### Explicit Non-Goals for the First Dashboard Workstream

- no dashboard-owned duplicate database or cache for meta-routing records
- no attempt to merge meta-routing analytics into the existing replay Insights page
- no requirement to add first-class Builder visual CRUD for `META` before the dedicated Config and operations surfaces exist
- no free-form playground or synthetic probe workflow in the first increment unless a later plan explicitly adds a safe request-inspection endpoint

## Consequences

- Dashboard users get one end-to-end product flow for meta-routing:
  - author policy
  - deploy or save config
  - observe runtime behavior
  - inspect individual records
  - decide whether to move from `observe` to `shadow` to `active`
- The repository needs new router-side read APIs over persisted `FeedbackRecord` data, but it does not need a second persistence owner.
- The dashboard backend remains a transport and composition layer. It proxies the new router API and keeps handler logic narrow instead of absorbing a second observability database.
- `ConfigPage.tsx`, `InsightsPage.tsx`, and `BuilderPage.tsx` avoid another round of responsibility collapse.
- The operations surface becomes explicitly meta-routing-specific, which makes later additions such as shadow-versus-active comparisons, rollout scorecards, or learned-policy calibration views easier to add without distorting replay analytics.
- Builder visual authoring for `META` remains a known follow-on, but it is not a blocker for a usable dashboard product surface because Config authoring becomes the primary supported path.
