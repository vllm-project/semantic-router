# ADR 0006: Productize Meta-Routing Through Operator-Oriented UX and APIServer-Owned Feedback APIs

## Status

Accepted

## Context

The repository already has a working meta-routing seam, bounded refinement behavior, persisted `FeedbackRecord` data, and dashboard surfaces for config authoring and record inspection. That closes the initial implementation loop, but it still leaves a product gap for real operators and demos.

The current state is functionally correct but awkward to explain and operate:

- the three rollout modes are named with internal engineering terms instead of user-oriented behavior
- the dashboard does not clearly answer the operator question "what changed because meta-routing was enabled?"
- the current aggregate view reports planned and executed refinement, but it does not foreground the "would change" versus "did change" distinction that matters during rollout review
- the detail view exposes internal artifacts, but the empty states and summaries do not teach first-time users what to look for
- request-phase feedback APIs are currently mounted on the extproc or Envoy-facing surface rather than the router APIServer
- dashboard backend therefore proxies meta-routing reads to Envoy, which makes the topology harder to explain and mixes data-plane and control-plane concerns
- the current path shape, `/v1/meta_routing_feedback`, is implementation-specific and does not fit the existing APIServer naming style

That leaves the repository with a feature that exists, but is harder than necessary to demo, teach, or operate confidently.

The next durable decision needs to answer two boundary questions:

- what should the operator-facing UX emphasize so meta-routing is understandable to non-experts
- which service should own the canonical read APIs for persisted meta-routing feedback records

The answer must stay compatible with the accepted meta-routing seam:

- extproc remains the request-time owner of assess, plan, execute, and emit behavior
- the APIServer remains the natural HTTP control-plane surface for router introspection and management
- dashboard stays a proxy and presentation layer instead of becoming a second persistence owner

## Decision

Treat the next meta-routing increment as a productization phase with two durable decisions:

- the canonical read API for persisted meta-routing feedback moves to the router APIServer
- the dashboard meta-routing UX shifts from implementation-centric inspection to operator-centric rollout guidance

### API Ownership

The canonical feedback read APIs will be APIServer-owned.

The durable ownership model is:

- extproc continues to emit and persist `FeedbackRecord` data at request time
- record loading, filtering, and aggregation move behind a shared router-owned support seam that can be consumed by both extproc and APIServer during migration
- the router APIServer becomes the canonical HTTP owner for meta-routing feedback reads
- dashboard backend proxies the canonical APIServer routes instead of special-casing Envoy for this feature
- the existing extproc or Envoy feedback routes remain as a compatibility bridge during migration, then become deprecated compatibility aliases rather than the primary contract

The canonical APIServer route family will align with existing control-plane naming:

- `GET /api/v1/feedback/meta-routing`
- `GET /api/v1/feedback/meta-routing/aggregate`
- `GET /api/v1/feedback/meta-routing/{id}`

The repository does not collapse this into the existing `/api/v1/feedback` write API for model-selection feedback. Meta-routing feedback is a read-only observability domain with different semantics, so it remains a distinct sub-resource.

### Operator-Oriented UX

The canonical dashboard UX keeps the existing `Config -> Meta Routing` and `/meta-routing` locations, but changes what they optimize for.

The durable UX direction is:

- user-facing copy explains behavior first and internal terminology second
- rollout modes are always shown with plain-language descriptions:
  - `observe` means record only
  - `shadow` means run refinement but keep the base result
  - `active` means allow the refined result to replace the base result
- the main analysis surface emphasizes proof of effect, not just artifact presence
- the dashboard must make it obvious whether meta-routing is:
  - planning refinements
  - executing refinements
  - changing outcomes counterfactually in `shadow`
  - changing adopted outcomes in `active`
  - adding latency

The `/meta-routing` page becomes an operator workflow, not just a record browser. It should foreground:

- whether the feature is producing plans at all
- whether those plans execute
- whether they would have changed the route or model in `shadow`
- whether they actually changed the adopted route or model in `active`
- what the latency cost is
- which triggers and root causes dominate
- which provider or artifact lineage produced the current behavior
- whether current evidence supports promotion or rollback

The record detail view should always explain the pass pair in human terms:

- what the base pass selected
- why it was judged brittle
- what the refinement plan did
- what the refined pass selected
- whether the refined outcome was only observed, only shadowed, or actually adopted

The config surface should guide rollout in order:

- start with record only
- verify the feature is generating meaningful opportunities
- move to shadow and inspect would-change behavior
- move to active only after the operator has evidence that changes are useful and bounded

The productized UX must also distinguish two different kinds of change:

- routing-config changes
  - edits to YAML or DSL route definitions, thresholds, projections, or model references
- meta-routing rollout changes
  - changes in `observe`, `shadow`, or `active` behavior
  - or changes in the internal policy artifact behind the same public `routing.meta` contract

Operators should not have to reverse-engineer which kind of change they are
looking at from raw traces alone.

### Metrics and Semantics

The productized dashboard and API vocabulary must distinguish:

- `planned_refinement_rate`
- `executed_refinement_rate`
- `counterfactual_change_rate`
- `adopted_change_rate`
- `average_latency_delta_ms`
- `p95_latency_delta_ms`
- `provider_identity`
- `artifact_version`
- `acceptance_status`
- `rollback_recommendation`

`counterfactual_change_rate` is especially important in `shadow`, where the system runs refinement but intentionally keeps the base pass. The repository should expose that rate explicitly instead of forcing operators to infer it from low-level traces.

### Migration and Compatibility

The migration is staged:

- first introduce shared feedback-query support and APIServer read handlers
- then switch dashboard backend and frontend fetches to the APIServer-backed routes
- keep extproc feedback routes as deprecated compatibility aliases during rollout
- only retire the Envoy-facing read routes after dashboard, tests, docs, and deployment recipes no longer depend on them

## Consequences

- Operators and demos gain a feature that can be explained in user terms instead of internal runtime vocabulary only.
- The repository separates request-time data production from control-plane data access more cleanly.
- Dashboard proxy logic becomes simpler because meta-routing feedback no longer needs a special Envoy transport exception once migration completes.
- APIServer route naming becomes more coherent with the rest of the router control-plane surface.
- The repository must add a shared feedback-query seam so APIServer can serve the same record inventory without duplicating storage or aggregation logic.
- The aggregate contract must grow a counterfactual-change concept, especially for `shadow`, so the UI can show meaningful effect before `active` is enabled.
- Extproc compatibility routes will need a deprecation period to avoid breaking existing users and scripts.
- Operator-facing product surfaces now also need provider-lineage and rollout-gate context, not just request records, so promotion and rollback decisions are explainable.
- Dashboard and API surfaces must distinguish meta-routing rollouts from route-graph edits or the feature will remain hard to teach and debug.
