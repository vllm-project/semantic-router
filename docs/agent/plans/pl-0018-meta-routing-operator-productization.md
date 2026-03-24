# Meta-Routing Operator Productization Workstream

## Goal

- Make meta-routing understandable and demonstrable to non-expert operators through clearer dashboard UX and clearer rollout semantics.
- Move canonical meta-routing feedback read APIs from the extproc or Envoy surface to the router APIServer.
- Close this workstream only when operators can configure the feature, see whether it has effect, and explain that effect without reading implementation code.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/extproc/**`
- `dashboard/backend/router/**`
- `dashboard/frontend/src/pages/**`
- `dashboard/frontend/src/components/**`
- targeted router, dashboard, and documentation validation
- nearest local rules for `src/semantic-router/pkg/extproc/`, `dashboard/frontend/src/`, `dashboard/frontend/src/pages/`, and `dashboard/backend/handlers/` when implementation starts

## Exit Criteria

- One indexed ADR records that meta-routing productization consists of APIServer-owned feedback reads plus operator-oriented dashboard UX.
- The canonical read API for meta-routing feedback is available on the router APIServer under `/api/v1/feedback/meta-routing`, `/aggregate`, and `/{id}`.
- Dashboard backend no longer needs Envoy-only routing exceptions for canonical meta-routing feedback reads.
- Dashboard meta-routing surfaces explain the three modes in plain language and clearly distinguish record-only, shadow counterfactuals, and active adopted changes.
- Aggregate and detail views expose effect-focused metrics, including counterfactual changes for `shadow` and adopted changes for `active`.
- Operator-facing views expose the active provider or artifact lineage, promotion-gate state, and rollback guidance.
- Dashboard and docs clearly distinguish routing-config changes from meta-routing rollout changes.
- Empty states and onboarding copy make it possible to explain and demo the feature without reading source code.
- The active plan backlog also records the next bounded action-expansion candidates so later loops can implement stronger refinement behavior without rediscovering the design space.
- Compatibility coverage exists for any deprecated extproc feedback routes until the migration window is intentionally closed.

## Task List

- [x] `MRP001` Add an indexed ADR and execution plan for meta-routing operator productization before implementation starts.
- [ ] `MRP002` Extract shared meta-routing feedback query and aggregation support so APIServer can read the same persisted records without duplicating logic.
- [ ] `MRP003` Add APIServer read handlers for list, aggregate, and detail under the canonical `/api/v1/feedback/meta-routing` route family.
- [ ] `MRP004` Add a compatibility bridge or deprecation path for the existing extproc or Envoy feedback read endpoints and document the migration contract.
- [ ] `MRP005` Switch dashboard backend proxying and frontend fetch helpers to the APIServer-backed canonical routes.
- [ ] `MRP006` Redesign the dashboard meta-routing analysis page around operator questions, including counterfactual and adopted change metrics.
- [ ] `MRP007` Improve the dashboard config surface with rollout guidance, plain-language mode descriptions, and stronger empty or prerequisite states.
- [ ] `MRP008` Add focused tests and docs updates covering APIServer ownership, dashboard behavior, and rollout semantics.
- [ ] `MRP009` Extend the bounded refinement action backlog with `expand_input_window` or `disable_truncation` so long requests can recover missing detail without widening the public `routing.meta` contract immediately.
- [ ] `MRP010` Extend the bounded refinement action backlog with `upgrade_signal_tier` so a fragile pass can escalate from cheaper signal families or backends to more expensive semantic evaluation only when needed.
- [ ] `MRP011` Extend the bounded refinement action backlog with `rerank_top_decisions` so low-margin top candidates can be re-compared without always re-running the full routing graph.
- [ ] `MRP012` Extend the bounded refinement action backlog with `force_verification_overlay_check` so verified-versus-non-verified overlays can be re-evaluated explicitly on refinement-sensitive domains.
- [ ] `MRP013` Extend fragility scoring with candidate-distribution features such as top-k entropy and winner concentration so brittle requests are not defined by winner-versus-runner-up margin alone.
- [ ] `MRP014` Extend fragility scoring with evidence-structure features such as evidence diversity, single-family dominance, and support-versus-contradiction balance so the router can tell the difference between broad multi-family support and narrow one-family wins.
- [ ] `MRP015` Extend fragility scoring with leave-one-family-out sensitivity checks so requests can be marked unstable when removing one signal family would flip the route.
- [ ] `MRP016` Extend fragility scoring with input-robustness features such as compression delta, truncation pressure, and representation sensitivity so the router can detect requests whose route depends too heavily on input preprocessing shape.
- [ ] `MRP017` Expose provider lineage, acceptance-gate state, and rollback recommendation on aggregate and detail surfaces so operators can explain why a learned or calibrated policy is currently trusted.
- [ ] `MRP018` Distinguish meta-routing rollouts from routing-config changes across config, analysis, and docs surfaces so operators do not confuse recipe edits with assess-and-refine behavior changes.

## Current Loop

- Date: 2026-03-24
- Current task: `MRP001` completed
- Branch: `meta-routing`
- Planned loop order:
  - `L1` lock the ownership and UX boundary in ADR and plan form
  - `L2` move feedback query ownership behind an APIServer-readable seam
- `L3` switch dashboard transport to the APIServer contract
- `L4` improve dashboard authoring and analysis UX around operator-facing proof of effect
- `L5` record the next bounded action-expansion backlog in the active plan so later loops can implement it deliberately
- `L6` record the next fragility-score expansion backlog in the active plan so later loops can implement higher-signal brittle-request detection deliberately
- `L7` expose provider-lineage and rollout-gate context so promotion and rollback are explainable
- `L8` distinguish meta-routing rollouts from routing-config changes across the operator workflow
- `L9` run focused validation and close the workstream when the feature is demo-friendly
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/README.md`, `docs/agent/governance.md`, `docs/agent/adr/README.md`, and `docs/agent/plans/README.md`
  - broad `codebase-retrieval` across governance docs, existing meta-routing ADRs or plans, extproc runtime, APIServer routes, dashboard transport, dashboard pages, and manifest indexes
  - targeted source reads for `src/semantic-router/pkg/extproc/meta_routing_runtime.go`, `src/semantic-router/pkg/extproc/meta_routing_feedback_api.go`, `src/semantic-router/pkg/extproc/processor_req_header.go`, `src/semantic-router/pkg/apiserver/server.go`, `dashboard/backend/router/proxy_routes.go`, `dashboard/frontend/src/pages/MetaRoutingPage.tsx`, and `dashboard/frontend/src/pages/ConfigPageMetaRoutingSection.tsx`
  - docs-only governance validation will be rerun after this plan and ADR are indexed
- This plan is intentionally not started beyond governance setup. No productization implementation work is in scope for this loop.

## Decision Log

- 2026-03-24: The next meta-routing increment is treated as productization, not more hidden runtime work.
- 2026-03-24: Canonical meta-routing feedback reads should be APIServer-owned because they are control-plane observability APIs, even though request-time feedback emission remains in extproc.
- 2026-03-24: The operator-facing dashboard must optimize for proof of effect, especially in `shadow`, instead of exposing implementation artifacts only.
- 2026-03-24: Route naming should align with existing APIServer conventions, so the canonical family will live under `/api/v1/feedback/meta-routing` rather than under a raw `/v1/meta_routing_feedback` path.
- 2026-03-24: Operator productization also needs provider-lineage, promotion-gate, and rollback evidence so the feature can be used for real rollout decisions, not only demos.
- 2026-03-24: Product surfaces must distinguish meta-routing behavior rollouts from route-graph edits or operators will not know what actually changed.
- 2026-03-24: The next bounded refinement-action candidates are `expand_input_window`, `upgrade_signal_tier`, `rerank_top_decisions`, and `force_verification_overlay_check`; they are intentionally recorded as follow-on work rather than folded into the current productization loop.
- 2026-03-24: The next high-value brittle-request scoring candidates are candidate-distribution, evidence-structure, leave-one-family-out sensitivity, and input-robustness features; they should be treated as explainable sensitivity analysis before any heavier learned trigger path.

## Follow-up Debt / ADR Links

- [ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [ADR 0004: Expose Meta-Routing Through Dedicated Dashboard Authoring and Operations Surfaces](../adr/adr-0004-meta-routing-dashboard-ui-surface.md)
- [ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [ADR 0006: Productize Meta-Routing Through Operator-Oriented UX and APIServer-Owned Feedback APIs](../adr/adr-0006-meta-routing-operator-ux-and-apiserver-feedback.md)
- [ADR 0008: Separate Routing Graph Optimization From Meta-Routing Reliability Improvement](../adr/adr-0008-routing-graph-optimization-separate-from-meta-routing.md)
- [pl-0015-meta-routing-refinement-workstream.md](pl-0015-meta-routing-refinement-workstream.md)
- [pl-0020-routing-graph-quality-and-optimization-workstream.md](pl-0020-routing-graph-quality-and-optimization-workstream.md)
- [pl-0016-meta-routing-dashboard-ui-workstream.md](pl-0016-meta-routing-dashboard-ui-workstream.md)
- [pl-0017-meta-routing-learned-policy-enablement.md](pl-0017-meta-routing-learned-policy-enablement.md)
