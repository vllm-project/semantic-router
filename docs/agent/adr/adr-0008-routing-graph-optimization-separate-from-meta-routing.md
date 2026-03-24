# ADR 0008: Separate Routing Graph Optimization From Meta-Routing Reliability Improvement

## Status

Accepted

## Context

The repository now has a durable meta-routing seam for request-time reliability
control and a separate offline learning direction for improving trigger, action,
and adoption behavior.

That progress creates a new risk: the project could start treating meta-routing
as the place to hide route-graph defects instead of diagnosing and fixing them.

That would blur several different problems:

- request-time reliability on top of an existing route graph
- long-term learning of when to refine and when to adopt refined outcomes
- route-graph quality itself, including overlapping rules, poor thresholds, and missing route structure

Meta-routing can detect many of those weaknesses indirectly:

- the same route pairs frequently overturn in `shadow`
- the same projection boundaries appear in fragile traces
- the same slices repeatedly need refinement to avoid weak outcomes

But that does not mean request-time meta-routing should rewrite YAML or DSL
route definitions automatically.

The repository needs one durable boundary for where route-graph optimization
belongs and how it should relate to meta-routing evidence.

## Decision

Route-graph optimization is a separate workflow from meta-routing reliability
improvement.

The durable boundary is:

- meta-routing owns request-time reliability assessment, bounded refinement, and provider-backed trigger or action overlays
- route-graph optimization owns evaluating whether the underlying signals, projections, decisions, thresholds, overlays, or maintained recipes are structurally well designed
- meta-routing traces, feedback records, replay outputs, and maintained probes are valid evidence sources for route-graph diagnosis
- route-graph optimization does not happen by mutating config from the request path or by silently expanding the `PolicyProvider` seam

### Evidence Handoff

The repository treats the following as primary inputs to route-graph diagnosis:

- route-pair confusion or overturn summaries
- persistent fragile slices
- projection-boundary hotspots
- root-cause concentration by route or slice
- maintained probe failures and near-miss matches
- action-effectiveness patterns that suggest the graph, not only the trigger policy, needs redesign

### Optimization Outputs

The first-class outputs of the route-graph workflow are human-reviewable
diagnostics and recommendations, for example:

- route overlap analysis
- confusion matrices between competing decisions
- threshold or boundary hotspot reports
- missing-route or over-broad-route candidates
- maintained recipe quality reviews
- probe-coverage or slice-coverage gaps

Those outputs inform normal config, DSL, and maintained-asset change review.
They do not become request-time self-modifying routing behavior.

### Contract Boundary

Changing the route graph remains a config-governed change:

- YAML, DSL, or maintained recipe edits
- review and validation in the normal repository workflow
- explicit deployment and rollback like other routing-policy changes

Meta-routing rollout remains separate:

- `observe`, `shadow`, `active`
- or internal policy-artifact promotion behind the same public `routing.meta` contract

The repository will not describe route-graph optimization as just another
meta-routing mode.

## Consequences

- The project can use meta-routing evidence to diagnose structural route issues without turning the request path into a self-modifying graph optimizer.
- Operator-facing explanations become cleaner because “meta-routing rollout” and “routing-config change” are treated as different workflows.
- A separate workstream is now justified for route overlap, maintained recipe quality, and probe-driven graph improvement.
- Meta-routing stays focused on bounded reliability control instead of becoming a catch-all layer for every routing weakness.
