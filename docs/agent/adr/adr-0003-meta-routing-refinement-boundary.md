# ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam

## Status

Accepted

## Context

The current routing path is a one-pass request pipeline:

- signal extraction
- projection derivation
- decision evaluation
- model selection

This keeps the signal, projection, decision, and selection layers clean, but it leaves the repository without a repo-native place to assess whether the intermediate routing state is reliable enough before finalizing the route.

The new meta-routing workstream needs a durable boundary for:

- optional observe, shadow, and active rollout modes
- deterministic assessment of low-quality intermediate routing state
- targeted refinement through selective subgraph recomputation rather than whole-pipeline retries
- pass-level tracing and durable feedback records for later calibration and learned policies

The repository also needs that boundary to stay compatible with existing layer ownership. Signal packages should keep extracting facts. Projection packages should keep deriving post-signal state. Decision packages should keep evaluating boolean routing rules. Model selection should keep choosing among candidates. None of those layers should absorb retry policy, trigger logic, or feedback persistence.

## Decision

Introduce meta-routing as a request-phase orchestration seam above the existing routing pipeline.

The durable architecture decision is:

- the existing one-pass routing flow remains the base pass
- meta-routing wraps that base pass with assess-and-refine control flow
- refinement is defined as targeted recomputation of affected routing subgraphs, not free-form agentic loops and not token-level regeneration
- the new public config surface is an optional `routing.meta` contract with only four concerns:
  - rollout mode
  - maximum pass budget
  - trigger policy
  - allowed refinement actions
- v1 runtime artifacts are stable named types:
  - `RoutingTrace`
  - `PassTrace`
  - `MetaAssessment`
  - `RefinementPlan`
  - `FeedbackRecord`
- `PassTrace` owns the pass-quality view needed for later calibration:
  - `signal_dominance`
  - `avg_signal_confidence`
  - `decision_margin`
  - `projection_boundary_min_distance`
  - `fragile`
- `MetaAssessment` is not just a boolean gate:
  - it records `needs_refine`
  - it records `triggers`
  - it records `root_causes`
  - it carries the pass `trace_quality`
- `FeedbackRecord` follows an observation, action, outcome envelope so later replay and learned-policy work can reuse the runtime schema instead of replacing it
- the runtime is split into narrow collaborators owned by the request-phase seam:
  - `BasePassRunner`
  - `MetaAssessor`
  - `RefinementPlanner`
  - `RefinementExecutor`
  - `PolicyProvider`
  - `FeedbackSink`
- the runtime flow is fixed to:
  - base pass
  - meta assessment
  - optional targeted refinement
  - final decision and model selection
  - feedback emission
- signal, projection, and decision packages remain pure with respect to meta-routing:
  - signal evaluation may be re-run for selected families, but the signal layer does not know why
  - projection recomputation is limited to affected derived outputs
  - the decision engine does not own retry logic, trigger policy, or feedback policy
- v1 trigger policy is deterministic and limited to:
  - low decision margin
  - projection score near mapping boundary
  - conflicting partition evidence
  - required family missing or low-confidence
  - disagreement between cheap and expensive signal families
- v1 refinement actions are narrow and explicit:
  - re-run selected high-cost semantic signal families
  - re-evaluate with less lossy input preparation
  - recompute affected projections
  - rerun decision and model selection
- meta outputs are orchestration-only in v1:
  - they are visible to logs, metrics, and feedback records
  - they are not directly referenceable from decision rules
- the v1 root-cause taxonomy is adapted to routing runtime rather than token search:
  - `missing_required_family`
  - `low_confidence_family`
  - `projection_boundary_pressure`
  - `decision_overlap`
  - `partition_conflict`
  - `family_disagreement`
  - `compression_loss_risk`
- backward compatibility is mandatory:
  - `routing.meta` is optional
  - unset config remains current one-pass routing behavior

## Consequences

- The repository gains one durable seam for routing reliability control without collapsing existing signal, projection, decision, and selection boundaries into a single hotspot.
- Cross-surface contract work is required across config, canonical import/export, DSL, CLI, dashboard, maintained config assets, and docs because `routing.meta` becomes part of the public routing contract.
- Request-phase telemetry must become pass-aware. The repo will record pass counts, trigger names, refined families, overturned decisions, boundary deltas, latency deltas, and final outcome summaries instead of adding more one-off booleans.
- Search-R2-inspired pass quality and root-cause semantics are now part of the runtime boundary, not post-hoc analysis only. That keeps future calibration and learned-policy work aligned with the same request-time artifacts.
- Runtime changes can ship incrementally. `observe` and `shadow` modes can land before `active`, while all three reuse the same trace, policy, and feedback interfaces.
- The repository gains a durable offline data lane for later calibration and learned policies without forcing the first runtime increment to depend on model training.
- Learned `PolicyProvider` enablement now has its own accepted follow-on contract in [ADR 0005](../adr/adr-0005-meta-routing-learned-policy-contract.md), so calibrated or learned providers can evolve behind the same seam without widening the public `routing.meta` surface.
- The decision also constrains future work: if later refinement needs free-form agentic behavior, that must be a new explicit architecture decision rather than an accidental expansion of the v1 meta-routing seam.
