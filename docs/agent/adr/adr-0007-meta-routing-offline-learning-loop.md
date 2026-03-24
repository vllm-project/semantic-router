# ADR 0007: Evolve Meta-Routing Through an Offline-First Learning Loop and Artifact Promotion

## Status

Accepted

## Context

The repository now has:

- a deterministic request-phase meta-routing seam
- bounded refinement actions
- durable `FeedbackRecord` persistence
- an internal `PolicyProvider` seam
- an artifact contract for calibrated or learned policy overlays
- dashboard and documentation surfaces for `observe`, `shadow`, and `active`

That is enough to run meta-routing safely, but it still leaves the long-term improvement loop underspecified.

The open question is not whether meta-routing can collect evidence. It already can. The open question is how the repository should improve trigger, action, and adoption behavior over time without turning request routing into an uncontrolled online-learning system.

That question matters because a naive design would create the wrong incentives:

- online request-path policy mutation would be hard to explain, reproduce, or roll back
- a single monolithic learned policy would entangle different decisions with different risk profiles
- a reward defined only by route change would ignore latency cost and downstream weak labels
- request-time RL inference inside extproc would widen the runtime seam far beyond the current bounded design

The repository needs one durable decision for how continual improvement should work:

- where learning runs
- how policy problems are decomposed
- which data model is canonical
- how rollout safety is preserved
- how request-time evidence, probe evaluation, and promoted artifacts connect end to end

## Decision

Meta-routing will evolve through an offline-first learning loop, not through request-path online self-modification.

The durable architecture decision is:

- request-time runtime remains responsible only for:
  - base-pass execution
  - assessment
  - bounded planning
  - bounded refinement execution
  - final pass adoption
  - feedback emission
- all learning, calibration, and policy fitting happen outside the request path
- runtime consumes only validated policy artifacts through the existing `PolicyProvider` seam

### Learning Problem Decomposition

Meta-routing learning is split into three policy problems instead of one monolithic policy:

1. `Trigger Policy`
   - decide whether a request is fragile enough to justify refinement
2. `Action Policy`
   - decide which bounded refinement action or action set to use
3. `Adoption Policy`
   - decide whether the refined pass should replace the base result

This decomposition is durable because the three problems have different labels, reward shapes, safety constraints, and rollout risk.

### Canonical Data Model

The canonical learning surface remains the existing runtime schema:

- `FeedbackRecord`
  - `observation`
  - `action`
  - `outcome`
- `RoutingTrace`
- `PassTrace`
- `MetaAssessment`
- `RefinementPlan`

Learning workflows must consume those artifacts directly or through versioned flattened derivatives. They must not invent a second incompatible request-history schema.

### End-to-End Lifecycle

The repository treats meta-routing as one evidence-to-promotion lifecycle rather
than a collection of disconnected scripts:

1. request-time extproc emits `FeedbackRecord`, `RoutingTrace`, and pass-level evidence
2. replay or probe-oriented evaluation produces compatible trajectory evidence or a lossless mapping into the same canonical learning surface
3. offline tooling derives canonical trajectory rows and task-specific feature views
4. reward summaries and slice-aware evaluation determine whether a candidate overlay is acceptable
5. accepted overlays are packaged as policy artifacts
6. runtime loads only validated artifacts through the existing `PolicyProvider` seam
7. dashboard and reports attribute observed behavior to exact provider and artifact lineage

This lifecycle is durable because it prevents runtime evidence, synthetic probe
results, and promoted artifacts from drifting into unrelated data formats or
unverifiable rollout stories.

### Shadow as Counterfactual Data Collection

`shadow` mode is the primary counterfactual data-collection mode for learning.

The repository treats `shadow` as the source of evidence for questions such as:

- would a refined pass have changed the route or model
- which refinement actions help on which request slices
- what latency cost refinement adds
- which fragile requests are worth upgrading to `active`

That makes `shadow` a measurement mode, not just a rollout convenience.

### Progression Ladder

The learning progression is intentionally staged:

1. deterministic heuristics
2. calibrated trigger or action overlays
3. supervised or ranking-based policies
4. contextual bandit or conservative offline policy learning
5. offline RL only when reward quality and evaluation coverage are sufficient

The repository does not treat online RL in the extproc request path as an acceptable default extension of meta-routing. If true online learning is later desired, that requires a new explicit architecture decision.

### Reward Design

The reward model for meta-routing is multi-objective and must combine:

- outcome quality
  - route stability
  - weak-label improvements
  - fallback reduction
  - safety or verification improvements
- process quality
  - decision-margin gain
  - projection-boundary gain
  - contradiction reduction
- cost
  - latency overhead
  - expensive action usage

The repository will not treat “route changed” as a sufficient reward signal by itself.

### Artifact Promotion and Rollout

Learned or calibrated policies continue to ship through the existing artifact contract and `PolicyProvider` seam.

Promotion rules remain:

- fail closed if artifact validation fails
- require replay or feedback-based acceptance thresholds
- attribute results to exact provider lineage in `RoutingTrace` and `FeedbackRecord`
- roll out through `observe`, then `shadow`, then `active`
- retain explicit rollback criteria and regression evidence for promoted overlays

Training infrastructure, feature extraction, and offline evaluation may evolve, but the runtime promotion boundary remains artifact-based.

### Boundary With Routing Graph Optimization

Meta-routing learning improves trigger, action, and adoption behavior on top of
an existing route graph. It does not assume the route graph is globally optimal,
and it does not rewrite YAML or DSL rules automatically.

The repository therefore treats route-graph quality as a separate optimization
problem:

- meta-routing feedback may reveal route overlap, boundary hotspots, or persistent fragile slices
- those diagnostics are valid inputs to a separate routing-graph optimization workflow
- request-time runtime and policy artifacts do not directly mutate route definitions

If the repository later wants automated route-graph rewriting, that requires a
new explicit architecture decision rather than being absorbed silently into the
meta-routing learning loop.

### Public Contract Boundary

The public `routing.meta` schema stays intentionally narrow.

Training knobs, reward weights, optimizer choices, replay datasets, and internal evaluation gates remain outside the public YAML and DSL surface unless a later architecture decision intentionally widens that contract.

## Consequences

- The repository gains a clear continual-improvement story without giving up request-path safety, reproducibility, or rollback.
- `shadow` becomes a first-class learning-data mode rather than an implementation detail.
- Future work can optimize trigger, action, and adoption policies independently instead of overfitting a single monolithic learned controller.
- Offline scripts and artifact validation become more important because they are now the bridge between runtime evidence and promoted learned policy.
- The current runtime seam remains explainable: request handling stays bounded, while learning stays outside the hot path.
- The repository deliberately postpones true online RL. That avoids accidental self-modifying behavior, but it also means continual improvement depends on export, training, evaluation, and artifact promotion workflows being healthy.
- Runtime evidence, probe evaluation, and rollout evidence now need to stay compatible across one end-to-end lifecycle instead of evolving independently.
- Route-graph optimization is clarified as an adjacent but separate problem so meta-routing does not become a dumping ground for recipe-quality defects.
