# Meta-Routing Offline Learning Loop Workstream

## Goal

- Turn meta-routing from a deterministic bounded-refinement feature into a continuously improving system through offline learning, calibration, and artifact promotion.
- Keep the request path bounded and explainable while allowing trigger, action, and adoption policies to improve over time.
- Close this workstream only when the repository has one coherent learning loop spanning feature extraction, reward definition, offline evaluation, policy artifact promotion, and safe rollout.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `src/semantic-router/pkg/extproc/**`
- `tools/agent/scripts/**`
- `website/docs/**`
- targeted dashboard or report surfaces needed to explain policy lineage or rollout evidence
- focused runtime, script, and docs validation for the changed-file set
- nearest local `AGENTS.md` files for `src/semantic-router/pkg/extproc/` and any touched dashboard surfaces when implementation starts

## Exit Criteria

- One indexed ADR defines the offline-first learning loop, policy decomposition, reward philosophy, and promotion boundary for meta-routing.
- The repository has a versioned trajectory and flattened-feature schema sufficient for trigger, action, and adoption learning tasks.
- The repository has one canonical evidence contract spanning request-time feedback, `shadow` counterfactuals, replay data, and probe-oriented evaluation inputs through native rows or lossless mappings.
- Offline tooling can extract canonical training rows, build leakage-safe dataset splits, compute reward-relevant summaries, and evaluate candidate policies before runtime promotion.
- Policy artifacts can represent calibrated or learned overlays for trigger, action, and adoption decisions without widening public `routing.meta`.
- Promotion gates include slice-aware acceptance thresholds, counterfactual `shadow` evidence, and explicit rollback criteria.
- Dashboard or report surfaces can attribute behavior to exact policy lineage and explain whether learned promotion improved outcomes relative to cost.
- The offline loop emits graph-diagnosis exports such as route-pair confusion, projection-boundary hotspots, or persistent fragile slices for a separate routing-graph optimization workflow without mutating config automatically.

## Canonical Learning Loop

The repository should treat one request-level meta-routing record as the canonical
learning unit:

- one `FeedbackRecord`
- one `RoutingTrace`
- one base pass
- zero or one bounded refinement pass in v1
- one final outcome plus downstream weak labels

The trainable offline loop should become:

1. export or query persisted feedback records
2. import replay or probe-oriented evaluation evidence through the same canonical contract or a lossless mapping
3. derive one canonical trajectory row per request or probe
4. build task-specific training views for trigger, action, and adoption
5. compute multi-objective reward summaries
6. run offline evaluation on held-out slices
7. package accepted overlays into one policy artifact
8. promote through `observe -> shadow -> active`

This workstream deliberately keeps the runtime seam unchanged. The change is that
offline tooling becomes rigorous enough that the repository can fit, evaluate,
and promote learned overlays without inventing ad hoc one-off datasets.

## Required Dataset Shapes

The offline loop needs two durable data shapes:

- `trajectory`
  - one row per request
  - includes request metadata, provider lineage, base-pass trace, optional refined-pass trace, plan or action details, and final weak-label outcome
- `flattened features`
  - one row per request-task view
  - derived from the trajectory row for trigger, action, or adoption learning

The trajectory row is the canonical source of truth. Flattened features are
versioned derivatives and must reference:

- the trajectory schema version
- the feature schema version
- the evidence source and provenance
- the policy-provider identity
- the config or recipe version when available
- the rollout mode under which the sample was collected

## Dataset Split Rules

Offline evaluation should stop relying on a single aggregate replay score. The
plan now assumes the repository will define leakage-resistant splits:

- time-based split
  - train on older records, validate on newer ones
- rollout-based split
  - keep `observe`, `shadow`, and `active` evidence distinguishable
- evidence-source split
  - preserve differences between runtime traffic, replay, and maintained probe evaluation
- slice-aware split
  - preserve route, domain, language, and verification-sensitive slices
- optional session-aware split
  - keep session-correlated samples from leaking across train and test when session identifiers exist

`shadow` rows are the primary source for action and adoption evaluation because
they preserve both base and refined outcomes without changing production behavior.

## Evaluation Contract

The offline loop should evaluate candidate policies at three levels:

- task-level policy metrics
  - trigger precision, recall, calibration, and unnecessary-refinement rate
  - action usefulness, top-k action ranking quality, and slice-level action win rate
  - adoption safety, false-adopt rate, missed-adopt rate, and net overturn gain
- process metrics
  - decision-margin gain
  - projection-boundary gain
  - contradiction reduction
  - root-cause concentration by slice
- cost metrics
  - added latency
  - added signal cost or backend cost
  - change in verification-path usage

Promotion should require both aggregate and slice-aware acceptance. No artifact
should be accepted only because it improves one global scalar.

Candidate overlays should also be evaluated across both evidence families when
applicable:

- runtime traffic evidence
- replay or probe evidence

## Root-Cause and Trajectory Requirements

The learning loop should make failure analysis first-class instead of leaving it
in markdown reports only. The repository should add:

- automated root-cause classification or trimmer output for fragile or failed records
- canonical trajectory export that links:
  - policy provider before promotion
  - request-level observation
  - chosen or planned action
  - base outcome
  - refined outcome when present
  - reward summary
- compatibility between runtime feedback exports and offline calibration reports so the repository does not maintain two incompatible learning datasets
- graph-diagnosis exports that summarize:
  - route-pair confusion
  - persistent fragile slices
  - projection-boundary hotspots
  - action effectiveness by slice

## Promotion Gate Requirements

The learned-policy promotion path should explicitly require:

- artifact validation against the supported schema version
- replay-count minimums
- slice-aware thresholds
- counterfactual `shadow` evidence for any learned trigger, action, or adoption overlay
- compatible acceptance evidence across runtime traffic and maintained probe or replay suites when both exist
- rollback criteria stating when an artifact must be disabled after promotion
- provider-lineage attribution so regressions can be tied to the exact artifact

## Documentation and Operator Evidence

The operator-facing proof surface for learned policies should answer:

- what policy provider or artifact is active
- which slices improved
- which slices regressed
- how often `shadow` would have changed the route or model
- whether active adoption improved outcomes enough to justify added latency or cost
- what rollback decision should be taken if thresholds are missed

## Task List

- [x] `MRL001` Add an indexed ADR and execution plan for the meta-routing offline learning loop before implementation starts.
- [ ] `MRL002` Define the canonical learning problem split for trigger, action, and adoption policies, including their labels and negative examples.
- [ ] `MRL003` Define the canonical trajectory-row schema and the versioned flattened-feature derivatives consumed by training and evaluation jobs.
- [ ] `MRL004` Define leakage-resistant dataset split rules, including time-based, rollout-based, and slice-aware evaluation partitions.
- [ ] `MRL005` Define the reward and objective contract, including outcome quality, process quality, and latency or cost penalties.
- [ ] `MRL006` Extend flattened feature extraction with the higher-signal fragility features already recorded in the active backlog, plus provider lineage, rollout-slice metadata, and root-cause summaries.
- [ ] `MRL007` Add canonical trajectory export plus automated root-cause or trimmer classification so offline learning consumes the same request-history semantics as runtime feedback.
- [ ] `MRL008` Add offline evaluation support for candidate policies, including trigger precision, action usefulness, adoption safety, overturn gain, calibration quality, and latency tradeoff analysis.
- [ ] `MRL009` Add slice-aware acceptance and rollback tooling so candidate artifacts carry explicit promotion evidence instead of a single aggregate score.
- [ ] `MRL010` Extend the policy artifact contract or support tooling so promoted artifacts can carry trigger, action, and adoption overlays with explicit acceptance evidence.
- [ ] `MRL011` Add policy-promotion and rollback workflow docs that explain how `shadow` data becomes a candidate learned artifact and how that artifact is promoted safely.
- [ ] `MRL012` Add operator-facing summaries or dashboard hooks that make provider lineage, counterfactual effect, and learned-policy rollout evidence visible.
- [ ] `MRL013` Define the canonical evidence unification contract between request-time feedback, replay data, and probe-oriented evaluation so offline learning and acceptance gates compare compatible samples.
- [ ] `MRL014` Add graph-diagnosis exports that summarize route-pair confusion, boundary hotspots, and persistent fragile slices for a separate routing-graph optimization workflow.

## Current Loop

- Date: 2026-03-24
- Current task: `MRL001` completed
- Branch: `meta-routing`
- Planned loop order:
  - `L1` lock the learning-loop architecture in ADR and plan form
  - `L2` define learning tasks, trajectory shape, dataset split rules, and evidence-source unification
  - `L3` define reward, feature extraction, root-cause enrichment, and graph-diagnosis exports
  - `L4` implement offline evaluation, acceptance thresholds, and rollback evidence
  - `L5` extend artifact promotion workflow and operator evidence
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/README.md`, `docs/agent/governance.md`, `docs/agent/adr/README.md`, and `docs/agent/plans/README.md`
  - broad `codebase-retrieval` across existing meta-routing ADRs or plans, policy artifacts, feedback scripts, runtime trace schema, and offline support tooling
  - targeted source reads for `src/semantic-router/pkg/extproc/meta_routing_types.go`, `src/semantic-router/pkg/extproc/meta_routing_policy_loader.go`, `src/semantic-router/pkg/extproc/meta_routing_policy_artifact.go`, `tools/agent/scripts/meta_routing_feedback_features.py`, and `tools/agent/scripts/meta_routing_feedback_report.py`
  - additional source reads for `.augment/search-r2.md` and `.augment/router-memory.md` to pressure-test the offline learning roadmap against trajectory logging, hybrid reward, and routing-state ideas
  - governance validation will be rerun after the strengthened plan is updated
- This plan now assumes later implementation loops will touch runtime support tooling, offline scripts, artifact validation, and operator evidence surfaces. It still does not authorize request-path self-modifying behavior.

## Decision Log

- 2026-03-24: Continual improvement for meta-routing stays offline-first; request-path online self-modification is out of scope.
- 2026-03-24: Learning work is decomposed into trigger, action, and adoption policies rather than a single monolithic controller.
- 2026-03-24: `shadow` is treated as primary counterfactual data collection for policy learning and policy promotion.
- 2026-03-24: The runtime promotion boundary remains artifact-based through the existing `PolicyProvider` seam.
- 2026-03-24: A trainable offline loop requires a canonical trajectory dataset and leakage-resistant dataset split rules; flattened features alone are not enough.
- 2026-03-24: Root-cause classification, hybrid reward summaries, and slice-aware acceptance gates are treated as first-class requirements rather than optional reporting niceties.
- 2026-03-24: Runtime traffic evidence and probe-oriented evaluation must converge on one compatible learning contract so offline acceptance does not compare incomparable data.
- 2026-03-24: Meta-routing learning must emit graph-diagnosis outputs, but route-graph optimization itself remains a separate workflow.

## Follow-up Debt / ADR Links

- [ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [ADR 0006: Productize Meta-Routing Through Operator-Oriented UX and APIServer-Owned Feedback APIs](../adr/adr-0006-meta-routing-operator-ux-and-apiserver-feedback.md)
- [ADR 0007: Evolve Meta-Routing Through an Offline-First Learning Loop and Artifact Promotion](../adr/adr-0007-meta-routing-offline-learning-loop.md)
- [ADR 0008: Separate Routing Graph Optimization From Meta-Routing Reliability Improvement](../adr/adr-0008-routing-graph-optimization-separate-from-meta-routing.md)
- [pl-0017-meta-routing-learned-policy-enablement.md](pl-0017-meta-routing-learned-policy-enablement.md)
- [pl-0018-meta-routing-operator-productization.md](pl-0018-meta-routing-operator-productization.md)
- [pl-0020-routing-graph-quality-and-optimization-workstream.md](pl-0020-routing-graph-quality-and-optimization-workstream.md)
