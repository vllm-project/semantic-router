# Routing Graph Quality and Optimization Workstream

## Goal

- Establish a durable workflow for diagnosing and improving routing-graph quality without overloading meta-routing request-time behavior.
- Use meta-routing traces, feedback records, replay results, and maintained probes as evidence for graph redesign rather than as justification for runtime self-modification.
- Close this workstream only when the repository has one coherent route-graph diagnosis and optimization loop spanning evidence intake, analysis outputs, human-reviewable recommendations, and config-governed rollout.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `deploy/recipes/**`
- `deploy/amd/*.probes.yaml`
- targeted router, dashboard, and tooling surfaces that expose route-pair confusion, boundary hotspots, or maintained recipe quality evidence
- focused docs and governance validation for the changed-file set

## Exit Criteria

- One indexed ADR defines route-graph optimization as a workflow separate from meta-routing reliability learning.
- The repository has a durable definition of the evidence sources that feed graph diagnosis, including runtime feedback, replay or probe evaluation, and maintained recipe coverage.
- The workflow emits human-reviewable graph-quality outputs such as route-pair confusion, threshold hotspots, persistent fragile slices, or maintained probe gaps.
- Graph-quality recommendations are explicitly reviewed and applied through normal config or DSL changes rather than request-time self-modification.
- The workstream documents how graph-quality findings relate to but do not replace meta-routing trigger or action learning.

## Task List

- [x] `RGQ001` Add an indexed ADR and execution plan for routing-graph quality and optimization before implementation starts.
- [ ] `RGQ002` Define the canonical evidence inputs for graph diagnosis, including meta-routing traces, feedback records, replay outputs, and maintained probes.
- [ ] `RGQ003` Define the first graph-quality outputs, such as route-pair confusion, projection-boundary hotspots, persistent fragile slices, and maintained coverage gaps.
- [ ] `RGQ004` Define the human-reviewable recommendation format so graph-quality findings can become normal YAML, DSL, or maintained-asset changes.
- [ ] `RGQ005` Define validation and regression gates for graph-quality changes, including probe coverage and overlap-reduction checks where applicable.

## Current Loop

- Date: 2026-03-24
- Current task: `RGQ001` completed
- Branch: `meta-routing`
- Planned loop order:
  - `L1` lock the graph-optimization boundary in ADR and plan form
  - `L2` define evidence inputs and diagnostic outputs
  - `L3` define recommendation and review workflow
  - `L4` define validation and regression gates for graph-quality changes
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/README.md`, `docs/agent/adr/README.md`, and `docs/agent/plans/README.md`
  - broad `codebase-retrieval` across meta-routing ADRs or plans, maintained probe assets, route-quality concerns, and operator workflow docs
  - governance validation will be rerun after these new artifacts are indexed
- This plan is governance-only for now. No route-graph optimization implementation work is in scope for this loop.

## Decision Log

- 2026-03-24: Route-graph quality is treated as a separate optimization problem from request-time meta-routing reliability.
- 2026-03-24: Meta-routing traces and feedback are inputs to graph diagnosis, not permission for request-time self-modifying config behavior.
- 2026-03-24: Graph-quality improvements remain normal config or DSL changes with normal review and rollback, not policy-provider overlays.

## Follow-up Debt / ADR Links

- [ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [ADR 0007: Evolve Meta-Routing Through an Offline-First Learning Loop and Artifact Promotion](../adr/adr-0007-meta-routing-offline-learning-loop.md)
- [ADR 0008: Separate Routing Graph Optimization From Meta-Routing Reliability Improvement](../adr/adr-0008-routing-graph-optimization-separate-from-meta-routing.md)
- [pl-0018-meta-routing-operator-productization.md](pl-0018-meta-routing-operator-productization.md)
- [pl-0019-meta-routing-learning-loop.md](pl-0019-meta-routing-learning-loop.md)
