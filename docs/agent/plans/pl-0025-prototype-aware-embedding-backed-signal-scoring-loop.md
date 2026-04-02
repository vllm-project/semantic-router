# Prototype-Aware Embedding-Backed Signal Scoring Execution Plan

## Goal

- Replace flat exemplar-ranking semantics in embedding-backed signal families with prototype-aware label scoring.
- Converge `embedding`, contrastive `preference`, `kb`, and `complexity` onto one durable scoring model while preserving the existing routing-facing result contract where possible.
- Close the workstream only after config, runtime scoring, downstream signal consumption, docs, and targeted validation all agree on the same prototype-aware contract.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/extproc/**`
- `src/semantic-router/pkg/decision/**`
- maintained configs or examples that exercise the affected signal families
- targeted tests and repo-native validation for the touched surfaces
- nearest local rules for `pkg/classification`, `pkg/config`, and `pkg/extproc`

## Exit Criteria

- The repository has one shared prototype-aware scoring contract for embedding-backed signals, with family-specific adapters instead of four unrelated exemplar-scoring implementations.
- `embedding`, contrastive `preference`, and `kb` all score labels from label-owned prototypes rather than flat raw exemplar banks.
- `complexity` scores `hard` and `easy` prototype banks per rule and exposes fused margins through the same shared scoring vocabulary.
- `top_k` no longer acts as the internal competition boundary for embedding signal evaluation; it only controls emitted results after full label scoring.
- Signal outputs expose consistent confidence and supporting numeric metrics, including at least score and runner-up margin where those concepts apply.
- Signal-group, partition, or downstream decision logic that depends on embedding-backed scores consumes the new label-score semantics correctly.
- Applicable repo-native validation is rerun until the changed-set gates pass or durable blockers are recorded.

## Task List

- [x] `PAS001` Create and index the durable ADR and execution plan for prototype-aware label scoring.
- [ ] `PAS002` Define the shared config and runtime contract for prototype builders, prototype metadata, label scoring, margin/abstain semantics, and score emission across the affected signal families.
- [ ] `PAS003` Implement the shared prototype builder and scorer, including label-local compression, retained prototype identity, and reusable score aggregation helpers.
- [ ] `PAS004` Migrate contrastive `preference` and `kb` scoring onto the shared scorer and unify their `SignalConfidences` and `SignalValues` emission.
- [ ] `PAS005` Migrate `embedding` signal scoring so rule evaluation uses the full label score distribution, and update grouping or partition logic that currently depends on pre-truncated matches.
- [ ] `PAS006` Migrate `complexity` scoring onto paired `hard` and `easy` prototype banks with explicit per-channel scores, fused margins, and confidence emission.
- [ ] `PAS007` Update maintained configs, docs, targeted tests, and validation paths so the repository exposes one coherent prototype-aware contract.
- [ ] `PAS008` Run the validation ladder for the migrated surfaces, record results here, and add indexed debt only for gaps that remain after the workstream closes.

## Current Loop

- Loop status: opened on 2026-04-02.
- Completed in this loop:
  - audited the current `embedding`, `complexity`, contrastive `preference`, and `kb` scoring paths, including their config surfaces and signal evaluators
  - confirmed the main failure modes: flat `max similarity` dominance, premature `top_k` truncation, inconsistent confidence emission, and no shared label-level margin semantics
  - locked the durable architecture direction in `ADR 0004`
  - created this execution plan and indexed it for resumable follow-up work
- Next loop focus:
  - execute `PAS002` by defining the repo-native shared scorer contract and the smallest compatibility-preserving migration boundary for the first implementation pass

## Decision Log

- 2026-04-02: keep this as one execution plan because the migration spans multiple signal families, shared runtime helpers, config schema, and downstream signal consumers.
- 2026-04-02: treat prototype-aware scoring as label scoring, not exemplar retrieval with threshold patches.
- 2026-04-02: prefer retained medoid-like prototypes over pure centroids so runtime decisions remain debuggable against real exemplars.
- 2026-04-02: preserve existing matched-rule outputs where possible and move the semantic upgrade behind score computation, confidence emission, and raw metric plumbing.
- 2026-04-02: `top_k` becomes an emission control, not the main competition boundary for embedding-backed signals.
- 2026-04-02: ANN or HNSW acceleration is optional follow-up implementation detail over prototype sets, not the durable scoring contract.

## Follow-up Debt / ADR Links

- [adr-0004-prototype-aware-label-scoring-for-embedding-backed-signals.md](../adr/adr-0004-prototype-aware-label-scoring-for-embedding-backed-signals.md)
- [pl-0018-generic-embedding-kb-workstream.md](pl-0018-generic-embedding-kb-workstream.md)
- [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
- [TD015 Weakly Typed Config and DSL Contracts Obscure the Canonical Routing Surface](../tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md)
