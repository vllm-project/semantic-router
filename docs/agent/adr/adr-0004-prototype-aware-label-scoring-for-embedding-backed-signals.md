# ADR 0004: Adopt Prototype-Aware Label Scoring for Embedding-Backed Signals

## Status

Proposed

## Context

The repository currently has multiple embedding-backed signal families that all start from exemplar-style similarity, but they do not share one durable scoring contract.

- `embedding` signals score a query against every candidate exemplar, aggregate candidate similarities per rule with `max`, `mean`, or `any`, then apply hard thresholds, soft floors, and `top_k` truncation.
- contrastive `preference` scoring embeds descriptions and examples per rule, but still chooses the winner with a flat `max similarity` pattern.
- `kb` scoring computes per-label scores from exemplar similarity and then lifts those scores to groups with `max`-style aggregation.
- `complexity` scoring compares `hard` and `easy` exemplar banks with `maxHardSim - maxEasySim`, then fuses text and image margins into one discrete difficulty output.

These implementations all work for small exemplar sets, but they do not age well as the repository accumulates more failure cases, more labels, and more routing diversity.

- Flat exemplar banks over-weight single noisy near-neighbors.
- `top_k` truncation collapses second-best evidence before later signal-group logic can reason about ambiguity.
- A single global threshold or winner-take-all rule does not fit labels whose exemplar distributions have different density or multiple semantic modes.
- Confidence and raw-score emission is inconsistent across signal families, which makes downstream decision, projection, and replay behavior harder to reason about.
- The repository already stores exemplar-like routing knowledge at roughly utterance scale, but the runtime still scores raw exemplar collections instead of stable label-owned prototypes.

The durable problem is no longer how to retrieve one nearest exemplar. It is how to score labels reliably when each label is represented by many heterogeneous examples.

## Decision

Adopt prototype-aware label scoring as the durable runtime contract for embedding-backed signal families.

- Treat `embedding`, contrastive `preference`, and `kb` signals as label-scoring systems rather than flat exemplar-ranking systems.
- Treat `complexity` as paired label scoring over `hard` and `easy` prototype banks, with text and image channels scored separately before fusion.
- Build label-owned prototypes from exemplars with label-local compression rather than scoring every raw exemplar as an equal first-class runtime object.
- Prefer medoid-like retained exemplars as runtime prototypes so scores remain explainable against real utterances instead of opaque centroids.
- Score every label from its prototypes before any emission-time truncation. `top_k` remains an output-shaping control, not the primary classification mechanism.
- Introduce shared acceptance semantics centered on label scores, runner-up margin, and abstain behavior instead of relying only on flat hard thresholds plus raw nearest-neighbor rank.
- Keep the external routing contract stable where possible:
  - matched rule outputs remain the public signal result
  - `SignalConfidences` remains the main confidence surface
  - `SignalValues` carries supporting numeric scores such as best score, support score, and margin
- Keep prototype-aware scoring in the signal-runtime layer. Existing ANN or indexing helpers are optional accelerators over prototype collections, not the source-of-truth scoring contract.
- Migrate the signal families through one shared scorer and family-specific adapters rather than growing four separate scoring stacks.

## Consequences

- The repository gets one durable scoring model for the embedding-backed families instead of separate `max similarity` variants that drift apart over time.
- Query-time competition moves from raw exemplars to labels, which improves resilience when exemplar banks grow, diversify, or contain noisy near-duplicates.
- Downstream routing and replay surfaces can consume more consistent confidences and raw metrics, especially margin and abstain signals.
- Existing config and runtime code will need staged migration so legacy exemplar scoring can coexist with prototype-aware scoring until the workstream closes.
- Signal-group behavior, especially winner-take-all or exclusive partition logic, must be updated to reason over full label score distributions instead of already-truncated candidate lists.
- Complexity scoring remains a special case because it compares positive and negative prototype banks plus optional multimodal fusion, but it now fits the same underlying label-scoring architecture.
- If future performance work adds ANN or other indexing, it must preserve prototype-aware label scoring semantics rather than reintroducing raw exemplar competition as the contract.
