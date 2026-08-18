# PL-0042: MoM Training Source Consolidation

## Goal

Make Semantic Router the canonical, reproducible source for the model families
published in the multilingual MoM classifier and embedding collections.

## Scope

- Maintain a public mapping from published model artifacts to producing source.
- Consolidate the mmBERT-32K foundation, embedder, and reranker workflows.
- Consolidate one canonical small and one canonical large multimodal workflow.
- Reconstruct and release the missing safety-classifier workflow.
- Pin model, dataset, dependency, seed, export, and evaluation contracts.

## Non-Goals

- Committing model weights, datasets, caches, checkpoints, logs, or
  machine-specific deployment details.
- Retaining duplicate standalone repository layouts when a narrower in-tree
  module owns the same model family.
- Claiming historical checkpoints are exactly reproducible when their original
  training state was not published.

## Exit Criteria

- Every current collection model family maps to one documented in-tree source.
- The foundation, text embedding, reranker, small multimodal, large multimodal,
  and safety workflows live under narrow `src/training` owners.
- Revisions, dependencies, seeds, outputs, and evaluation commands are explicit.
- Lightweight tests validate configuration and data contracts without model
  downloads or production accelerators.
- The public provenance page matches current collection membership.
- The training-stack harness gates pass for the complete consolidation.

## Task List

- [x] `MOMSRC-01` Audit and pin the four source repositories at upstream `main`.
- [x] `MOMSRC-02` Map classifier and embedding collection members to source families.
- [ ] `MOMSRC-03` Consolidate mmBERT-32K foundation, embedder, and reranker source.
- [ ] `MOMSRC-04` Consolidate the canonical small multimodal source.
- [ ] `MOMSRC-05` Consolidate the canonical large multimodal source.
- [ ] `MOMSRC-06` Complete the safety reconstruction tracked by PL-0043.
- [ ] `MOMSRC-07` Publish provenance, run lightweight tests, and pass harness gates.

## Next Action

Normalize each selected upstream workflow into a narrow in-tree module while
the safety reconstruction proceeds through accelerator validation.

## Operating Rules

- Preserve the upstream commit identity and authorship in each module README.
- Keep one canonical implementation per model family.
- Replace machine-specific paths with explicit CLI or configuration inputs.
- Keep private research checkouts and infrastructure details outside Git.
- Distinguish adapters, merged checkpoints, ONNX exports, and packaged custom
  models in source and documentation.

## Related Docs

- [PL-0043: Safety classifier reconstruction](pl-0043-mmbert32k-safety-classifier-reconstruction.md)
- [Training stack change skill](../../skills/training-stack-change/SKILL.md)
