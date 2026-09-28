# Eval M3 preregistration — amendment 5: dataset-level source independence for C1 v1.1 (2026-09-28)

Coordinator decision, 20:30: before any scoring event, drop every C1 task whose source
(any split) appears anywhere in training data at dataset level. Committed before the check
runs. CPU only.

## Training data in scope

- **Private training dataset.**
  - `5c0255ed` (root TRAIN/SELECT/CAL, v1 arms A0–A6, replay RP-v1q/R2, the old
    noncommercial pilot);
  - `39a120ca` (A7 v1);
  - `ed87a03a` (data-v2 `m2/` arms and recipes);
  - current main `d8eae3e4` (pk1 re-keyed A0/A0s/A0p/RP-v1q/R2, the rebuilt A7r and encoder
    sub-arms, Lux teacher target files, CAL698).
- **Teacher prompt pools on node A.** RP-v2 and TS-v2 (the Lux and AutoJev screen and
  target pools), plus the canonical Lux A0 file.
- **Track branches.** Code and records of the data, A7, 0.6B, decoder, 9B, 27B and release
  branches: any dataset loaded outside the private dataset.

## Checks

1. **Names and provenance.** Search every registry, manifest, readme and audit file in
   those snapshots, every training row's provenance fields (`source`, `family`, `id`,
   `group_id`, `render_template`, all `audit_metadata`), and the track branches' trees for
   each C1 source. Terms: dataset id, repository name and distinctive dataset name.
2. **Content.** Convert every row of every file of each C1 source, all splits, into a
   protected file of its text fields (≥ 20 characters). Scan it with
   `v2.eval.sealed.overlap` against the training corpora and teacher pools.

## Rule

A C1 source is in training at dataset level when either condition holds:

- check 1 finds it; or
- at least 5 of its rows match training rows at containment ≥ 0.8, or by an exact match of
  ≥ 8 tokens, **and** the matching training rows' provenance is the C1 dataset or a
  derivative of it.

Matches that trace to an independent upstream corpus both sides draw on are text-level
overlap, not dataset presence. Examples are Wikipedia passages in TyDi QA, or public
court opinions. Such matches are disclosed. C1 v1 already excluded the affected selected
items at item level.

## Outcome

Every task of a present source is dropped. All other C1 v1 items stay byte-identical.
C1 v1.1 is re-sealed with new hashes, logged, and disclosed with counts per type, task and
language.
