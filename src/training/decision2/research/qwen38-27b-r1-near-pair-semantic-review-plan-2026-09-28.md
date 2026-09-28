# Locked r1 near-pair semantic review plan

This plan fixes a bounded, input-only review **before** opening the private
pair-ID receipt. It does not alter the already failed group-atomic400 r1
schedule, its zero-near requirement, or its GPU HOLD.

## Pinned evidence and sampling

- Locked schedule digest: `0743da3f1ed18cc8eb824c41c736d92b23fb3d8a738e1b3401ff830229424be5`.
- Independent CPU admission receipt SHA-256: `3f3da6e6ba616565a5ceb0cc7a4552d7c5aa563577598354b7aa29b7e88ae49d`.
- Review both reported plain-state near pairs: one SELECT and one CAL.
- For the 606 complete-input near pairs, draw 12 from SELECT and 12 from CAL
  by ascending SHA-256 of `r1-semantic-review-v1/<locked-schedule-digest>/<role>/<train-id>/<protected-id>`.
  Then add six further pairs per role by descending normalized state
  `SequenceMatcher` ratio, breaking ties by that same SHA-256 and excluding
  already selected pairs. The resulting fixed sample is 36 distinct pairs.
  This deliberately includes ordinary and highest-risk lexical collisions;
  it is **not** a random estimate of the whole 606-pair population.

Only the pinned TRAIN input-only fields, protected gold-free projection and
SELECT/CAL input-only source/group metadata may be opened. Do not read their
answers, any formal score, model prediction or teacher output. Keep pair IDs,
raw text and source records in a mode-0600 private packet. The public note may
contain only counts, evidence categories, anonymized structural descriptions
and file hashes.

## Judgment rules

For each pair, inspect state evidence, question/instruction and options:

1. **Material semantic overlap:** the same source event/record or distinctive
   answer-determining facts recur, or one input substantially paraphrases the
   other so a solution could transfer without the intended reasoning.
2. **Common template only:** the scaffold matches, but entity, evidence and
   decision-relevant facts differ materially. Matching task family or option
   labels alone is insufficient to call this overlap.
3. **Unresolved:** input-only evidence does not distinguish the above or the
   original protected source record ID is needed. Do not infer a PASS from a
   lexical screen or an apparently benign sample.

Report state and complete-input sample categories separately, with short
non-verbatim reasons. Inspect source/group metadata for every sampled pair,
but mark original-corpus provenance unresolved where the protected projection
does not carry original record IDs. Any prospective r2 exclusion or threshold
must be documented as a **new** policy before new selection; r1 remains HOLD.
