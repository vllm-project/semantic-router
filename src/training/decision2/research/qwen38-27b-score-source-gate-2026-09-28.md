# Qwen3.8-27B next Score arm: source-data readiness

**Status: CPU-data HOLD.** This note does not change the prospective A/B
contrast or its thresholds in
`qwen38-27b-next-score-arm-prereg-2026-09-28.md`. No GPU training, inference,
selector-key read, formal assessment or model upload was performed for this
data gate.

## Reused evidence and its limits

The earlier v7p matched-arm CPU assembly had 80 TRAIN and 80 SELECT triplets,
2,288 examples per arm and roughly 438,000 native tokens. Its one synthetic
evidence-join template family and insufficient independent construct review
made it unsuitable for the newly specified six-mechanism arm. Score v8.3 had
workflow realism and copy faults; v8.4 had source/copy shortcuts; v8.5 missed
its frozen long-row quota. The later 12-group hand-authored pilot was short
and not admitted to a model arm. Reusing, relabeling or backfilling these
groups would not satisfy the new prospective comparison.

No new source-grounded 160-group corpus, separate fresh selector, source-rights
ledger or 480-item blinded review seal exists in this branch. Therefore the
currently admitted count is **0/80 TRAIN and 0/80 SELECT independent groups**.
This is a readiness count, not a negative model result. The existing BEST368
Score development result remains 162/400; no updated 27B score follows.

## CPU-only custody helper

`training/data/score27_case_registry.py` accepts a private JSONL registry of
case JSON files. Each case supplies a stable ID, role, one of the six fixed
mechanisms, a distinct source family and author, at least two hashed private
source documents with rights references, and three different native Score
inputs with levels 0/1/2 and structured facts. The helper verifies source
bytes and file privacy, rejects duplicate IDs/source bytes/families across
groups, enforces disjoint TRAIN/SELECT authorship IDs and the frozen
14/14/13/13/13/13 mechanism quotas per role. Its receipt contains aggregate
counts and digests, not source text, private paths or answers.

Even a structurally complete 160-group receipt deliberately remains
`HOLD_PENDING_ORACLE_OVERLAP_TOKENS_AND_BLIND_REVIEW`. Source hashes do not
prove the rendered documents support the labels; distinct author IDs do not
prove blinding. This helper does not certify realism, rights terms, source
necessity, native token limits, exposure matching, semantic independence or
reviewer agreement. Those are separate gates in the frozen preregistration.

## Next admission sequence

1. Author the 80 new TRAIN cases and separately author 80 fresh SELECT cases
   from disjoint source families. Fix structured facts and source documents
   before rendering each three-level native Score triplet; record source
   versions, rights and removal decisions privately.
2. Freeze the case registry, rendered rows, source inventory, independent
   oracles and separate answer-free review packets. Run this structural helper,
   then the exact tokenizer/context, full reference overlap, source-removal,
   one-field and option-order audits. A failure holds the whole fixed version.
3. Have independent reviewers answer all 160 groups/480 variants and review
   source necessity, ambiguity and realism without the keys. Seal their
   decisions before comparison with the frozen oracle. Any unresolved mismatch
   or systematic shortcut remains HOLD; no favorable subset is substituted.
4. Only after data admission, build fresh A/B replay/control manifests and
   demonstrate exact 2,288-row arms, common 430–450k raw native-token budget,
   ≤1% raw and ≤5% dynamic padded-token exposure differences. The subsequent
   zero-step and one-step gates remain mandatory before either optimizer.

Current blocker is the **160 independently sourced and reviewed new case
groups**. The script makes progress measurable but cannot manufacture that
evidence; prior generated data are a useful failure analysis, not eligible
replacement rows. Training should remain unscheduled until this blocker and
the remaining admission checks are resolved.
