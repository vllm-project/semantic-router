# 0.6B Milestone 2 amendment (M2b): candidate-vector collapse

Frozen 2026-09-28 while the QB seeds were still running, before any QB readout.

## Observation

Both QB seeds (bidirectional official Qwen, marker readout) stall like EB610:
SELECT near 250/700 through update ~260 of 466 and gradient norms falling to
≈0.06 with loss at chance, while the causal Qwen control under the same recipe
reached 460 by update 192. mmBERT-base trains under identical code. One mechanism
fits all three: if every candidate's head input becomes the same vector, the head
gradient Σₖ(pₖ−yₖ)·f(cₖ) is exactly zero because Σₖ(pₖ−yₖ)=0.

## Changes

1. **QBO is not run as specified** (it shares the marker readout); nothing was
   trained. QB runs to completion and is read out once as preregistered.
2. **Collapse probe** (`probe_collapse.py`, ≲0.05 GPU-hour, gold-free): mean
   pairwise cosine of the head's normalized candidate inputs within 64 SELECT
   Choice rows, for the marker state and for the candidate-span mean, on the
   zero-step Qwen/EuroBERT/mmBERT sources and the QB-s1, EB610 and MB307 exports.
   The diagnosis is supported if the QB-s1 and EB610 marker cosines are ≥ .95
   while MB307's is below both.
3. **QBS (s1, s2)**, run only if the probe supports the diagnosis: QB with the
   candidate vector replaced by the mean of the marker and its description tokens
   (`candidate_pool = span-mean`). Nothing else changes. This is the single
   readout change being tested.
4. **QBOS (s1, s2)** and **QBLS (s1, s2)** replace QBO and QBL on the span
   readout, and run only if both QBS seeds reach BEST SELECT ≥ 450/700 (the L8H
   level, i.e. no collapse). LXL is unchanged.
5. The EuroBERT root-cause item becomes one EuroBERT span-readout seed, run last
   (≤ 0.2 GPU-hour).

Finalist, formal-run and stop rules are unchanged from `m2-prereg-2026-09-28.md`.
