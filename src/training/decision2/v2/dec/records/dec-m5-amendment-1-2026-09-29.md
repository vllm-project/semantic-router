# Decoder Milestone 5 — amendment 1 (segment rule; implementation clarifications)

Written and pushed before any N5B / N5BN training and before any MLX-DEV readout of any model. At this point only
N5N (whose rows are N4XF's, unaffected by this amendment) is training; no Milestone 5 development readout exists.
Preregistration: [`dec-m5-prereg-2026-09-29.md`](dec-m5-prereg-2026-09-29.md) (`c62cc0853`); data lock part 1:
[`dec-m5-datalock-2026-09-29.md`](dec-m5-datalock-2026-09-29.md) (`513810437`).

## 1. Segment-disjointness rule: ignore template lines (> 50 N4XF groups)

**Problem found while building, before any readout.** Taken literally, the prereg's rule drops a candidate MLX-DEV
group if any normalized input segment of ≥ 20 characters occurs in any N4XF row. Instruction and option lines are
shared templates: "rate the overall sentiment that the message expresses." occurs in 2,716 N4XF rows and "final
assistant reply:" in 3,237. So every N4XF-unseen candidate group of JCommonsenseQA, MTOP, A7q, A7k, A7s and
SentiMix Spanglish conflicts, and all six Choice / Score cells come out empty (Choice-ML, Score-ML and M_dev would be
undefined). The two Noul cells fill under either rule.

**Amendment.** A segment that occurs in more than 50 distinct N4XF groups is a template line and is ignored by the
rule. This applies both to filtering MLX-DEV candidates and to dropping added N5B rows that share a segment with
MLX-DEV. Content lines occur in about 5 N4XF groups at most; template lines occur in 800–3,200. Every threshold from
10 to 50 gives identical exclusions. The rule is implemented as `v2.dec.m5_block --template-max-groups 50`; the
evidence is kept with the build receipts.

## 2. Implementation clarifications (the prereg did not fix these; chosen before any readout)

1. A group's language is its first row's language, for strata and for growth shares. MTOP groups mix languages;
   per-row shares had left MTOP 12–15% short of its quota in the dry run.
2. The shrinking Noul cell is H5 Noul and H8 Noul together, stratified by language only.
3. An added group that shares any non-template segment with MLX-DEV is dropped whole.
4. **Choice-ML** is the macro over the MLX-DEV Choice cells (JCommonsenseQA, MTOP). **Score-ML** is the macro over
   the Score cells (A7q, A7k, A7s, SentiMix Spanglish). **Noul-ML** stays the macro over languages.
5. The paired bootstrap resamples groups within each cell (stratified cluster bootstrap; 2,000 replicates, seed
   20260929).
6. The 300-row bitwise re-label check is also run on the rows added for N5BN.
7. **Schedule.** N5N runs first on both GPUs while N5B / N5BN wait for this amendment. Every seed keeps its
   preregistered recipe and seed.

Nothing else changes: arms, quotas, targets, selection, formal plan, 2B trigger and budget are as preregistered.
