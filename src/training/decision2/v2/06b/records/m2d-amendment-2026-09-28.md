# 0.6B Milestone 2 amendment (M2d): causal official-Qwen control with the Lux1 teacher

Frozen 2026-09-28 before any QC/QCL run.

## Why

- QBQ s1 hit its preregistered collapse stop (238/700 SELECT at update 175,
  gradient norm 0.04), so neither the candidate readout (QBS) nor the query (QBQ)
  explains the collapse; QBQ s2 and QBQL are not run.
- At update 0 the backbone gradients are right under every training setting
  (padding, contiguous bool/float masks, deterministic algorithms, gradient
  checkpointing: cosine ≥ .99 against unpadded one-row passes for Qwen, EuroBERT
  and mmBERT). The collapse is an optimization failure of fresh heads on these
  RMSNorm/SwiGLU backbones in the bidirectional marker layout, not a code bug.
  The EuroBERT root-cause item stops here (≈0.1 GPU-hour of probes).
- The strongest 0.6B result remains the official-Qwen causal control (post-key
  v3 38.520, transfer H .480, but typed Choice 109/800 and constant typed Score).
  Its weaknesses are what a strong teacher targets, and Lux1's TRAIN screen
  (Choice .835, Noul .893, Score .866, three-level 102/102) covers all three types.

## Arms (official Qwen backup weights, approved backbone; same data and budget)

| Arm | Seeds | Definition |
| --- | --- | --- |
| QC | s1, s2 | The archived control's model reused unchanged from `training.model.decision_model` (state-first renderer, causal option endpoints plus global query, shared head), trained by this track's trainer: CE + 0.5 Brier, 466 updates, LR 2e-5/2e-4, 5% warmup, cosine to 1e-6, micro-batches ≤ 8 rows |
| QCL | s1, s2 | QC + 1.0·KL(Lux1 ∥ student) on every TRAIN row, teacher `2d90bc5b…` |

QC reproduces the control inside this trainer (its differences from the archived
run are the cosine floor, micro-batching and data order) and is the paired
control for QCL. Both carry the collapse stop (SELECT < 350 at update 175).
Readout uses `training.model.infer` at the 8,192 cap in the image runtime
(Transformers 5.17), as the archived control did. Finalist, formal and stop rules
are unchanged from `m2-prereg-2026-09-28.md`.

## M2e diagnostic (frozen before its run)

QC s1 and QCL s2 hit the collapse stop (311 and 259 at update 175); QCL s1 trains
(403 at 175, 511 at 350). The remaining difference from the reference trainer
that trained the archived control is micro-batching: it used unpadded one-row
micro-batches (unmasked causal attention path), while this trainer pads up to
eight rows (masked attention kernel; step-0 gradient cosine .991 for Qwen versus
≥ .998 for the other backbones). **QCMB1 s1/s2** repeat QC with
`max_micro_rows = 1` and nothing else changed, keeping the collapse stop. If both
seeds pass it they are QC candidates under the unchanged finalist rule; s2 runs
only if s1 passes.
