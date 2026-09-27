# Official Qwen3 0.6B: ordered Score objective ablation

**Status:** prospective development experiment. This document freezes the
intervention before its first GPU model load. It does not authorize a formal
JevArena, JevBench, Hugging Face upload, or SOTA claim. The comparison is to
the archived [same-source control](qwen3-06b-official-base-full-clean-v2-prereg-2026-09-27.md),
whose SELECT result was 562/700 and family macro accuracy 0.77259.

## One change and why

System One Score exposes an **ordered**, runtime-supplied 2–10-level rubric and
returns a probability distribution and its expected zero-based level. A
categorical loss does not distinguish an adjacent-grade error from a far-grade
error. This treatment retains the same dynamic-option backbone/readout and
adds a normalized ranked-probability score (RPS) term on **Score TRAIN rows
only**. The native option keys define level order, so the term is correct even
when a training row presents levels out of order. For `K` levels and label
level `y`, the term is

`sum_{j=0}^{K-2} (P(level<=j) - 1[y<=j])^2 / (K-1)`.

The coefficient is fixed at **1.0**. The existing cross-entropy plus
0.5×categorical-Brier loss remains active on every row. No extra training
rows, teachers, replay, task weighting, changed prompt, generated answer
tokens, or new chat path enter this arm. This is an objective ablation, **not**
a claim of a new backbone architecture. It tests whether the ordinal
inductive bias can repair the observed zero-level collapse without losing
Choice, Noul or real-task transfer. Limited Score TRAIN coverage (516 related
synthetic rows) remains a separate constraint.

## Frozen source, data, schedule and controls

- Direct start: official `Qwen/Qwen3-0.6B-Base` revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, fresh random shared
  candidate head, seed 20260926. The already trained control is **not** the
  initializer.
- Rights-clean v2 TRAIN 7,455 rows SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  SELECT 700 SHA-256
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL 700 SHA-256
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  TRAIN must encode to the control's 4,094,489 tokens, with zero 8,192-token
  overflows and unchanged token IDs/candidate positions. CAL is lineage-only
  during training and never used to select a checkpoint.
- Full backbone plus shared head, one epoch, 466 updates, microbatch 1,
  accumulation 16, AdamW weight decay .01, clip 1.0, BF16 backbone compute,
  FP32 parameters/head/loss, checkpointing on. Backbone peak LR `2e-5`, head
  peak LR `2e-4`, warmup .05, cosine decay. SELECT at 64-step multiples and
  466; choose one BEST by family macro accuracy, then normalized Brier, then
  earliest step. No other checkpoint search.
- One verified idle GPU on an authorized node. Cap **1.0 GPU-hour** including
  preflight, full training, SELECT and saves. Keep failures and stop on source,
  token, data, gradient, finite-loss, memory, saved-output or budget failure.
  A failed preflight does not turn into a modified recipe in this arm.

## Preflight and result gates

Before the full run, confirm Score level keys are exactly the contiguous
0-based rubric and that permuted keys give the same ordered RPS, non-Score
rows have zero ordinal term, gradients are finite and nonzero, and native
zero-step Choice/Noul/Score output plus save/reload are stable. Preserve the
source and artifact hashes. An incompatible direct source or different token
exposure stops the arm.

The **development SELECT gate** is at least 562/700 correct **and** family
macro accuracy at least 0.77259 against the exact archived control. If it
fails, retain the bounded negative result and do not use typed DEV/CSS pilot,
keyed v3 or public JevBench to rescue it. If it passes, run the same native
typed DEV1,600 and CSS pilot1,430 adapters as the control exactly once. The
diagnostic target is Score at least 100/400 versus control85/400, typed family
macro at least 0.289375, CSS pilot task-median macro-F1 at least 0.280, and
`100*sqrt(T_dev*H_pilot)` at least 29.02. These diagnostics do not alter
SELECT's frozen checkpoint. Report Choice, Noul, Score, expected-value MAE,
Brier, ECE, option-order robustness and source-specific errors, even if any
target fails.

Only if those development diagnostics are coherent may a **new** separately
frozen package enter post-key same-panel JevArena v3/public JevBench; the
formal minimum would be v3 composite **at least 41.0** versus Kai1 35.9383,
with paired uncertainty and all tradeoffs disclosed. The v3 key has already
been accessed in the project, so that panel is not a fresh blind test. An
independent source-disjoint or external result is required before a general
SOTA/Pareto claim. This arm is not a replacement for better Score data.
