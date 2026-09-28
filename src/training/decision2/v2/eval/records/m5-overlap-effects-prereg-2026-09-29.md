# Rescreen overlap vs released scores — preregistration (eval & peers, 2026-09-29)

Written and pushed before any score without the flagged items is computed. Question: the research & data
rescreen (M3b amendment 4; 305 training groups excluded from r2) hit evaluation items after the release decisions.
Models trained on r1 / v2 / v1 / A7 data, including the released DEV2.0-0.8B and DEV2.0-0.6B and the approved 2B
candidate (S2T), may have seen them. Do the reported scores or any release conclusion change when those items are
removed? Stored sealed predictions only, node A CPU, no GPU, no C1.

## Flagged items

- Collected by `python3 -m v2.eval.overlap_effects flagged` from the rescreen's per-pool private receipts under
  `/data/dev2/runs/data/m3b/xl-r2/rescreen/` (plus the union receipt), for the seven roles with excluding hits:
  `css15_goldfree`, `css15_native`, `jevbench_public231`, `jevbench_public231_native`, `mlx_diag_v1`,
  `decision_bench_v4`, `ml_parallel_dev_v1`. The group set must equal the 305 groups of `rescreen.private.json`.
- Expected from the data-track record: CSS15 82 items (73 `media_ideology`), public 231 1, mlx-diag 1, typed FINAL 0.
  Decision Bench v4 (33) and ml-parallel-dev (3) are in no reported panel: counted, not scored.
- The id list stays on node A; the record gives counts, pools, methods and a hash only.

## Models

[`m5-overlap-effects/spec.json`](m5-overlap-effects/spec.json): each candidate with its own 1.0 model(s) and the tier
peers bound on its card (plus JPT-0.8B, which is in the 0.8B threshold pool but not on the card). Card-bound runs are
primary; the gate runs they were checked with must give identical outcomes.

## Scores (full panels and without the flagged items; the same items for every model)

- typed T (unchanged when no typed item is flagged), human transfer H (median over the 15 tasks of macro-F1 over each
  task's full label set), v3 = 100·√(T·H), per-task macro-F1, public 231 (count; 230 items without), mlx-diag type
  macro accuracy and per language (2,274 items without).
- Paired candidate-minus-comparator 95% CIs: the joint v3 / T / H bootstrap of `jev_arena.compare_v3` (5,000 draws,
  seed 20260927) on the full and reduced panels, plus the full panels with seed 20260928 as the reference for how far
  an interval moves from resampling alone; `media_ideology` macro-F1 (paired items within the task); public 231
  (paired items within tier); mlx-diag (paired items within type × language).

## Validation (stop and record if it fails)

The full-panel rescore must equal every model's `REPORT.json` (v3, T, H, per-task macro-F1, public 231) and stored
mlx-diag score exactly, and the full-panel v3 bootstrap must reproduce the 17 stored paired files in the spec (gate
files and release `PAIRED-vs-*` files) bit for bit.

## What counts as a changed conclusion (fixed now)

1. the own-1.0 gate flips (v3 CI lower bound > 0);
2. the first-release threshold flips (0.9 × the best open peer's v3; the best peer and threshold are recomputed
   without the flagged items);
3. "human transfer not significantly below the best peer" flips;
4. any paired CI changes status (above 0 / includes 0 / below 0) for v3, H, T, `media_ideology` F1, public 231 or
   mlx-diag;
5. the rank order within a tier changes on v3, H, `media_ideology` F1, public 231 or mlx-diag.

A status change that also appears between the two full-panel seeds is reported as resampling-sensitive, not as an
effect of the overlap. **Material** = any change of type 1–3, or a type 4–5 change on a number the card states. If
nothing is material, the card gets one disclosure line; otherwise the record states the exact correction.

## Contamination signature

On CSS15, for each model: accuracy on the flagged items minus accuracy on the unflagged items of the same tasks
(unflagged accuracy weighted by the flagged items per task). For each pair: the candidate's gap minus the
comparator's gap (difference in differences), with a paired bootstrap (items within task and flag; 5,000 draws, seed
20260927), over all 82 flagged items and over the 73 `media_ideology` items. **Signature** = a difference whose CI lies
above 0 against the candidate's own 1.0 model or against a peer. Descriptive only: mean gold-label probability on
flagged vs unflagged items, and which training pools hit the flagged items. Power is limited (73 items: the standard
error of an accuracy gap is about 0.06), so the record reports the intervals, not just the verdict.
