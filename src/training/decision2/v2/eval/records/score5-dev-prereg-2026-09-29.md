# Score5-DEV v1 preregistration: a 5-level Score development check (2026-09-29)

Eval track, development readout only. Written and pushed before any model is run on the new panel.

## Why

Coordinator request (2026-09-29 08:20 UTC+8): the 0.6B candidate `m6-mxcx-soup` collapsed on the formal typed FINAL Score
(recorded aggregate: 379 of 400 answers = level 4; accuracy 131 vs always-4 128) while `m6-mxcxa-soup` did not. The
development readout's typed-DEV Score has only 3 levels, so it can miss a 5-level collapse. Plausible cause: Lux
distillation on the human-rated 5-level Score rows (A7q, OASST1 reply ratings). This check reports level usage and
accuracy on a held-out slice of that 5-level Score arm.

## Source

- A7q held-out slice `v2/a7/arms/A7q/aho.jsonl` of the private dataset `llm-semantic-router/decision-2.0-training-data`
  at `0b47239e` (node A copy under `/data/dev2/private/eval/c1-corpora/training/d8eae3e4/`), 2,413 rows, sha256
  `f04557e25bcbeb6b…` (full hash in the build MANIFEST; must match before use), Apache-2.0 (OASST1).
- Five levels (row `label` 0–4; 445 / 523 / 472 / 507 / 466), four rating axes (helpfulness 911, creativity 731,
  quality 530, humor 241), 790 source groups (message trees), multilingual (English 975 of 2,413; Spanish 733).
- Held out from every training recipe by group hash, but about 1,324 A7q training groups overlap it by n-grams: the
  panel is **FAMILIAR** (in-family, same source, same templates), not a transfer test. Disclosed in every readout.

## Panel `score5-dev` v1

- Seeded (seed 20260929), gold-level-stratified sample of 500 rows = 100 per level from aho.jsonl, spread as evenly as
  possible over the four axes and over source groups (at most one row per group if feasible); the other 1,913 rows are the
  fit pool for training tracks; score5-dev rows are never used for fitting, calibration or selection other than this check.
  Development readout only: never a release score, never in v3, charts or cards.
- Procedure (details; the rule above governs): rows are ordered by `sha256("20260929:" + row id)`. Target cells are
  level × axis with 25 per cell (water-filled onto other axes of the same level if a cell cannot supply 25 under the
  group rule). Cells are filled scarcest-first (fewest eligible rows), each taking the first rows in seeded order whose
  group is not yet used. If the one-row-per-group rule cannot fill 500, the remaining slots are filled with a cap of two
  rows per group and the MANIFEST says so.
- Rendering: each item is the row's own state and instructions with the five level descriptions as a Score `criteria`
  list in ascending order (level 0 first), i.e. the native Score interface; the native 0.6B adapter maps it back to
  options keyed `0`–`4`. Prompts are gold-free `id`/`state`/`questions` lines; gold lines follow HT-DEV's format
  (`v2/eval/htdev/build.py`). Panel item ids are `score5-` + the first 16 hex of `sha256("20260929:id:" + row id)`.
- Outputs on node A: `/data/dev2/private/panels/goldfree/score5-dev.prompts.jsonl` (0644),
  `/data/dev2/private/panels/gold/score5-dev.gold.jsonl` (0600), and `/data/dev2/runs/eval/score5-dev/build-v1/` with a
  MANIFEST (counts per level / axis / language, source sha, per-row id hashes, fit-pool size) and the selected source row
  ids + input hashes (`panel-rows.jsonl`) that fitting code must exclude. The option-key / position leak audit runs on
  the built files before registration; the panel is registered by hash in `v2/eval/panels.py` DEVELOPMENT.

## Metrics per model

Level histogram, modal share (top level's share of answered items), rare levels (< 2% of answered items), invalid/missing
count (counts as wrong), accuracy with 95% item-bootstrap CI (2,000 draws, seed 20260929), macro-F1, quadratic weighted
kappa, always-modal baseline accuracy (0.20 on the balanced panel).

Details: the predicted level is the adapter's Score point (`benchmark.score.evaluate_answer`); macro-F1 is over the five
gold levels with missing answers wrong (`v2.eval.sealed.score.macro_f1`); QWK is over answered items
(`v2.eval.sealed.score.quadratic_kappa`). The "answered items" denominator excludes invalid/missing answers.

## Flags

- **COLLAPSE** if modal share >= 0.60 or >= 2 rare levels;
- **WARN** if modal share >= 0.40 or exactly 1 rare level;
- **NO-SIGNAL** if the accuracy CI lower bound <= 0.20.

COLLAPSE supersedes WARN; NO-SIGNAL is reported independently.

## Validation

- Validation models: m6-mxcx-soup, m6-mxcxa-soup, m6-cx-soup, m4-t-a7-soup.
- Prediction (not a gate): mxcx COLLAPSE; cx COLLAPSE or WARN; mxcxa and T soup neither.
- The check 'works' if it flags mxcx and does not flag mxcxa and the T soup; otherwise it is reported as not detecting
  the collapse, plainly.
- Each model runs with its formal run's adapter spec (`v2/06b/records/adapters/dev2-06b-causal-8k.json`, 8,192-token cap,
  over-budget invalid), image, `model_id` extra and revision (export manifest hash); the m6 soups reuse the frozen Triton
  autotune snapshot their formal runs used (per-run copy); `m4-t-a7-soup` runs without autotune settings, as formally.
  `m6-cx-soup` has no formal run; it uses the m6 settings.

## Compute

Short shared-lease jobs on node A, preferring GPU5, then GPU0-1; each <= 15 min; memory headroom checked before launch;
no running job paused or disturbed; stop and record if 0.5 GPU-h would be exceeded.

## Amendment 1 (before the build and before any model run)

Selection detail only (thresholds and rules unchanged): the level × axis cells are filled round-robin, one row per cell per
round with cells in scarcest-first order, instead of cell by cell, so that a cell early in the order cannot use up the
groups a later cell of another level needs. Water-filling and the group-cap fallback are as above.
