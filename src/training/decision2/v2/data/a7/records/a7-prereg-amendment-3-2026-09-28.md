# A7 preregistration amendment 3: recovering the held Stage1/2 Noul and Score rows (2026-09-28)

Committed before any `a7-rec10-v1` row is built. It adds rule 7e and one new
sub-arm, **A7r**, built by `v2.data.a7.recover` from the pinned sources of
`specs/a7-dec10-v2.nodeB.json`. The `a7-dec10-v3` files are not changed.

## What was held and why

Rule 7b held 19,075 rows (Stage1 6,486 Noul + 4,887 Score; Stage2 6,076 Noul +
1,626 Score) whose option keys are opaque: `result_<n>` numbered in construction
order (Noul 5,689, Score 2,867) or random words (the rest). In Stage2,
`result_1` is the gold in 1,984 of 2,740 `result_<n>` Noul rows, so these keys
leak like the Choice keys of amendment 1. Every held row has one of six Noul
description pairs (authorization, policy and transition-set templates in
English and Chinese) or rubric Score descriptions that state the level and the
condition range.

## Rule 7e (A7r)

Keys are derived from the option descriptions only, never from the label:

- **Noul:** the description pair must be one of the six whitelisted
  (affirmative, negative) pairs in `recover.NOUL_PAIRS`; the affirmative option
  is keyed `true`, the negative `false`. Option order and label are unchanged.
- **Score:** each description must match "Level k: between a and b conditions
  hold, inclusive." or "等级 k：成立的条件数为 a 到 b（含边界）。"; the levels must
  be exactly 0..K-1 and the ranges contiguous from 0. Options are reordered by
  level and keyed "0".."K-1"; the label moves with its option. The gold range
  must contain the number of true conditions in the (JSON) state, an
  independent recomputation of the oracle; a disagreement drops the row.
- Anything else stays held (counted by reason).

After mapping, the original keys and order are gone from the input, so neither
the construction-order numbers nor the word keys can reach the model; the
original keys and input hash stay in `audit_metadata.a7.rule_7e`.

## Build, screens and admission (as prereg §2 unless stated)

- Deduplication by input hash against every `a7-dec10-v3` file and among the
  recovered rows; identical inputs with different gold descriptions are dropped.
- Isolation: rows sharing a group, input hash or normalized state with SELECT700,
  CAL700 or a v1 arm held-out slice are dropped. Components use the union-find of
  `build_a7.build` over all source rows plus the frozen files; a component that
  already holds `a7-dec10-v3` rows inherits their partition (TRAIN or AHO),
  otherwise the prereg AHO hash rule applies (never AHO when touching A0 or a
  published TRAIN file).
- Lexical overlap (four methods) against PI-v3 without TRAIN roles (the
  relocated 47-role manifest of amendment 2), whole-group quarantine; native
  budget 8,192 tokens; shortcut rule 7c (A7r counts as generated: family × type
  cells with ≥ 30 rows whose state-removed or option-only accuracy exceeds the
  cross-validated majority by more than 5 points are dropped); the GPU embedding
  scan with the settings of amendment 2 before A7r is released for training
  release candidates (until then it is marked development-only).
- Freeze, isolation (with the `a7-dec10-v3` files as extra partitions) and
  upload as prereg §2.10 and §4; A7r is published under `v2/a7/arms/A7r/`.
