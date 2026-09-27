# Kai2 0.6B: one development-only Choice/Noul recovery arm

This experiment responds to the post-key same-panel result in
`kai06-v3-postkey-result-2026-09-27.md`: Kai2 gained typed Score but lost
Choice, Noul and the v3 composite against its own Kai1 parent. The arm is
predeclared before its data is materialized or a GPU is used. It cannot by
itself supply an independent release result because the project's v3 labels
were previously accessed.

## Frozen data construction

- Initial weights: the completed own-Kai2 clean-v2 one-epoch native export,
  manifest `03d90e449078bacd18390ad52c3ba3af7cff86bb507c2c2f9088f0260305d1ab`.
- Inputs: clean-v2 Kai-native TRAIN SHA
  `fc8496c76c84b37f0a9308b13a270b69aab608defb8a9e4a3f62011352b6976c`
  and balanced-human5824 Kai-native TRAIN SHA
  `af03fc360392f6b521cf8659874cab98e3dcb6b76592e586f6d2a24b724f7300`.
  Both inputs and their manifests are checked byte-for-byte by
  `training.data.build_kai06_recovery1024`.
- Deterministic seed: `20260927-kai06-human-recovery1024-v1`. Choose complete
  component groups and never rewrite texts, instructions, options or labels.
  Select 64 human-labeled TweetEval Choice records each from hate, offensive,
  irony, emotion and sentiment (320 total). Replay 256 clean-v2 Choice,
  320 Noul and 128 Score native records. The 1,024 rows are one fixed arm;
  no quota, source or seed search after seeing outcomes.
  Clean replay components already represented anywhere in the human-source
  catalogue are excluded before deterministic selection; one failed CPU-only
  materialization exposed this cross-catalogue duplication before any output
  dataset, training run or development score existed.
- Continue for exactly 16 logical steps at batch size 64, microbatch 8,
  maximum full input 1,024 tokens, no truncation, seed 20260927, encoder
  learning rate `5e-6`, decision-head rate `2e-5`, minimum rate `5e-7`,
  weight decay `.01`, clip norm `1.0`, native FP32 parameters with BF16
  autocast loss. Save steps 0, 8 and 16 for diagnosis; step 16 is the only
  selectable trained checkpoint. No calibration fit; raw probabilities.

## Pre-GPU gates

1. Materialize the dataset once from pinned files. Require exactly 1,024
   unique rows, 1,024 or fewer complete source groups, type counts
   Choice 576 / Noul 320 / Score 128, and the five human buckets at 64 rows
   each. No group or original record may be split or duplicated.
2. Audit selected TRAIN against clean-v2 SELECT 700, CAL 700, typed DEV 1,600,
   CSS pilot 1,430, typed FINAL 1,600, CSS FINAL 6,547, and public231 using
   source ID/group checks, normalized exact text and the frozen approximate
   near rule. Any definite TRAIN/holdout content overlap stops the arm;
   uncertain semantic matches require manual review. Preserve exclusions and
   audit output privately. No gold labels are used for this audit.
3. Recheck rights per selected source. The existing human dataset manifest
   permits a noncommercial research scope and raw text remains private; its
   earlier attestation was tied to another run, not a blanket permission for
   a public Kai release. The present arm stays research-only until the exact
   resulting weights have a separate redistribution review.
4. Confirm exact zero-step native output matches the selected Kai2 export,
   all 1,024 examples fit the native input length, one-GPU runtime stays
   numerical-stable, and source train code/revisions are locked. A failed
   preflight stops training and is logged without changing thresholds.

## Development decision and stop rules

Use the clean-v2 SELECT 700 for training diagnostics, and evaluate the fixed
step-16 export on typed DEV and CSS pilot with the same native adapters.
Require SELECT row hard accuracy ≥0.7271 (Kai2 control 0.7371 minus 0.01),
typed DEV Choice/Noul mean accuracy ≥0.37875 (Kai2 control 0.35875 plus 0.02),
typed DEV Score accuracy ≥0.48 (Kai2 control 0.52 minus 0.04), and CSS pilot
median task macro-F1 ≥0.17977 (above Kai1 control 0.17477). The CSS pilot is
diagnostic, not independent proof: it has been used previously. If any
threshold fails, retain the failure and stop this arm rather than selecting
step 8 or changing the mixture.

Even if all development gates pass, require a new, source-disjoint, untouched
human-label transfer check before another v3 run can support release. The
project's current FINAL panel can then provide same-panel comparability only,
with its post-key status clearly marked. This experiment does not reuse public
JevBench or known FINAL scores for selecting the next checkpoint.
