# Sol 2B rights-clean control

This is a separate size-matched control for the structured replay experiment.
It tests what the published Decision 1.0 Sol 2B weights can learn from a
TRAIN partition that removes every directly identified TweetEval and
MultiNLI-origin row in our new curriculum. It does **not** establish that
the pre-existing Sol weights or the upstream Qwen backbone have no historical
exposure to those sources. The synthetic final and CSS15 gold remain sealed.

## Frozen data and isolation

The audited rights-clean v1 builder retains 4,655 rows from a previously
frozen 8,255-row no-MultiNLI TRAIN and removes all 3,600 TweetEval rows by
whole source group. The TRAIN file SHA-256 is
`4973f999b7a19c69a8d7236ab941556e788a9ab69f208c015d3768fd7c86841b`.
Its new independent SELECT300 and CAL300 file SHA-256 values are
`1d564becab12717f7883c77131b8e8611a2e495c102a71789a1b1e5c2cf9afd4`
and `35c27a2a16271b7295afa2d65474dbdc7c64742fdce23ba26bb2d3792a0921d6`.
The manifest SHA-256 is
`a4ded3bc13f5dcc9cffbd98899714507f17d29b0f4084cc0319f030a32728bdf`.
TRAIN types are Choice 2,508, Noul 1,631, Score 516; SELECT and CAL each
contain Choice 120, Noul 90, Score 90.

The private manifest records source counts, license evidence, input/output
hashes, and 27 exact/approximate near-context audits. The TRAIN/SELECT/CAL
files retain the selected row IDs. All overlap
counts were zero for TRAIN/SELECT/CAL, the old SELECT/CAL, synthetic DEV,
CSS pilot, gold-free CSS15, and frozen RQ1/RQ2/RQ3 pressure prompts. This
is a reproducible near-duplicate screen, not proof of semantic independence.
The new SELECT/CAL are programmatic oracles; they are narrower than a natural
human-label calibration distribution. The 120 FLUTE TRAIN rows also make
that specific CSS task same-task supervised. No CSS15 labels are read.

## Matched training recipe

Initialize from the same pinned Sol 1.0 revision
`0665a41108e8f0b33a9515c98311c45947b99399` as the structured replay
arm. Use its dynamic candidate head and rank-16 LoRA (alpha 32, dropout
.05), CE + 0.5 Brier, one epoch, microbatch 1, accumulation 16,
8,192-token limit, LoRA LR `1.5e-5`, head LR `7.5e-6`, fixed seed
`20260926`, and SELECT every 32 updates. The only intentional changes are
TRAIN/SELECT/CAL data; the smaller TRAIN produces fewer updates per epoch.
SELECT chooses a complete checkpoint by family-macro accuracy,
normalized Brier, then earliest step. CAL labels are audit-only until
selection freezes; native type temperatures then fit on the separate CAL300.
DEV1600 and CSS three-task pilot1430 measure the independently calibrated
checkpoint. No final gold is used for training, checkpoint choice or pilot
evaluation.

This control can be compared to Sol 1.0 and the prior targeted/human LoRA
arms on the *same* DEV/CSS pilot items. Its SELECT score cannot be compared
numerically with those arms because the frozen SELECT items differ. In
particular, CSS transfer success is uncertain when SELECT/CAL contain only
new synthetic oracles and most human-label training sources are absent.

## Frozen result

The GPU run completed all 291 planned updates and selected
`checkpoint-0000288`. Its `COMPLETE.json` SHA-256 is
`bc0b37bc6b7fd179dc835d2e52bdf830319c060950cd767352e54b725ed4a278`;
the full selected model fingerprint is
`f9dda160b6b54af6bc5286721ec5ad076247dade9d57b33bb41538e9efc392ed`.
The separate native CAL300 receipt is
`2f58225f5c2f1504103b626020a86b64b9237080df38307a6cc0909048c68c6a`.
It fitted Choice `0.790025678`, Noul `0.05` (the lower search bound), and
Score `0.648732637`. CAL accuracy was 239/300 (79.67%). The boundary
temperature, combined with a programmatic CAL set, is evidence that this
calibration is not a reliable natural-transfer fit.

The independently calibrated native adapter was valid on all 1,600 DEV and
1,430 CSS pilot questions. DEV was 952/1,600 (59.50%) overall:
attribute gate 261/400 (65.25%), rule precedence 224/400 (56.00%), set
307/400 (76.75%), and transition 160/400 (40.00%). Its prediction receipt
SHA-256 is `f238cafed7f208710854db3ae85c08a847f0c19d3db0507c5fb067fbdc06de02`
and score SHA-256 is
`8a4f1d0dfb06166fde189a0ed6e42027734c5e2a3d5d8f982bf1cb72592af47f`.
CSS pilot was 565/1,430 (39.51%) micro accuracy, with median task macro-F1
`0.30712`; task correct counts were discourse 169/497, implicit hate
152/498, and SemEval stance 244/435. Its prediction SHA-256 is
`7f4d317128f9fdd9e4a6128bdbf538a6b1034adafe713abafb91af0252f56ce1`
and score SHA-256 is
`9c49fd5cd1edd5c316dcfa9fcdc78808b2f82f6479627e8984c1b08377dd9bc4`.

For the matched Sol 1.0 baseline, DEV was 58.94%, CSS micro accuracy
38.46%, and CSS median task macro-F1 `0.31554`. This clean control adds
0.56 DEV and 1.05 CSS accuracy points, but loses `0.00842` CSS macro-F1.
The previous targeted3024 arm reached 59.13% DEV, 42.52% CSS micro, and
`0.34763` CSS macro-F1. Thus this control is an informative source ablation,
not a broad transfer improvement or release selection.

On the pinned *public-only* JevBench231 diagnostic, the same native
calibrated checkpoint was strictly valid on all 231 questions but scored
160/231 (69.264%): easy 48/48, standard 66/72, hard 46/111. Tier-macro
accuracy was 77.703%, Brier `0.25873`, and pmax ECE `0.23614`.
Sol 1.0 was 161/231 with Brier `0.20545` and ECE `0.10394`. The poorer
calibration is consistent with the synthetic-only CAL300 fit reaching its
Noul temperature lower bound; the public panel is not the sealed official
JevBench composite. Prediction, companion manifest, and score SHA-256 values
are `e3c2641fd30c4ba36045cc94a19ea67a2cb14502fb2dd1d26d310d71cb928b18`,
`eb665350ba533244e12c64d6815dd5faa35f10dd9eba859adeac313ce79e35fb`,
and `9f6dc9baf90e0f69943714ba2fe885a78842a18e30b9a524239991e05670c9bd`.
