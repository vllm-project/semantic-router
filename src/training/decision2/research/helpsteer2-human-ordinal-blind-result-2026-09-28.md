# HelpSteer2 correctness-to-Score blind pilot: HOLD

This is a **data-construct screen, not a model evaluation or training result**.
No GPU time was used. The source is the pinned HelpSteer2 TRAIN correctness
grade (0–4). The proposed native System One Score question would ask about a
candidate response's factual correctness and completeness. It is distinct from
the project's existing three-level rule/evidence Score tasks.

## Frozen comparison

- Pilot: 24 candidate responses in 12 independent prompt groups, two responses
  per group. Six distinct source-grade pairs were sampled with one group where
  the higher source grade had the longer response and one where it had the
  shorter response. This is a deliberately small quality probe, not a random
  performance estimate.
- Blind packet SHA-256:
  `cfc0df08d7bb12bcc67918f64d1fbc6f29199acb86a9e6e24a21b44505aa00af`.
  Separate key SHA-256:
  `d996c64cbfb68addaacbfbdaba5d3388bb7c7a9db2dda0c5e06a1a15cc65b097`.
  Sealed independent review SHA-256:
  `042e39812251204e91a5180dd6b2064175f9a3388b4fd96128ab97a6736c7787`.
- The frozen packet and key were written at 2026-09-27 18:10:04 UTC; the
  answer-blind review file was sealed at 18:22:37 UTC. Its declared input hash
  matches the packet, and it declares no key access. File times and that
  declaration document the order of this comparison; they cannot alone prove
  that a reviewer had no other access to source labels.
- The private aggregate-only receipt is mode 0600, SHA-256
  `52e5a230be59206565e317dd312b9cf6ca1079837f1c92d56882b058d38b3bf1`.
  The comparison code is `training/data/compare_helpsteer2_blind.py`; raw
  prompts, responses, IDs, review prose and the key stay private.

## Observed agreement

| Measure | Result |
| --- | ---: |
| Exact 0–4 grade agreement | 8 / 24 (33.3%) |
| Within one grade | 18 / 24 (75.0%) |
| Disagreement by at least two grades | 6 / 24 (25.0%) |
| Mean absolute grade difference | 1.00 |
| Blind minus source mean grade | −0.75 |
| Quadratic weighted κ | 0.480 |
| Pair order concordant / tied / reversed | 9 / 3 / 0 of 12 |
| Concordant when source-higher response longer / shorter | 5 / 6; 4 / 6 |

The source assigns grade 4 to six pilot responses; the blind reviewer assigns
grade 4 to none. Review flags overlap: rubric mismatch 14, unverifiable facts
12, ambiguity 10, length/style shortcut 5, multi-turn delimiter 2. Overall,
22 of 24 responses carry at least one flag. The blind review explicitly judged
the construct unsuitable as a clean native ordinal correctness corpus: many
requests permit multiple acceptable responses, external facts cannot be
verified from the item, and response form can cue a grade. These counts are
quality-screen signals from **one AI reviewer**, not human adjudication.

## Leakage and decision

The prior gold-free overlap audit found two HelpSteer2 prompt groups (six
responses) exactly or nearly matching protected CSS15 states. The 12-group
pilot had no ≥0.60 five-gram match in that audit; this limited scan cannot
prove semantic source independence. The suspect groups must be quarantined
before any possible future partition.

**HOLD:** do not add these rows to TRAIN/SELECT/CAL, call this a validated
Score training source, or claim cross-source transfer from the pilot. Pair
ordering is promising enough to investigate later, but 12 pairs and the grade
disagreement cannot establish a reliable absolute five-level target. If this
line is revisited, first obtain independent rubric adjudication and verify
factual answerability and source disjointness. Separately test evidence-grade
and retrieval-grade Score sources under their native criteria; do not conflate
them with HelpSteer2 response correctness.
