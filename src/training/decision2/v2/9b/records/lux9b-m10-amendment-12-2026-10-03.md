# 9B M10 amendment 12: arm KIB4H, KIB4 at half learning rates (2026-10-03)

Written 2026-10-02 ≈21:45Z (10-03 05:45 UTC+8) by the M10 continuation worker, before any KIB4H data entry or GPU job,
and before any half-learning-rate 9B result exists. COORDINATION 05:18 makes half-LR arms of the best 9B recipe the
9B recipe pivot. The 4B half-LR soup read far above the released Nox; that result was disclosed before this was
written. The 9B evidence in the same direction is KIB4L2-a40 (backbone LR ×2), significantly below K-a13IB
(state 21:00Z).

## Arm

| Arm | Data | Change vs KIB4 | Seeds |
| --- | --- | --- | --- |
| **KIB4H** | KIB4's TRAIN `2e72bcfd…` / teacher `377f8878…` and data flags, locked on node B by `prep.sh alias-lock KIB4H KIB4` (every file re-hashed against KIB4's entry) | backbone LR 5e-6 and head LR 5e-5 (both halved, as in the 4B lever); everything else is KIB4's recipe | s1 = 20260926, s2 = 1 (KIB4 s1 / s2's seeds) |

- **Seeds.** The seeds are KIB4 s1 / s2's, so each half-LR seed is a seed-matched ablation of a released-point seed.
  This keeps it distinct from the arm factory's backlog arm `KIB4-lrh` (backbone LR only, seeds 15 / 16, amendment 9
  `3ba71aeae`). A later four-seed half-LR soup can take both.
- **Placement.** Node B GPU5 (s1, owner file empty) and GPU7 (s2, M10's released lease), under
  `m10/chains.sh` phase 4 (`M10_PHASE=4`). The pre-warm marker is M10's, from m10-KUP-s1.
- **Preflights and stops.** M10's rules: zero-step, one-step and `preflight_dec` before the full run; a failed
  preflight stops the arm; the seed cap is 4.5 GPU-h; node B's M10 training gate is 50 GPU-h. A failed seed is not
  rerun.

## Candidate and gate

- **`KIB4H-a40`** = `[KIB4H, KIB4H, Lux × 3]` (`post.sh KIB4H` with `M10_POST_SEEDS="1 2"`): the uniform FP32 soup of
  the two seeds' BEST at α = 2/5, the released point's construction. It is measured once as `M10-KIB4H-a40-bf16` on
  node B GPU5 / 7 (or any free M10 GPU) and gated vs `M10-KIB4-a40-bf16` (amendment 7's IF1).
  - Its a33 / a25 points are built by the same chain but measured only if budget remains.
- **Integrity.** The TRAIN is KIB4's, so audit `m10c` applies (0 item rows). The formal path and the release path
  are amendment 7's.

## Budget

The two seeds take ≈ 6.8 GPU-h and the Index run ≈ 2.7. At writing, the continuation had used ≈ 19.3 GPU-h. The
internal stop rule moves from 27 to 30 GPU-h, the approved total, so the KIB4H-a40 Index run can start. A release
(≈ 2 GPU-h) or more half-LR points need a further budget line from the coordinator. The state log asks for it.
