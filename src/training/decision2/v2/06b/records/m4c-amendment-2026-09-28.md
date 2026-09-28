# 0.6B Milestone 4 amendment (M4c): adopt mixture v2 as a third three-seed arm

Frozen 2026-09-28 before any V2 run and before control seed 3 launches. T s1 and C s1
were training when this amendment was written; nothing about them changes.

## Why

The research & data track announced the D10 recipes at 18:20 (private HF revision
`5c602c5a…`, `m2/mixtures/`) and recommends `mx-v2-full-M` for a 0.6B model at the 8,192
cap. Milestone 4 item 3 says to adopt mixture v2 as soon as it is announced. Its Lux
targets on the v2 rows (RP-v2) are still being produced, so V2 uses the canonical Lux
teacher on the A0s-r rows only (as T does) and gold labels elsewhere.

## Arm V2

`m4-v2-s1|s2|s3` = the (a2) recipe of part 2 with mixture **`m4-mix-v2`**: the shared base
A0s-r (identical to T and C) plus every non-A0s row of `mx-v2-full-M`
(`5b8d5d29…`, 48,687 ids) joined by id to the frozen v2 arm files (HF `ed87a03a` arms,
all eight TRAIN hashes equal the v2 registry; `V1S` = the pinned v1 A6g/A6h files).
Built once (`mixture.py`, `id_manifest`): **47,922 rows, 20,032,163 Qwen3 tokens** (T
19,999,143; C 19,996,635), file `0eedee75…`, row ids `1bd2841d…`; 21 languages; Choice /
Noul / Score rows 11,399 / 24,699 / 11,824; no row over 8,192 tokens (max 6,559); 13 rows
repeating an earlier input dropped. 2,996 updates at batch 16 (warmup 299); collapse stop
at update 1,124; GPU-hour cap 1.8 per run. Same head seed 20260928 and data-order seeds
as T and C.

## Changes to part 2

- **Control C runs two seeds (s1, s2); C s3 is cancelled before launch** to fund V2 inside
  the milestone budget. The effect rule of part 2 §3 applies to seed pairs s1 and s2, with
  σ_seed from the three seeds of the treatment arm.
- **Candidate recipes are T and V2.** The recipe with the higher three-seed mean P is the
  first candidate; if the other is within |ΔP| < 4 it is the second candidate (the tie
  rule), otherwise it is not sent. Each candidate artifact is the uniform soup of its
  three seeds if the soup's P ≥ that recipe's seed mean, else its median-P seed; floors and
  the P ≥ 30.5 gate of part 2 §3 apply unchanged. C is never a candidate.
- **Queue:** GPU1 runs V2 s1 after C s2, then V2 s3; GPU0 runs V2 s2 after T s3.
- **Budget:** part 2 cap raised from 8.0 to 9.5 GPU-hours (T 3 × 0.8, C 2 × 0.9, V2 3 × 1.0,
  readouts, soups and at most two formal runs ≈ 8.2).
