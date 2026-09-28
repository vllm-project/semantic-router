# Decoder Milestone 2 — amendment 3 (0.8B release path under the newer coordinator rules)

Parents: prereg `143cfc214`, amendments 1 (`f316bad37`) and 2 (`181ec58a0`),
candidate record `f02b73966`. Written 2026-09-28 ≈19:15 UTC+8 while E8F-s2 and
E8F-s3 train; no E8F-s2/s3 result exists yet. Adopts three coordination
rules that postdate the prereg:

- 17:15 **small-tier seed rule:** 0.8B candidates need ≥ 3 seeds; recipes are
  chosen by the seed mean; the artifact is a uniform weight average ("soup")
  of same-recipe, same-init seeds if it is ≥ the seed mean on development
  panels, otherwise the median seed.
- 17:55 **packaging limit:** package at the largest input limit the runtime
  supports (≥ 8,192) and compare against a same-limit 1.0 control.
- 18:45 **clean CAL:** release candidates fit their final calibration on
  CAL698 (`19cc1a8c…`, CAL700 minus one GoEmotions group).

## Changes

1. **E8F-s3** (seed 20260928, identical configuration) launched at 19:05 on
   node B GPU2 (third seed).
2. **Artifact rule (replaces "s1 is the candidate"):** soup =
   uniform FP32 average of the SELECT-chosen BEST checkpoints of E8F-s1/s2/s3
   (`v2.dec.soup`; same init, same recipe). Development comparison on node B
   (same image): the soup's proxy P (typed DEV + CSS pilot, CAL700
   temperatures fitted with `calibrate_ckpt`, exactly as the seeds' postrun)
   vs the mean P of the three seeds. Soup if P_soup ≥ mean P; otherwise the
   seed with the median P.
3. **Final calibration** of the artifact: CAL698 per-type temperatures
   (`v2.dec.calibrate_ckpt`, `selection_policy = frozen_checkpoint`, now
   accepted by the shared loader, commit `0a399c1d9`), fit at the package limit.
4. **Package limit 16,384 tokens** (the Qwen3.5 text runtime has no lower cap;
   the Lux-family practice), removing the 18 over-8,192 CSS15 `tropes`
   failures. Formal: artifact at 16,384 + Eos 1.0 same-limit control at 16,384
   (`adapter-spec-infer-1p0.json`, Eos ships no temperatures → 1.0), node A.
5. **Release reading:** the artifact qualifies if its v3 paired 95% lower bound
   vs the adopted Eos 1.0 run (42.547) is > 0 on node A, the E8F seed mean on
   the development proxy is above Eos 1.0, and no decision type collapses.
   The per-seed 8,192-token formal runs (s1 done; s2, s3 when complete) are
   reported as seed-robustness evidence; B8F-s2 keeps its preregistered
   8,192-token formal run (information only; B8F-s1 did not qualify).
