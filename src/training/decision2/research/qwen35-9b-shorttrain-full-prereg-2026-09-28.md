# Official Qwen3.5-9B: short-TRAIN complete development arm

**Prospective status: no update of this arm has started.** The independently
saved one-update/reload smoke passed on the group-filtered TRAIN: one finite
update, 700/700 valid SELECT answers, then 32/32 native reload items with zero
category changes and zero probability drift. That technical checkpoint is not
the initializer for this arm. The failed 8,192-token arm remains unchanged.

## Frozen cell

- Fresh official general posttrained `Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, source hashes in the
  [original admission](qwen35-9b-official-posttrained-native-preflight-2026-09-27.md).
  A new 256-dimensional dynamic-option Decision head is initialized with seed
  `20260926`. No third-party Decision weights or smoke checkpoint initialize
  it. The checked-in trainer's seven source files retain the byte hashes in
  the one-update provenance; the native reload verifier is at `a5d3e4e9a`.
- Group-preserving rights-clean v2 filtered TRAIN 7,324 rows / 5,261 groups /
  3,579,176 native tokens, SHA-256
  `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`.
  Original SELECT700 SHA `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL700 SHA `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  TRAIN max 4,089 native tokens, SELECT/CAL unchanged. Group removal loses
  14.67% of old TRAIN tokens, mainly long composition; no long-context
  improvement claim follows from this variant.
- Exactly one epoch, **458 optimizer updates** (final partial accumulation),
  microbatch1, accumulation16, max length4096, no gradient checkpointing,
  rank16/alpha32/dropout .05 LoRA, peak adapter LR 1e-4, head LR 2e-4,
  AdamW weight decay .01, gradient clip1.0, warmup .05, cosine tail,
  CE + .5 Brier. BF16 backbone compute, FP32 LoRA/head/loss. No replay,
  teacher, extra data, CAL selection or inherited optimizer state.
- Evaluate and save SELECT at steps 64,128,192,256,320,384,448,458. Select
  one BEST by family-macro accuracy descending, Brier ascending, earliest
  step. No alternate checkpoint search on DEV, pilot, public or formal panels.
  One isolated GPU; pinned offline ROCm image
  `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`;
  wall cap 4.0 GPU-hours including load, SELECT and saves. On nonfinite
  loss/gradient, OOM, native fault, missing source, code/data hash difference,
  failed save or cap, stop and retain artifacts. Do not change this recipe to
  rescue a weak outcome.

## Development promotion only

Reload the single SELECT BEST in a fresh process and require first-32 native
SELECT parity: zero category changes, p99 probability drift <=.005, maximum
<=.02. Then run complete typed DEV1,600 and CSS pilot1,430 once, under a
documented native adapter, with own Lux1 and pinned JPT-9B controls on the
same development inputs. Promote only if `100*sqrt(T_dev*H_pilot)` exceeds
matched Lux1 by at least 2.0 points and the invalid/overbudget excess is no
more than one percentage point. Report Choice, Noul, Score, probability
quality, long-input failures and each human task, including regressions.

Only a development pass permits a separate frozen post-key JevArena v3 and
public JevBench evaluation. The full arm remains HOLD for release until that
same-panel evidence, paired uncertainty, package readback and independent
corroboration are available. Decision Index values select JPT as a peer but
never substitute for its native same-panel score.
