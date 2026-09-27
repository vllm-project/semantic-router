# Own-Nox 4B: fixed Choice human-label weight screen

**Prospective, open-development experiment.** This document fixes the arm
before any optimizer step or new score. No typed FINAL, CSS15, authored release
gold, or Hugging Face publication is part of this screen.

## Decision question and matched control

The completed own-Nox rights-clean-v2 control is retained unmodified. It
starts at `llm-semantic-router/Decision-1.0-Nox-4B` revision
`0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` (4,208,383,488 loaded
parameters) and trained one epoch on the exact 7,455-row TRAIN with
CE + 0.5 Brier. Its fixed step-128 SELECT result was 574/700, family macro
.805185, GoEmotions Choice 144/200 and Score 64/90. Its full step-466
development diagnostic was typed DEV 1,049/1,600, Choice 454/800, Score
367/400, and CSS pilot task-median macro-F1 .443807. The full arm did not
meet its development promotion threshold; it remains HOLD. The completed
human-only 5,824-row and structured 8,522-row continuations used different
data and selection rules, so they are descriptive controls, not matched
causal comparisons.

The hypothesis is that **more loss mass on human-labeled Choice rows in the
same TRAIN** improves open Choice and human-task transfer without needing a
new backbone, data source or inference adapter. The sole intervention is
1.5× loss weight on the 2,240 Choice rows with source IDs
`google_goemotions_official_train` (1,400),
`legacy:cosmos_qa` (448), `legacy:snli` (272), and
`css_flute_official_train` (120). Every other row, including all Noul and
Score rows, retains weight 1.0. Normalize each accumulation window by its
sum of weights. This changes the gradient target only: example order, 4,194,465
unpadded input tokens, 7,455-row one-epoch horizon and 466 planned updates
stay equal to the control. The weighted cohort is 2,162 distinct source
groups, all English. GoEmotions Choice and Noul share 1,400 groups; the
experiment intentionally weights only Choice in those groups.

CSS pilot tasks are SemEval stance, implicit hate and discourse. None is an
exact training source ID. The frozen data manifest reports zero raw,
normalized, group-ID, input-hash or bounded near-context matches between
TRAIN and CSS pilot. This supports source-ID disjointness; it does **not**
prove mechanism-level independence or rule out pretraining exposure. Typed
DEV shares some broad rule mechanisms with TRAIN, so it is development
diagnosis. No claim of blind transfer follows from this screen.

## Frozen identities and runtime

| Input | SHA-256 or fixed value |
| --- | --- |
| TRAIN 7,455 | `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| SELECT 700 | `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` |
| CAL 700, lineage only | `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a` |
| Rights/source/overlap manifest | `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8` |
| Nox 1.0 release manifest | `50c2f77c7c3f6c1efae7014ccfc4aedc3b185e52731ba721a1c6d6192668d945` |
| Completed control provenance | `496eff65241c0bd0d0f8b83cf377885994c6ceb5b8c22fb1ece5fd4c0def65a1` |
| Control zero-step SELECT predictions | `eabd788e4e656974e3f380cdff0d32f1ad0887ddb2034235b2c43b9c3f994de3` |
| Control trainer source | `b4414a5a3b4b2fbd3ec2ef3ad68e04480b6f695f8a05d239189dc033c1eb628d` |
| Treatment trainer source | `47902f9cc71ab8fc65cda9b43c029dab3c3217cf2503b615ea064a912bdc2476` |
| Runtime image | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |

Use the same rank-16/alpha-32/dropout-.05 LoRA, LoRA LR 1.5e-5,
head LR 7.5e-6, BF16 backbone/FP32 head, microbatch 1, accumulation 16,
max length 8,192, seed 20260926, warmup/cosine policy and fixed data order
as the completed control. The control's trainer, renderer, loader, plan,
loss and LoRA file hashes are in its preserved provenance. Apart from
the new exact-source weight and its audit fields, the local treatment
trainer is byte-identical to that archived control trainer. The treatment
runtime must recompute and freeze the source package inventory, data/code
hashes, encoded length maximum 6,596 and token count before step one.
No alternate checkpoint, label, source or weight may be selected after
observing a score.

## Stop gates and fixed observation

1. Use a newly empty output directory on a confirmed unoccupied card. The
   source model must load with the same native dynamic candidate head and
   resolve every model-file hash. The TRAIN/SELECT/CAL loader must find zero
   ID/group/input-hash intersections and exactly 2,240 weighted Choice
   rows. Abort on row/token changes, length overflow, altered options or
   unsupported device precision.
2. Before training, compare **all 700 zero-step SELECT predictions** to the
   preserved control: same ID, native input/token hashes and option keys;
   zero categorical changes, max absolute option-probability drift
   `<=1e-4`. This is stricter than checking aggregate accuracy. A failed
   comparison stops the arm without relaxing the threshold.
3. A separate one-update smoke may use `max-steps=1` only to establish
   finite loss and gradients, nonzero weighted Choice examples, no OOM,
   exactly one physical GPU, and a durable checkpoint. Its altered LR
   schedule makes it a numerical gate, never a matched score. Stop if its
   measured pace projects the 128-update screen above 1.0 GPU-hour.
4. Run the **full 466-step LR schedule** but stop externally only after the
   step-128 checkpoint and SELECT receipt are durable. This is the sole
   candidate. No step-32/64 or later checkpoint may replace it. Stop on
   nonfinite loss/gradient, changed row/token trace, changed native
   inference, resource collision, incomplete checkpoint or >1.0 allocated
   GPU-hour. The step-128 early screen passes only if SELECT has at least
   150/200 GoEmotions Choice correct, at least 574/700 total correct,
   family macro at least .805185 and Score at least 62/90, with 700 valid.
   These thresholds compare with the saved matched control step 128.
5. If and only if that screen passes, evaluate the fixed treatment step 128
   and untouched control step 128 on typed DEV1,600 and CSS pilot1,430
   **once**, with the same uncalibrated native adapter and no truncation.
   Report Choice/Noul/Score, every CSS task, Brier and invalid counts.
   Advancement to a *separate later* full-arm protocol requires treatment
   Choice at least control +12/800, CSS median macro-F1 at least control
   +.015, no individual CSS task F1 drop >.02, Score no worse than
   control −8/400 and no extra invalid answers. The exposed panels are
   not used to fit calibration or choose a checkpoint. CAL remains unused.

The short screen budget is approximately 0.2–0.4 training GPU-hour plus
at most 0.1 GPU-hour each for native control/treatment development
diagnostics if the SELECT gate passes. Record actual wall allocation,
GPU-hours, hashes, zero-step drift, one-step numeric values and all stop
reasons. A positive screen merely authorizes a new preregistered full arm;
it is not JevArena v3 evidence and does not qualify 4B publication.

## Read-only device preflight

An initial device probe set `ROCR_VISIBLE_DEVICES=2`,
`HIP_VISIBLE_DEVICES=2` and `CUDA_VISIBLE_DEVICES=2` together. ROCm
applied overlapping visibility filters and exposed no device; it loaded
no model and ran no optimizer. Separate probes using only
`ROCR_VISIBLE_DEVICES=2` or only `HIP_VISIBLE_DEVICES=2` each exposed
exactly one BF16-capable device. The treatment will use only the former,
with the full device mapping, after a fresh physical-card memory/process
check. This fixes device visibility before any model score or update; no
data, weight, numerical gate or selector changes. CPU loss/plan tests in
the pinned image passed (4 tests).
