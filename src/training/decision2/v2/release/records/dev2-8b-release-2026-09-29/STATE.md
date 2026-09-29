# DEV2.0-8B (9B tier, candidate K-a13) release: state file

Updated: 2026-09-29 14:45 UTC+8 (release-9b worker 1)
Branch `xunzhuo/decision-2-training-release-9b` (worktree `/home/xunliu/code/vllm-sr-dev2-release-9b`, merge-only).
Gist file `07f-decision-2-release-9b.md`. Scope: release pipeline up to, NOT including, `--collect`.

## Fixed facts (verified read-only on node A)

- Candidate K-a13 = 1/3 x K soup (`20dbb8999f21…`, three seeds of own Lux 1.0 full fine-tuned on 60M XL r2 tokens,
  1.0 x KL(own Lux) on every row) + 2/3 x Lux 1.0 zero-step export. Soup `runs/9b/m4/K-a13-build/soup` (16 files, all F32,
  31.78 GB), model_sha256 `b9d973b3ef55…`, manifest `SHA256SUMS` sha256 `6913eb61836a…` (23 files incl. `m4/K-a13-cal`).
- Loaded parameters 7,940,895,744 = text backbone 7,936,684,544 (embeddings 1,017,118,720 and layers/norm 6,919,565,824)
  plus decision head 4,211,200. No lm_head, vision or MTP tensors. Name **DEV2.0-8B** (`name_basis`, 27B-branch `2417f4033`).
- Scored run `runs/9b/formal-m4/K-a13-16k` (+ `-16k-mlx`): image `decision20-train-fast:host2` = `f83b1d10…`, FLA 0.5.2,
  causal-conv1d 1.7.0, 16,384 tokens, BF16 backbone / FP32 head, env HIP_FORCE_DEV_KERNARG=1 + TRITON_CACHE_AUTOTUNING=1,
  Triton cache `formal-m4/triton-cache` (copy of the frozen `formal-m3` cache `af623300…`, final tree `5604ffdc5f19`, shared
  with mlx-diag). Scored WITH CAL698 temperatures (`m4/K-a13-cal/calibration.json` sha `65297c6d…`, fitted at 8,192 tokens:
  Choice 1.4203 / Noul 1.0368 / Score 0.5626). SEAL `11abc1cc7818…`, REPORT `b0b805ccc3ef…` (v3 67.737).
- Comparators: Lux1 native 16K `runs/eval/m1/d1-lux1-autotune-cache` 65.808 (gate; `m1-adopt/lux1` is an older image);
  same-renderer `runs/9b/formal-m3/lux1-16k-shared` 65.231; Nimble v2 `runs/eval/m2/q6-nimble2` 62.056 (apache-2.0, card);
  JPT-9B `runs/eval/m1-adopt/jpt9b` 60.994 (CC BY-NC, excluded from the card). mlx: Lux1 eval `eval/m2/mlx/x-lux1` .8278,
  same-renderer `9b/formal-m3/lux1-16k-shared-mlx` .8315, K-a13 .8224.
- Eval-track 9B gate record: not out at 14:40 (card placeholder).

## Done

- 14:05 worktree from `a6b8f0572`; merged the 27B worker's shared-module prefix `0347ccc2e` -> `5dcb8fc6e`; tests 60/60.
- 06:03Z calibration rule: `runs/release/dev2-8b/devcal-20260929T060324Z/dev2-8b.json` adopt=false: CAL698
  worsens typed-DEV Brier .05370 -> .05809 and CSS-pilot Brier .55352 -> .55531 (ECE improves .0157 -> .0107 and .1326 ->
  .0577); 0 answer changes. -> **T = 1**.
- T = 1 derivation (`ops/derive-t1.sh`, mirror `08382e424`): 0 answer changes on 4 panels; derived run
  `runs/release/dev2-8b-t1-derived` REPORT `66106ef9…` (v3 67.737), SEAL `d00e453b…`, PAIRED-vs-adopted-1.0 `ae559e9a…`
  (+1.929 [+0.607, +4.144]), vs same-renderer +2.507 [+1.037, +4.599], vs Nimble v2 +5.681 [+3.190, +10.090];
  mlx `runs/release/dev2-8b-t1-derived-mlx/mlx-diag.score.json` `0ab43a76…`.
- BF16-storage copy (`v2.release.bf16_copy` `7e20cd36b`, `ops/bf16-copy.sh`): `runs/release/inputs/dev2-8b-bf16/checkpoint`,
  identity `b1ed5a71038b…`, 248 projection tensors BF16 (6,918,504,448 params), 178 FP32 (1,018,180,096) + FP32 head,
  17,946,656,577 bytes; receipt `bf16-copy.json` `ccdcf1c3…`; unit tests passed in the image.
- Banner `DEV2.0-8B-owl-banner.png` `7263ca30…` (`efd36ac20`); specs `dev2-8b-release.json` (BF16 package) and
  `dev2-8b-bf16-parity-staging.json` (`aa957cec7`). HF headroom 06:37Z: 44.63 GB used, 55.37 GB free.

## Running

- 06:33Z node A GPU6: `ops/bf16-parity.sh` (mirror `aa957cec7`), work `runs/release/dev2-8b-bf16-parity-20260929T063342Z`,
  log `runs/release/dev2-8b/logs/bf16-parity.log`; no upload; parity typed-final/css15/public231/mlx-diag, tolerance 1.

## Next

1. BF16 decision: adopt iff 0 answer changes on all four panels; else switch the spec to the FP32 soup (identity b9d973b3).
2. Build-draft decision -> `runs/release/decisions/DEV2.0-8B.decision.json`; headroom; `release.sh --upload` (no --collect);
   card HTTP + link checks; `gate evaluate`; verified draft; record + gist 07f; merge into integration.

## GPU-hours (node A GPU6)

- bf16 parity run from 06:33Z (running).
