# DEV2.0-8B (9B tier, candidate K-a13) release: state file

Updated: 2026-09-29 14:20 UTC+8 (release-9b worker 1)
Branch `xunzhuo/decision-2-training-release-9b` (worktree `/home/xunliu/code/vllm-sr-dev2-release-9b`, merge-only).
Gist file `07f-decision-2-release-9b.md`. Scope: release pipeline up to, NOT including, `--collect`.

## Fixed facts (verified read-only on node A)

- Candidate K-a13 = 1/3 x K soup (`20dbb8999f21…`, three seeds of own Lux 1.0 full fine-tuned on 60M XL r2 tokens,
  1.0 x KL(own Lux) on every row) + 2/3 x Lux 1.0 zero-step export. Soup `runs/9b/m4/K-a13-build/soup` (16 files, all F32,
  31.78 GB), model_sha256 `b9d973b3ef55…`, manifest `SHA256SUMS` sha256 `6913eb61836a…` (23 files incl. `m4/K-a13-cal`).
- Loaded parameters 7,940,895,744 = text backbone 7,936,684,544 (embeddings 1,017,118,720 and layers/norm 6,919,565,824)
  plus decision head 4,211,200. No lm_head, vision or MTP tensors. Name **DEV2.0-8B** (`name_basis`, 27B-branch `2417f4033`).
- Scored run `runs/9b/formal-m4/K-a13-16k` (+ `-16k-mlx`): image `decision20-train-fast:host2` = `f83b1d10…`, FLA 0.5.2,
  causal-conv1d 1.7.0, 16,384 tokens, BF16 backbone / FP32 head, Triton cache `formal-m4/triton-cache` (copy of the frozen
  `formal-m3` cache `af623300…`, final tree `5604ffdc5f19`, mlx-diag shared it). Used CAL698 temperatures
  (`m4/K-a13-cal/calibration.json` sha `65297c6d…`: Choice 1.4203 / Noul 1.0368 / Score 0.5626). SEAL `11abc1cc7818…`,
  REPORT `b0b805ccc3ef…` (v3 67.737), paired vs Lux1 16K `e7419957…`.
- Comparators: Lux1 native 16K `runs/eval/m1/d1-lux1-autotune-cache` 65.808; same-renderer `runs/9b/formal-m3/lux1-16k-shared`
  65.231; Nimble v2 `runs/eval/m2/q6-nimble2` 62.056; JPT-9B `runs/eval/m1-adopt/jpt9b` 60.994 (CC BY-NC, excluded).
- Dev readouts (CAL698-tempered): `m4/K-a13-dev/dev.predictions.jsonl`, `m4/K-a13-css-pilot/css-pilot.predictions.jsonl`.
- Eval-track 9B gate record: not out at 14:10 (card placeholder).

## Done

- 14:05 worktree from `a6b8f0572`; merged the 27B worker's shared-module prefix `0347ccc2e` (name_basis, no-1.0 profile,
  board-sibling licence, banner --label) -> `5dcb8fc6e`; release tests 60/60, common tests OK.

## Next

1. Take node A GPU6 lease; mirror; calibration rule (dev_calibration) -> T decision.
2. bf16 copy test (autocast-exact conversion, no upload, parity on typed-final/css15/public231/mlx-diag).
3. Spec + banner DEV2.0-8B + build draft decision; headroom; release.sh --upload; verified draft; records; merge.

## GPU-hours (node A GPU6)

- none yet
