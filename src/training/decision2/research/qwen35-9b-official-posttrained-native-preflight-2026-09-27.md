# Official Qwen3.5 9B posttrained: native admission

Prospective source and numerical preflight for an independent 9B arm. It
does not initialize from a third-party Decision model or inherit the
structured8360 Lux research checkpoint. The existing Lux 1.0 and BEST224
results remain read-only controls. No typed FINAL, CSS15 or public231 answers
are used here.

## Fixed source, data and runtime

- Official `Qwen/Qwen3.5-9B` general posttrained revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0. Its pinned
  `config.json` SHA-256 is
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`;
  four weight-shard SHAs are
  `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`,
  `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`,
  `7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`,
  and `b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`.
  The tokenizer JSON SHA is
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
- Rights-clean v2 TRAIN 7,455 SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  The source tokenizer admits every item at max length 8,192: TRAIN has
  4,194,465 encoded tokens and maximum length 6,596; SELECT/CAL have 0
  overlength. Read-only audit SHA-256 is
  `67033387a7a7475d6dd81370487e94b4b808e8cf4c6be0777360ae7a2a6708f7`.
- Native Qwen3.5 text model, new 256-dimensional dynamic-option head,
  segmented option/query prompt, seed 20260926, BF16 backbone computation,
  FP32 parameters and head, same local signed trainer. The proposed first
  complete arm, if admitted, is LoRA rank16/alpha32/dropout .05 plus head,
  CE+0.5 Brier, 1e-4 LoRA LR, 2e-4 head LR, microbatch1/accumulation16.

## Ordered admission

1. Recheck full source/data SHA map and split isolation in the runtime. Use
   one exclusively reserved GPU and two independent fresh source starts on
   SELECT700. Require all 700 valid in each, same ordered native inputs,
   zero category changes, p99 probability drift ≤.005 and maximum ≤.02,
   including all Choice/Noul/Score rows. Combined cap 0.45 GPU-hour.
2. If step 1 passes, run exactly one LoRA/head optimizer update from a new
   independent source initialization. Require finite loss/gradient, save,
   and same-batch first-32 SELECT native reload with zero category changes,
   p99 drift ≤.005 and maximum ≤.02. Additional cap 0.15 GPU-hour.
3. Only a passing smoke authorizes a separately frozen complete one-epoch
   training arm and its SELECT-only checkpoint rule. A weak baseline or one
   update does not trigger formal v3 or model release. The matched comparator
   is own Lux 1.0 and pinned JPT-9B; Index values guide peer choice only.

Stop on any missing source file, invalid type, OOM, nonfinite result, timeout,
or failed native/parity gate. Preserve unsuccessful receipts and do not change
the rule within this version.
