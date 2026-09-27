# Official Qwen3.5 4B Base: native admission before training

This is a prospective weight-lineage and runtime preflight. It uses no
Decision 1.0 or third-party Decision weights as an initializer and no typed
FINAL, CSS15 or public231 answers. The existing own-Nox/third-party 4B
experiments retain their recorded outcomes and are not repeated here.

## Frozen source and data

- Official `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`, downloaded by the HF CLI
  and pinned before this run. Its `config.json` SHA-256 is
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
  two weight-shard SHAs are
  `df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712`
  and `590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef`;
  tokenizer JSON SHA is
  `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Rights-clean v2 TRAIN 7,455 SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
  CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  Preserve their source/group isolation; CAL is never a model selector.
- Native segmented-option prompt, Qwen3.5 text body and dynamic-option
  head from the signed local Decision 2.0 branch. Stage is `base`, new head
  dimension 256, seed 20260926, maximum input length 8,192, BF16 backbone
  computation and FP32 decision head. This is a separate architecture arm
  from earlier own-Nox training.

## Ordered checks and stop rules

1. Audit all TRAIN, SELECT and CAL native token lengths without truncation,
   by type and language, at the pinned source tokenizer. Stop on any
   overlength or split-integrity problem; record the audit JSON and SHA.
2. On one reserved GPU, run two independent fresh source starts and complete
   SELECT700 inference, including Choice/Noul/Score. Require 700 valid each,
   identical ordered input IDs, zero category changes, p99 probability drift
   at most .005 and maximum at most .02. Record both complete output hashes,
   model/runtime hashes and GPU seconds. Combined cap: **0.30 GPU-hour**.
3. If step 2 passes, run exactly one optimizer update using the proposed
   LoRA plus head objective: LoRA rank 16, alpha 32, dropout .05, LR `1e-4`;
   head LR `2e-4`; CE + 0.5 Brier; microbatch 1, accumulation 16,
   gradient clip 1.0, weight decay .01 and seed 20260926. Require finite
   loss/gradient, save and a native 32-row same-batch reload with zero
   category changes, p99 drift at most .005 and maximum at most .02. Cap:
   **0.10 GPU-hour**. Preserve any failed attempt; no fallback runtime or
   altered threshold within this version.
4. A passing smoke only authorizes a **separately signed** one-epoch budget,
   checkpoint selector and open-development gate. The one-step smoke does
   not initialize the full run. No 4B JevArena v3 or release claim follows
   from SELECT; post-key formal comparisons need separately frozen packages
   and prediction files.
