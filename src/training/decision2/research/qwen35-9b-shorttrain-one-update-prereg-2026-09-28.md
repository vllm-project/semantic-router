# Official Qwen3.5-9B short-TRAIN: one-update admission

**Prospective status: no optimizer update has started on this subset.** This
cell tests whether the exact trainer, one real update and native package reload
work after the [CPU and synthetic admission](qwen35-9b-short-context-admission-result-2026-09-28.md).
It does not continue the prior failed 8,192-token run or use its weights.

## Fixed inputs and recipe

- Fresh initializer: official general `Qwen/Qwen3.5-9B` posttrained revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with the source file
  hashes in the original [admission](qwen35-9b-official-posttrained-native-preflight-2026-09-27.md).
  The initializer is not a third-party Decision model.
- Group-preserving short TRAIN: 7,324 rows, 5,261 groups, 3,579,176 native
  tokens, maximum 4,089, SHA-256
  `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`.
  Original SELECT700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
  and CAL700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`
  are unchanged. No final, public or pilot labels enter the arm.
- Use the signed local trainer at `d8442b6e9`, with the exact seven training
  module SHA-256 values already present in the earlier 9B provenance. One
  offline BF16 ROCm image pinned as `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
- Exactly one fresh LoRA/head optimizer update, seed 20260926, microbatch 1,
  accumulation 16, maximum native length 4,096, **no gradient checkpointing**,
  rank16/alpha32/dropout .05, LoRA LR 1e-4, head LR 2e-4, AdamW weight decay
  .01, CE + .5 Brier, BF16 backbone compute and FP32 trainable parameters.
  No replay, teacher or additional data. The absence of checkpointing follows
  the successful synthetic cell; it is a separately named variant, not an
  unrecorded resume of the earlier full-arm recipe.

## Ordered gates and decision

1. Before launch, verify the source, data, selected local code files, image,
   split isolation, fresh output directory and exclusive single GPU. Stop on
   any difference. The wall cap is 900 seconds, including load, SELECT and
   save, on one accelerator (0.25 GPU-hour).
2. Require exactly one completed update, finite loss and gradient, 700 valid
   SELECT answers before/after, a complete checkpoint and no OOM, native fault
   or time cap. SELECT is only a numerical diagnostic, not a release score or
   checkpoint search.
3. In a separate fresh process, reload the saved adapter/head against the
   exact source and rerun the same first 32 SELECT inputs, batch size two.
   Require zero category changes, p99 absolute option-probability drift at most
   .005 and maximum at most .02 against the original step-one predictions.
   Compare all 32 IDs and all candidate probabilities using the checked-in
   `scripts.verify_native_select_reload` implementation. Freeze its hash before
   reload, along with the original SELECT prediction hash.

A failed gate leaves 9B optimizer development on HOLD with its failure receipt.
A pass only admits a separately frozen full development arm with new budget
and SELECT-only selection; it does not establish quality, long-input ability or
permission to publish. Formal JevArena and public JevBench remain untouched.
