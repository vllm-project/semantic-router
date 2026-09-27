# Official Qwen3.5-9B Base initialization contrast

**Prospective status:** no Base-source optimizer update has started. The
completed, separately frozen official Posttrained short-TRAIN arm is a read-only
control. Its SELECT BEST458 failed the typed DEV/CSS pilot promotion gate;
this Base cell is not a resume or an alternate Posttrained checkpoint search.

## Hypothesis and frozen contrast

Test whether the official Base versus general Posttrained initialization alone
changes native Decision transfer under the same short-TRAIN curriculum. The
only intended differences from the [Posttrained full-arm protocol](qwen35-9b-shorttrain-full-prereg-2026-09-28.md)
are source weights, official source revision, and source-stage metadata.

| Input | Frozen value |
| --- | --- |
| Base source | `Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404`, Apache-2.0, 9,653,104,368 repository parameters |
| Posttrained control | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, already completed |
| Base config / tokenizer JSON | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` / `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| Base four weight shards | `862bf7bba8a50145d19d0ae463931fae515284024736592a73a336bc4dfa54ee`, `bace8e115e11ca93c22f0352a60d2fb0c76ac6d7d1c2993c143b7ad2b6c8868c`, `63a021ac0011cbfc66166e77103327a8b45dee95832e36551f6b4c3337448959`, `1a643bbed669266917b5058b5d3f660c03233599249ff7d8fd083decfe662ae0` |
| TRAIN | Group-filtered rights-clean v2, 7,324 rows / 5,261 groups / 3,579,176 native tokens, SHA-256 `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c` |
| SELECT / CAL | Original 700 / 700, SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` / `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`; CAL is identity-audited only |
| Trainer | Exact signed code at `3d29a41f6`; seven trainer file hashes must match the Posttrained provenance. Same native segmented-option prompt and shared 256-dimensional dynamic-option head. |
| Optimizer budget | One fresh epoch, exactly 458 updates, microbatch 1, accumulation 16, max length 4,096, no gradient checkpointing, seed 20260926. LoRA rank 16/alpha 32/dropout .05, LR 1e-4; head LR 2e-4; AdamW decay .01, clip 1.0, warmup .05, cosine tail, CE + .5 Brier; BF16 backbone and FP32 trainables. |
| SELECT rule | Evaluate/save at updates 64, 128, 192, 256, 320, 384, 448, 458. Select by family macro accuracy descending, normalized Brier ascending, then earliest step. |

The Base tokenizer files have different hashes from Posttrained, but the
read-only [native-input audit](../scripts/audit_qwen35_9b_tokenizers.py) encoded
all **7,324 TRAIN, 700 SELECT, and 700 CAL rows** with identical ordered token
IDs and equal token totals for both sources. Its report SHA-256 is
`7d609ce72bbfefb11330f420018dc8ea9751b7eebe85a5ecda4cade418c6fe9c`.
This establishes token equivalence on the frozen inputs, not full tokenizer
identity on arbitrary future text. Formal and public inputs must still use the
model's native tokenizer and no truncation.

## Ordered admission and stop rules

1. Recheck the complete Base shard/config/tokenizer hashes, data bytes, code
   hashes, split isolation, source-stage metadata, pinned ROCm image, fresh
   output directory, and exclusive single-GPU assignment. No third-party
   Decision weights or teacher initialize this run. Stop if any identity or
   length gate differs.
2. Run one fresh Base zero-step SELECT700 native response pass. Require 700
   valid outputs and a saved ordered prediction receipt. This is a numerical
   source check, not a model-quality gate.
3. Independently start from the same Base revision and run exactly one real
   LoRA/head update on the first 16 TRAIN rows. Require finite loss/gradient,
   a complete checkpoint, 700 valid SELECT outputs, and a fresh-process first
   32 SELECT native reload with zero category changes, p99 absolute probability
   drift <= .005 and maximum <= .02. The one-update diagnostic and reload
   together have a 0.25 one-GPU-hour cap. On failure stop, retain all receipts,
   and do not widen thresholds or try another source checkpoint.
4. Only after the ordered admission passes, start **one** fresh 458-update Base
   arm from the original official Base source, never from the smoke checkpoint.
   Use the frozen SELECT rule above and a four one-GPU-hour wall cap including
   load, evaluations, and saving. OOM, native fault, nonfinite result, missing
   source, hash drift, save failure or cap breaches leave the arm HOLD.
5. Independently reload BEST on the first 32 SELECT rows with zero category
   changes, p99 drift <= .005 and maximum <= .02. A pass permits exactly one
   gold-free typed DEV1,600 and CSS pilot1,430 run at temperature 1.0 with the
   unchanged native adapter, then sealed predictions before scoring. Compare
   the same-input own Lux1 and pinned JPT-9B controls. Development promotion
   requires `100 * sqrt(T_dev * H_pilot)` to exceed Lux1 by at least 2.0 points
   and invalid/overbudget excess <= one percentage point. Report all typed
   families, three pilot tasks, Brier/ECE and failures. Do not change this rule
   after seeing SELECT or DEV.

No CAL fitting, typed FINAL, CSS15 FINAL, public JevBench, HF upload or release
follows a failed development gate. Even a pass only admits a separately frozen
formal protocol. This 4,096-token filtered TRAIN does not establish long-input
transfer. GPU-hours and source/prediction hashes will be added in a signed
result note, and the unified research gist will retain the outcome including
failures.
