# Gemma 4 ~26B text Decision: prospective development arm

**State: CPU admission passed; no new GPU training authorized by this note.**
The separate official-source zero-step identity and one TRAIN-row optimizer/
reload gates passed. Neither measured model quality. This design asks whether a
small text-only Gemma 4 q/o adapter can become a useful ~27B Decision 2.0
candidate under the existing rights-clean v2 data and native three-type head.
It is a development screen against the completed official Qwen 27B run, not a
strict same-initialization or equal-trainable-capacity causal ablation.

## Pinned sources and completed controls

| Property | Prospective Gemma arm | Completed Qwen reference |
| --- | --- | --- |
| Official source | `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Loaded text parameters | 25,233,141,760; separate source inspection loaded 25,805,933,872 total including vision | 25,624,600,064 |
| Trainable adapter/head | q/o-only rank 8, alpha 16, dropout .05: 3,645,440 + 2,895,360 = **6,540,800** | 496 LoRA targets plus head: **63,627,776** |
| Initialization | Official general instruction source and fresh Decision head | Official general posttrained source and fresh Decision head |
| Native input | Shared segmented options and final query, with official Gemma BOS | Shared segmented options and final query |
| Qwen reference contract | Proposed to mirror CE + 0.5 Brier, LoRA LR `2e-5`, head LR `1e-4`, weight decay `.01`, warmup `.05`, microbatch 1, accumulation 16, one epoch, seed `20260926`, max input 4,096 | Actual private checkpoint contract confirms those settings; no replay |

The prior Qwen source and checkpoint are already complete; **do not retrain or
relabel them as a matched control**. Their selected development checkpoint is
step 368 of 458, SELECT 566/700, family macro accuracy .79398. Its separate
typed DEV, human CSS pilot and JevBench public-subset results are in the
[Qwen research result](qwen38-27b-clean-v2-full4096-2026-09-27.md), not formal
Gemma evidence. Its checkpoint Decision configuration SHA-256 is
`ea89252f0d5d1b084c69368ed7028c506198bf0a43712a46f3ba8876856e2b0b`.
The sources are different pretrained distributions and model architectures;
the Gemma target trains about one tenth as many adapter/head parameters.
Any observed difference can be due to any of these factors, tokenization,
data admission or optimizer behavior. The upstream `A4B` active-parameter
figure has not been independently measured and must not replace loaded
parameter counts in any Pareto comparison.

## CPU-only full TRAIN admission and three-type mechanics

The exact rights-clean v2 TRAIN file SHA-256 is
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
It contains 7,455 rows: Choice 3,908, Noul 3,031, Score 516; English 6,085,
Chinese 1,370. The pinned Qwen and Gemma tokenizer JSON SHA-256s are
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
and `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The CPU-only, no-network image audit used the production segmented prompt,
the exact Gemma BOS shift, strict schema validation and **no truncation** at
4,096 tokens. Audit code SHA-256 is
`718a5c7d058b8696c163735e89522bb48e2a52d7c5678b6ae58a80b158158c8a`;
private aggregate-only receipt SHA-256 is
`0ac2d161fad3430fbc93a413a35a65cf894d48f82e50c691b843207e103449eb`.

| Admitted cohort | Rows | Choice / Noul / Score | Unpadded source tokens | Updates at batch 16 |
| --- | ---: | ---: | ---: | ---: |
| Existing Qwen admission | 7,324 | 3,824 / 2,993 / 507 | 3,579,176 | 458 |
| Gemma admission | 7,287 | 3,804 / 2,982 / 501 | 3,620,578 | 456 |
| Same **7,287** row intersection under Qwen tokenizer | 7,287 | 3,804 / 2,982 / 501 | 3,432,920 | 456 |

The common ordered row-ID roster SHA-256 is
`3d6f96168e38761b45630e9a8a61c60c85b19811df3049e715696acae0edd30b`.
It contains 5,998 English and 1,289 Chinese rows. Within this *same* roster,
Gemma emits 5.47% more non-padding tokens than Qwen. Conversely, its total
training tokens are 1.16% above the **different-row** completed Qwen cohort.
No exact same-row, same-token, same-step control exists. The prospective arm
can match the source TRAIN dataset, row intersection and optimizer settings,
but it cannot honestly claim exact token equality across tokenizers or exact
matching to the already finished Qwen arm.

The Gemma common-roster median/p95/max are 167/2,377/4,090 tokens. Type-wise
longest rows are Choice 4,090, Noul 4,076 and Score 4,044, so a short-row
optimizer pass alone cannot prove full-cohort memory safety. The CPU-only
production head/collator/CE and CE+0.5 Brier loss smoke on one actual TRAIN
row of each type passed finite forward and backward at 166/104/369 tokens,
respectively.
It proves three-type schema and head gradients, **not** Gemma backbone LoRA
backward at 4K tokens. SELECT, CAL and formal labels were not read in this
audit; the private receipt contains counts/hashes and no text or row IDs.

## Proposed gated run, pending code and separate GPU review

1. **Long-input numeric gate first.** Freeze the exact longest admitted TRAIN
   row from each type under the audited common roster in a private lock. Use
   one freshly verified official source, q/o adapter and shared head. Run at
   most three sequential optimizer updates (one each), retaining finite
   native logits/loss/gradients, optimizer-only adapter/head changes, peak
   HBM, save/reload parity and device ownership. No SELECT, CAL or benchmark
   access. Cap at **45 wall minutes / 0.75 GPU-hour** on one GPU. OOM,
   nonfinite values, unverified hash, or timeout means stop without reducing
   length, dropping a type, relaxing tolerance or searching another device.
   This is a proposed gate, **not a launch approval**; the three exact private
   row hashes, runner/code hashes and receipt thresholds must be locked first.
2. **Train runner admission.** Implement a Gemma-only text path that reuses the
   shared `CandidateHead`, valid-K CE/Brier loss, exact length-bucket batching,
   sample-weighted final accumulation and checkpoint contract. Do not alter
   the Qwen path. Freeze the code/image, model shards, tokenizer, common row
   IDs and seeded order. A CPU dry run must reproduce 7,287 admitted examples,
   three task types, 3,620,578 Gemma unpadded tokens and **456 planned
   updates** with microbatch 1/accumulation 16, no replay, no hidden
   truncation or extra rows. Before the optimizer, require official-versus-
   fresh-adapter source identity on a fixed unlabeled roster and strict
   zero-step native output, as in the completed prior gate.
3. **Staged one-epoch development arm.** If the long-input and runner gates
   pass, execute the exact 456-update common-roster arm with the Qwen
   reference's CE + 0.5 Brier objective and optimizer values shown above.
   Reserve at most **24 GPU-hours** total for this one arm, with hard saved
   boundaries at updates 16, 64, 128, 256 and 456. At update 16, inspect
   numerical stability, peak HBM and observed throughput only. If projected
   completion from measured token throughput exceeds the remaining 24-hour
   cap, stop and retain the partial result; do not extend the budget.
   SELECT700 is read only at the predeclared 64/128/256/456 checkpoints.
   CAL, typed DEV, CSS pilot, JevBench and v3 formal labels do not select
   checkpoints. Any source drift, invalid run contract, changed data/step
   budget, nonfinite loss/gradients, OOM or timeout stops the arm.
4. **Development choice and next decision.** Among available predeclared
   checkpoints, choose highest SELECT family-macro accuracy, then lower
   normalized Brier, then earliest step, as in the completed Qwen arm.
   At update 256, stop for futility if the best SELECT family macro remains
   below `.70`; this cutoff is prospective, not a claim about a measured
   Gemma score. A completed candidate advances to a **separate** calibration
   and one-time independent development diagnosis only if its selected
   SELECT family macro is at least `.794`, matching the rounded Qwen
   reference. Record all Choice/Noul/Score subresults and any regression;
   this is a compute-allocation rule, not a release gate. Failure retains
   the candidate as research evidence. A positive development result still
   needs package parity, same-panel baselines and the separately governed
   JevArena v3/JevBench publication protocol before any public score or
   `DEV2.0-27B` release.

The 24 GPU-hour ceiling is a **maximum reservation**, not a throughput
prediction. The 94-second one-row update receipt included two full model
loads and package reload and cannot be extrapolated to 4K rows. The
long-input gate and first 16 updates are specifically designed to measure
runtime and memory before consuming the full cap. GPU assignment, exact
private lock, image digest and launch command require a fresh occupancy and
hash audit and separate approval. This document authorizes **zero** further
GPU steps by itself.
