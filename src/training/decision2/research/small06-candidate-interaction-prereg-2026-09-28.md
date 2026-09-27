# 0.6B candidate-interaction head: matched prospective contrast

**State: CPU implementation and synthetic contracts only; no optimizer step on
the 0.6B source has occurred.** The released private 0.6B package, its already
opened JevArena v3 results and the completed shared-head control are unchanged.
This note freezes the comparison before any new model update. It is an internal
experiment plan, not model-card text or a performance claim.

## Why this contrast

The [Decision 1.0 paper](https://vllm-sr.ai/decision-paper.pdf) gives Kai
independent typed paths. The completed official-Qwen3 0.6B arm instead has
one causal endpoint/global-query head. Its [same-panel diagnostic](qwen3-06b-official-full466-v3-postkey-result-2026-09-27.md)
was Choice 109/800, Noul 458/800 and Score 80/400; the original open typed
DEV showed 186/800, 192/400 and 85/400, with every Score point prediction
at level zero. A three-way type-separated head then missed the original
shared-head SELECT gate after the same 466-step schedule. The candidate
[Score replacement D](small06-score-data-d-cpu-hold-2026-09-28.md) is on HOLD
after AI blind review found reasoning shortcuts. None of these results proves
that head sharing causes the deficits.

The next architecture-only hypothesis is that each candidate needs to see
the **other candidates' representations** before its logit is formed. The new
[`CandidateInteractionHead`](../training/model/candidate_interaction_head.py)
inherits the old head and adds a 64-dimensional, shared, leave-self-out
attention over candidate vectors for Choice and Score. Noul uses the old
logit path exactly. Its residual output weight starts at zero. With identical
official-source and seed initialization, the entire new model has the same
zero-step logits as the shared-head control; a tiny Qwen3 CPU roundtrip and
direct inherited-weight comparison test this invariant. The first optimizer
update can change the residual output weight; deeper interaction projections
receive gradients after the residual becomes nonzero. The preflight must
verify finite gradients and native save/reload before a full treatment run.

There is no candidate-position parameter and no hardcoded answer vocabulary.
The head is exactly equivariant if its **input candidate vectors and query are
held fixed** while candidates are permuted. The causal backbone sees preceding
options and its hidden states can change after reordering, so the end-to-end
model is **not guaranteed** order invariant. The independent paired test below
measures that question rather than assuming an architectural property.

The output remains a normalized distribution over the request's live option
keys: Choice accepts 2–255 options, Noul two, Score 2–10 ordered levels.
The [System One API](https://docs.typesafe.ai/api) uses a structured state
and named typed questions; this is a decision scorer, not a chat generator.
The unchanged renderer retains the current 8,192-token complete-input cap,
structured request support and an explicit overlength failure. Score value
is still derived downstream as the probability-weighted level index; no
ordinal-loss improvement is assumed.

## Frozen source, data and treatment

| Item | Fixed rule |
| --- | --- |
| Source | Official `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd4`; source config/model safetensors SHA-256 `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59` / `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| Control | Completed shared-head full466 arm, reuse only; no re-training or checkpoint search |
| TRAIN | Rights-clean v2 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`; exactly 4,094,489 tokenizer-measured input tokens; source/order/groups unchanged |
| SELECT / CAL | 700 each, SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` / `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`; CAL is split-audited only and is never used to select this arm |
| Initialization | Same official backbone and prompt as control; seed `20260926`; baseline candidate head weights must be byte-identical at zero-step; new residual output starts at zero |
| Exposure | One epoch, microbatch 1, accumulation 16, 466 updates, same TRAIN row order and 8,192-token cap |
| Objective | Same categorical CE + `0.5` normalized Brier, AdamW decay `.01`, clip `1.0`, BF16 backbone math / FP32 head and parameters, gradient checkpointing |
| Optimizer | Backbone peak LR `2e-5`, head peak LR `2e-4`, warmup `.05`, cosine tail; no teacher, replay, data replacement or changed task weights |
| Checkpoints | SELECT at updates 64/128/192/256/320/384/448/466; choose highest six-family macro accuracy, then lowest macro normalized Brier, then earliest update |
| GPU budget | At most 2.0 single-GPU hours for a separately approved full arm; preflight caps below are additional and separately recorded |

The local code keeps a separate checkpoint architecture ID and the old shared
head format. A full run is **not** authorized by a CPU pass. Before any real
0.6B optimizer step, freeze the final local commit, source and data file hashes,
exact runtime image and command, exclusive GPU reservation, preflight output
paths, and an answer-free paired-order roster manifest. If a hash or source
precheck differs, stop without adapting the experiment.

## Cost and technical gates

The pinned Qwen3 source's loaded text hidden width is 1,024. At that width and
head dimension 256, the existing FP32 head has **1,053,184** parameters and
the new head has **1,204,864**: **+151,680**, approximately +0.0254% relative
to the 597,103,104-parameter completed model. A treatment package with the
identical backbone would load 597,254,784 parameters. The new leave-self-out
attention's dominant pairwise work is proportional to `K² × 64`, where K is
the live number of candidates; its 255-option affinity matrix holds 65,025
FP32 entries (260,100 bytes per item before temporary buffers). This is a
head-only analytical cost, **not** measured end-to-end latency or memory.
Record peak GPU memory and same-hardware input-length/K latency if the arm
eventually reaches development validation.

1. **CPU gate:** Assert zero-step control logits and inherited weights match;
   padding cannot change valid scores; Choice and Score set-head outputs
   permute with fixed candidate representations; Noul is unchanged; 2/255
   Choice and 2/10 Score limits, arbitrary option keys, finite gradients,
   native full-checkpoint reload and variant-spoof rejection. Validate the
   unchanged TRAIN/SELECT/CAL split and exact token exposure against the
   pinned files. Do not read formal, public or CAL labels.
2. **Real-source zero-step gate:** On one reserved GPU, fresh-load the same
   source and seed twice; compare the old shared-head zero-step path with the
   new inherited path for one Choice, Noul and Score row, then repeat the new
   path. Require exact zero-step logits relative to the shared head, 3/3
   valid outputs, zero changed winners and maximum probability drift `≤1e-4`.
   Cap source-load/zero-step work at 540 seconds.
3. **One-update gate:** Use exactly the completed control's first shuffled
   16-TRAIN-row accumulation window and optimizer. Check finite per-type
   losses/gradients and nonzero Choice/Score residual-output gradient; Noul
   alone must not update interaction parameters. Take one optimizer step,
   save a full checkpoint and reload it in a fresh process, then compare the
   fixed first 32 SELECT IDs (Choice 12, Noul 10, Score 10). Require zero
   changed winners and maximum option-probability drift `≤1e-3`, within
   180 additional GPU seconds. An out-of-budget, numeric, identity or reload
   miss is a recorded HOLD without an automatic retry.

## Development selection and independent order test

Only after those gates pass may one frozen full treatment be launched. Its
SELECT advancement thresholds are **at least 562/700 correct and at least
`.77259` six-family macro accuracy**, the completed control's values. All
eight prescribed milestones and 700 valid answers are required. A failure
stops this arm before typed DEV, CSS pilot, CAL or formal evaluation; there
is no second checkpoint choice.

If SELECT passes, seal predictions on the already open typed DEV 1,600 and
CSS pilot 1,430 **before** scoring them once. An architecture signal requires
typed DEV Choice at least **226/800** (+40 versus control), Score at least
**125/400** (+40), Noul at least **182/400** (within -10), CSS pilot task
median macro-F1 at least **.280959** (within -.01), and full valid coverage.
Score level-use histogram, Brier/ECE and task-level regressions remain
mandatory even if the aggregate increases.

The option-order diagnostic is independent of SELECT-based checkpoint choice:
use the **existing 400 typed DEV Choice groups and their already defined
order-reversed paired questions**, preserving semantic candidate keys and
gold mapping. Freeze the answer-free original/reversed prompt roster hashes
before the optimizer; do not generate new variants after observing treatment
predictions. The completed control's paired jointly correct count is **69/400**.
Require treatment jointly correct at least **109/400** (+40), and report
answer-key consistency and marginal accuracy for both orders. This panel is
already open development material, not a new blind independent test.
Together with the CSS transfer check it can only justify a newly designed,
source-disjoint validation, never a rewritten formal v3 or SOTA claim.

Any future promotion needs a product runtime adapter, full same-panel
re-evaluation, loaded-weight parity and a new independent check because v3
labels are already known. This arm remains private until those gates pass.
