# 0.6B typed-head recovery: prospective, development-only protocol

**State: CPU head preflight complete; GPU training is not authorized by this
document.** This note fixes one architecture contrast before any treatment
optimizer update. It does not reinterpret the completed 0.6B package or claim
an independent formal result. All future model publication remains private.

## Evidence and hypothesis

The [Decision 1.0 paper](https://vllm-sr.ai/decision-paper.pdf) describes
Kai-0.6B as a bidirectional encoder with separate Choice, Noul and Score
paths and a 1,024-token complete-input cap. Its causal family uses one shared
candidate endpoint/global-query head. The completed 2.0 0.6B control instead
uses official Qwen3-0.6B-Base, an 8,192-token cap and a newly initialized
**shared** endpoint/global-query head. Thus Kai versus the current 2.0
checkpoint changes backbone, tokenizer, data exposure, context and readout
together; it cannot isolate an architectural cause.

The completed Qwen control scored 562/700 on SELECT, but only 186/800 Choice,
192/400 Noul and 85/400 Score on typed DEV; all 400 DEV Score predictions
selected level zero. CSS pilot task-median macro-F1 was .290959. The later
post-key v3 panel yielded Choice 109/800, Noul 458/800 and Score 80/400; the
composite 38.5200 was above Kai1's 35.9383 mostly through human transfer.
The paired v3 interval includes zero and that formal result is **diagnostic
only** for this arm, never a checkpoint selector. The 7,455-row rights-clean
v2 control contains only 516 Score rows across different level counts. Head
interference, insufficient Score coverage and causal candidate-order effects
are separate **hypotheses**, not established causes. Prior paired-order
training on a different Qwen reranker failed its SELECT gate, so this arm
does not repeat that recipe.

## Fixed architecture contrast H

H replaces the one shared `CandidateHead` by three independent copies of the
*same* FP32 candidate/query bilinear-plus-MLP head, routed by the typed
question. It preserves the exact candidate renderer, backbone, prompt,
candidate mask, output softmax and calibration contract. It scores the
runtime-supplied candidates and ordered Score levels directly, without a
vocabulary-generation/chat path. The new module is
[`type_separated_head.py`](../training/model/type_separated_head.py); its
CPU tests check mixed-type routing, 2–255 candidates, padding, finite
gradients, independent parameters and option-permutation behavior **given
fixed candidate representations**. Actual causal endpoint vectors change
when options move, so this head alone does not guarantee order invariance.

| Frozen item | H treatment and completed control |
| --- | --- |
| Initialization | `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd4`; random heads seeded 20260926; **not** the completed 2.0 weights |
| TRAIN | Same rights-clean v2 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`; 4,094,489 encoded tokens under the unchanged Qwen tokenizer |
| SELECT | Same 700 rows, SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` |
| CAL | Same disjoint 700 rows, SHA-256 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`; never used for training or checkpoint selection |
| Exposure | One epoch, microbatch 1, accumulation 16, exactly 466 planned updates; same row order/seed and 8,192-token complete-input cap as completed control |
| Optimizer | Full backbone plus head, CE + 0.5 Brier, AdamW weight decay .01, clip 1.0, BF16 backbone compute / FP32 head and parameters, gradient checkpointing; backbone/head peak LR `2e-5`/`2e-4`, warmup .05 and cosine tail |
| Selection | Only SELECT family-macro accuracy, then lower normalized Brier, then earliest update; evaluate steps 64,128,192,256,320,384,448,466; do not choose on typed DEV, CSS pilot, public231 or formal v3 |
| Budget | One isolated GPU, at most 2.0 GPU-hours including checkpointing and SELECT; no rerun or threshold change after seeing outcomes |

The completed 466-step shared-head control is reused; it is **not retrained**.
This is a matched-data, matched-token and matched-backbone comparison of head
sharing. The treatment has additional parameters and perhaps slightly higher
memory, so report exact count, peak memory and time rather than calling the
comparison parameter matched. No teacher or appended training data is allowed.

## Technical gate, development screen and stop

Before reserving a GPU, finish explicit trainer/checkpoint/native-adapter
integration in local code and test both old shared-head and new typed-head
reload without altering the old package. Confirm all three task types and
dynamic candidate counts; verify source revision, every file hash, 7,455 row
identities, all token lengths and TRAIN/SELECT/CAL group isolation. Freeze
the implementation commit, training command, source-package hash and exact
stop receipt. A one-update numeric/reload smoke is a **separate**, bounded
GPU gate requiring root review. It must show finite per-type loss/gradients,
no backbone source drift at zero steps and saved-vs-in-memory native
probability maximum drift ≤1e-3. A failed gate stops H without an altered
retry.

If technical gates pass, the fixed full arm is eligible. After selecting
one checkpoint, seal gold-free predictions, then inspect SELECT700, typed
DEV1600 and CSS pilot1430 once with unchanged native scorers. The forward
development criterion is simultaneous: typed DEV Choice at least **226/800**
(+40 over the control), Score at least **125/400** (+40), Noul at least
**182/400** (no worse than -10), CSS pilot task-median macro-F1 at least
**.280959** (no worse than -.01), all 1,600 typed outputs valid, and no
new CSS overflow. Also publish the SELECT and paired option-order/counter-
factual views, per-level Score confusion, Brier/ECE and actual parameter
count. These thresholds are a *development screen* chosen now, not a promise
of improvement or a formal-release gate. Failure stops H; no alternative
checkpoint, extra steps or changed threshold. Passing would justify an
independent source-disjoint validation design before any new post-key v3
prediction; v3 labels already known at project level cannot be presented as
untouched blind evidence.

## Separate data contrast D: not GPU-ready

D would keep the **existing shared head** and official Qwen source, replacing
384 of the 516 existing Score TRAIN rows with 128 independent, reviewed
three-level groups (three levels per group, multiple mechanisms), while
keeping 7,455 total rows, the same update schedule and a tokenizer-measured
training-token total within **0.5%** of the 4,094,489-token control. Choice
and Noul TRAIN IDs stay fixed. The existing 12-group quality packet is only
an audit pilot; it is far below 128 approved groups and cannot initialize D.
Freeze every candidate group's oracle, source, language, rendered input,
licensing, near-overlap checks and exact SHA-256 **before** requesting GPU.
If quality or token-matching fails, D remains HOLD. D isolates Score-data
coverage better than mixing new head, replay and data in one run. Its
development gate should be locked only after that immutable data roster is
complete; do not infer a v3 or release gain from the present outline.

## Implementation handoff: CPU audit before a separately approved GPU gate

The typed readout is now explicitly wired through `DecisionModel`, collated
task IDs, full-checkpoint save/reload and native benchmark inference. It has a
distinct architecture ID and rejects a missing type ID or a contradictory
checkpoint declaration. Existing shared-head checkpoints continue to load
through the old path. The trainer's typed mode rejects changes to the official
source revision, TRAIN/SELECT/CAL hashes and row counts, full 466-step schedule,
loss, seed, optimizer settings and tokenizer-measured TRAIN token total. The
native adapter hashes `type_separated_head.py` only for typed checkpoints. No
old published package or frozen benchmark receipt is rewritten.

The immutable technical lock is
[`small06-type-separated-head-gate-lock-2026-09-27.json`](small06-type-separated-head-gate-lock-2026-09-27.json).
It pins source/data/code hashes and two separate one-GPU caps: 540 seconds
for the zero-step repeat and 180 seconds for the one-update/reload stage. Its
CPU stage must finish and its output SHA-256 be reviewed before an authorized
GPU slot is used. The following placeholders refer to private, access-controlled
paths and must not be committed or copied into a model repository:

```bash
cd src/training/decision2
python -m scripts.preflight_qwen06_type_head audit \
  --lock research/small06-type-separated-head-gate-lock-2026-09-27.json \
  --source "$PINNED_OFFICIAL_SOURCE" --train "$TRAIN_V2" \
  --select "$SELECT_V2" --cal "$CAL_V2" \
  --output "$NEW_PRIVATE_CPU_AUDIT"

# After independent receipt review and a root-approved exclusive GPU slot:
timeout 720s python -m scripts.preflight_qwen06_type_head run \
  --lock research/small06-type-separated-head-gate-lock-2026-09-27.json \
  --source "$PINNED_OFFICIAL_SOURCE" --train "$TRAIN_V2" \
  --select "$SELECT_V2" --cal "$CAL_V2" \
  --audit "$NEW_PRIVATE_CPU_AUDIT" \
  --audit-sha256 "$REVIEWED_AUDIT_SHA256" \
  --output "$NEW_PRIVATE_GPU_GATE_DIR"
```

The GPU stage compares two fresh zero-step source starts on all three types,
checks finite per-type losses/readout gradients, takes exactly one first-window
optimizer update, then compares 32 mixed-type SELECT outputs before and after
a full native checkpoint reload under identical adjacent-pair batching.
Admission requires no category changes and maximum option-probability drift
at most `1e-3` within both original stage caps. The GPU stage writes a
separate phase receipt before any optimizer update and another after the
one-update reload comparison; its final receipt binds both phase hashes.
The gate is technical only;
no typed DEV, CSS pilot, JevBench or formal v3 labels are read. A timeout,
missing receipt, nonfinite result or parity miss is a recorded HOLD without an
automatic retry.

Only after a reviewed gate PASS may a **separate** full-run authorization use
the unchanged 466-step treatment command:

```bash
cd src/training/decision2
python -m training.model.train \
  --model-path "$PINNED_OFFICIAL_SOURCE" --init-kind base \
  --base-revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --train "$TRAIN_V2" --select "$SELECT_V2" --cal "$CAL_V2" \
  --output "$NEW_PRIVATE_FULL_RUN" --head-variant type-separated \
  --train-mode full --objective ce_brier --brier-weight 0.5 \
  --epochs 1 --max-steps 466 --microbatch 1 --accumulation 16 \
  --eval-batch 2 --max-length 8192 --head-dim 256 \
  --backbone-lr 2e-5 --head-lr 2e-4 --weight-decay 0.01 \
  --warmup-ratio 0.05 --save-every 64 --seed 20260926 \
  --gradient-checkpointing
```

This full run remains **not started** at this handoff. A selected typed
checkpoint would require a separate native release packager and exact
download/parity verification; the current 0.6B package remains unchanged.
