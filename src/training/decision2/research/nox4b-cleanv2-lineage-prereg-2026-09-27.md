# Decision 2.0 4B: own-lineage clean-v2 screen

**Status: prospective development arm; no optimizer step or new model score.**
This note fixes one training experiment before a new checkpoint is examined.
It does not qualify the existing third-party-initialized 4B research package
for Decision 2.0 publication. No JevArena FINAL or CSS15 label is an input to
this experiment. Those labels have already been opened for a different 4B
candidate, so a subsequent model's score on that same panel will be a
same-panel comparison, not a fresh blind release result.

## Why this arm

The exact published `llm-semantic-router/Decision-1.0-Nox-4B` revision is
`0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` with 4,208,383,488 loaded
parameters. It is an eligible initialization under the owner's lineage rule.
The local source package's `release-manifest.json` SHA-256 is
`50c2f77c7c3f6c1efae7014ccfc4aedc3b185e52731ba721a1c6d6192668d945`;
all 97 listed files passed an independent size and SHA-256 readback before
this preregistration. Recheck that exact manifest and every listed file in
the runtime container before loading the model.
Its native dynamic candidate head has already passed a two-process, 32-prompt
zero-drift smoke. A completed Nox human-only 5,824-row continuation improved
CSS pilot task-median macro-F1 from .4114 to .4484 but moved typed DEV from
66.75% to 66.625%. A completed 8,522-row structured replay continuation gave
typed DEV 66.4375% and CSS pilot F1 .4431. These are development results, not
release claims, and their different data and selection sets prevent causal
comparison.

The single new question is whether the already frozen, source-audited
rights-clean v2 mix can yield a **combined typed and human-pilot gain** from
our own 4B source under the established dynamic-head LoRA objective. This is
faster and less confounded by a new architecture than a simultaneous Qwen Base
and recipe search. The official `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b` is eligible for a later
separate arm; no completed official-Qwen 4B training result was found in the
current research ledger. Third-party Decision models may be controls or
teachers, but no third-party weight initializes this arm.

## Frozen initialization, data and budget

| Input | Frozen identity |
| --- | --- |
| Direct initialization | `llm-semantic-router/Decision-1.0-Nox-4B@0bb833504965c0eabdb9630b7bbd385cb2fe5cd4`; require a complete local package inventory and zero-step native model/input/output identity before training |
| TRAIN | rights-clean v2, 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| SELECT | TRAIN-disjoint 700 rows, SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` |
| CAL | independent 700 rows, SHA-256 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a` |
| Data manifest | SHA-256 `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`; private source-use and redistribution audit remains authoritative |

Do not add Score curriculum v1-v6 or translated rows to this arm: their
training admission remains conditional or blocked. Do not use teacher
probabilities or extra replay. This isolates the fixed clean-v2 mixture from
the previously completed Nox human and generic structured mixtures, while
acknowledging that the comparison is not a strict same-data causal ablation.

Use the existing `training.model.train` native Qwen3.5 dynamic-head path:
`init-kind=decision1`, rank-16 LoRA, alpha 32, dropout .05, LoRA LR `1.5e-5`,
head LR `7.5e-6`, objective CE plus `0.5` Brier, one epoch, microbatch 1,
accumulation 16, effective batch 16, seed `20260926`, maximum input length
8,192 tokens. Run **all 7,455 rows exactly once** for 466 optimizer updates;
save and evaluate SELECT at steps 0, 128, 256, 384 and 466. An exact read-only
CPU admission pass using the byte-identical local trainer/tokenizer code and
pinned Nox tokenizer measured **4,194,465 unpadded TRAIN input tokens**,
maximum row 6,596, with 7,455/7,455 admitted; SELECT is 97,711 tokens,
maximum 222, 700/700 admitted, and CAL is 97,139 tokens, maximum 222,
700/700 admitted. The data presentation budget is therefore 7,455 examples
and 4,194,465 unpadded input tokens. Recompute and seal these totals from the
exact runtime image and source files before optimization; abort on any change
or any row over 8,192 tokens. Do not
truncate, silently quarantine, substitute a dataset, lengthen the context,
or change the budget under this experiment name. A distinct new protocol is
required after an admission failure. The read-only preflight used trainer
`train.py` SHA-256 `b4414a5a3b4b2fbd3ec2ef3ad68e04480b6f695f8a05d239189dc033c1eb628d`,
renderer `decision_model.py` SHA-256
`fd9b76cacb1e3598d560122c8b86152b816a0235bd8303496af2effe57ebcae2`
and loader `data.py` SHA-256
`632bd60555f63459ff08ffe82c8263ca6a96f75360bc8c06469fcb10f1e2e99c`.
Freeze the runtime image digest, model file hashes, encoded row roster and all
trainer source hashes in the private run manifest before step one.

The completed Nox structured run used 533 updates on one GPU. A preliminary
resource envelope for this 466-update arm is **1–3 GPU-hours on one GPU**,
including periodic SELECT; this is an estimate, not measured utilization.
A short preflight must prove the exact source load, zero-step SELECT native
answer/probability parity, finite forward/backward/gradient and atomic
checkpoint/resume. Abort before the full run if the first 32 updates project
more than four GPU-hours, or if the numerical, memory or input gates fail.
Record actual accelerator allocation, occupancy and GPU-hours afterward.

## Selection and decision rules

The only weight selector is SELECT700: maximize four-family macro accuracy,
break ties by lower normalized Brier, then earliest step. Do not inspect
DEV, public JevBench or exposed FINAL to choose a checkpoint. Once BEST is
fixed, fit three positive temperatures on CAL700 only and record both
pre- and post-calibration Brier/ECE. The completed 1.0 Nox, human5824 and
structured8522 runs are read-only comparators; they are not restarted.

After the selected and calibrated checkpoint is immutable, run exactly one
native development pass on typed DEV1,600 and CSS pilot1,430, plus the public
JevBench231 diagnostic. Report all four typed families, Choice/Noul/Score,
each CSS task, language coverage, invalid/overflow counts, calibration,
and pairwise uncertainty against the pinned own-source baseline. The
development promotion rule is `100 * sqrt(T_dev * H_pilot) >= 56.0`, where
`T_dev` is four-family macro accuracy and `H_pilot` is the median of the three
CSS pilot task macro-F1 values. Also require complete typed coverage, at most
the source's CSS pilot invalid count, and no more than ten percentage points
of Score accuracy loss against source Nox on the identical typed DEV. A
positive public231 result cannot rescue a failure on the typed/CSS gate.
These gates determine whether to prepare a release-validation candidate, not
whether it is already publishable. Disclose any task or type regression.

JevArena v3 typed FINAL1,600 plus CSS15 6,547 and separate public231 can be
run only after a new candidate and protocol freeze. Since the v3 labels have
already been accessed for prior 4B work, require a new untouched validation
panel or independent external confirmation before calling a later 4B gain
independently verified. Do not transfer any prior third-party-initialized 4B
score, package hash or calibration to this weight. Publication additionally
requires lineage attestation, exact package/serving parity, HF private
download readback and an accurate product model card.

## Preflight record

The first one-step container attempt stopped in the device gate before model
loading or optimization: exposing only one DRM render device did not
make a BF16 CUDA/ROCm device visible to the trainer. Its output directory is
retained as a failed attempt. A separate read-only probe with the full
`/dev/dri` mapping and `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, and
`CUDA_VISIBLE_DEVICES=0` exposed exactly one ROCm GPU with BF16 support.
Retry the exact frozen training settings in a distinct output directory with
`--device=/dev/kfd --device=/dev/dri`; retain the visibility variables, verify
one visible GPU and record the physical GPU allocation before optimization.
This device mapping correction does not change the data, model, selector,
objective, or budget. The first attempt produced no optimizer step or score.
