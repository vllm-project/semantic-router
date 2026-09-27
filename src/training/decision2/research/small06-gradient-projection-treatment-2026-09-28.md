# 0.6B projected-gradient treatment: implementation and stop gates

**Prospective status:** the read-only gradient diagnostic found six of eight
frozen mixed-type windows with conflict. A projection-disabled independent
one-update comparison subsequently passed with no changed category or option
probability on 32 fixed SELECT inputs. Neither result establishes a model
gain. This note pins the single treatment before its first optimizer update.

The official-source, hard-label shared-head control is already complete; do
not retrain it. The only treatment change is the shared-backbone gradient
direction. Use the identical official Qwen3-0.6B-Base revision, rights-clean
v2 TRAIN/SELECT/CAL partitions, tokenizer, seed, full 7,455 TRAIN rows /
4,094,489 native tokens, epoch-zero order, CE + 0.5 Brier loss, 8,192-token
no-truncation input, FP32 parameters with BF16 backbone autocast, AdamW
groups, clipping, one epoch / 466 planned steps and checkpoints at
64/128/192/256/320/384/448/466. The head retains ordinary gradients.

The isolated `train_gradient_projection.py` entrypoint requires its opt-in
`--gradient-projection` flag and accepts only this contract. The archived
ordinary `train.py` remains byte-for-byte unchanged, preserving exact-resume
source hashes and old checkpoint schemas.
[`gradient_projection.py`](../training/model/gradient_projection.py) collects
per-type FP32 backbone gradients in each unchanged accumulation window. For
each nonempty task in Choice, Noul, Score order, it projects a negative
component against each **original** other-task gradient in that same fixed
order. The projected task vectors are summed and their combined backbone norm
is matched to the ordinary backbone sum before the usual global clip. If
either norm is zero, the ordinary backbone sum is used. Missing types are
skipped. The optimizer, head gradient, data and all other loss terms are
unchanged. Every step records aggregate task norms, pair cosines, projection
count and final backbone norm without input text.

Before the full arm, an independent opt-in one-update diagnostic must pass:
fixed official source/TRAIN/SELECT/CAL hashes, zero-step native parity to the
archived ordinary arm, finite projected gradients, one actual optimizer step,
successful checkpoint save/reload, and no categorical changes or more than
`1e-5` maximum option-probability drift on the fixed 32 SELECT prompts.
The first trainer window has 11 Choice, 5 Noul and 0 Score rows under the
frozen order; CPU algorithm tests explicitly cover all three tasks and an
opposing-gradient case. This one-step preflight therefore cannot establish
Score behavior. The full-arm log must show later Score-containing windows
and finite projection statistics without choosing a different first window.
The full run remains bounded by an external two
GPU-hour watchdog. Failure at any preflight or numeric gate is a HOLD with no
threshold adjustment or replacement checkpoint search.

After exactly one complete treatment, choose the checkpoint by the control's
fixed SELECT family-macro accuracy, Brier and earliest-step ordering. The
prospective SELECT advancement criteria remain at least 569/700 correct,
family macro at least .78259, quantized Score at least 37/90, and human
GoEmotions Choice at least 156/200. Only if all pass should the already opened
typed DEV and CSS pilot be inspected once, with the previously pinned
Choice/Score/transfer gates. Those are development checks, not fresh formal
evidence. No JevArena v3, JevBench public or other revealed formal key is
used to choose a checkpoint. A new source-disjoint confirmation is needed
before any major 0.6B product claim.
