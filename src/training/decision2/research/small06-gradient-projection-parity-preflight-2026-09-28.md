# 0.6B gradient projection: one-update parity boundary

**Status: opt-in technical code and CPU checks; no projection treatment or new
model score.** The prior, read-only diagnostic found conflict in six of eight
frozen TRAIN windows. This is a mechanism signal, not evidence that conflict
caused the Choice or Score regressions.

[`gradient_projection_parity.py`](../training/model/gradient_projection_parity.py)
implements the next frozen gate without changing the ordinary trainer. Its
default `plan` phase verifies the pinned official Qwen source, every local
source file, TRAIN and SELECT hashes, all 7,455 TRAIN rows / 4,094,489 native
tokens, the trainer's epoch-zero order, the first 16-example accumulation
window, and 466 planned updates. It emits no prompts or labels. The two
explicit GPU phases independently reload the same source and head seed:

1. `ordinary` reproduces the trainer's usual first-window backward,
   AdamW update, BF16 backbone / FP32 head computation, 466-step warmup
   schedule, and global gradient clipping. It is a separate executable
   reproduction, not a call to an older `train.py --max-steps 1` run.
2. `grouped` performs the identical forward and per-item backward operations,
   accumulates backbone gradients separately for Choice, Noul and Score,
   then **sums without projection**. Head gradients retain the ordinary sum.
   Both arms perform exactly one optimizer update and retain the full
   466-step horizon for the first learning rate.

Before and after the update, each arm predicts the same first 32 SELECT inputs
in the trainer's native two-item BF16 batches. SELECT labels are replaced by
dummies before inference; no SELECT scoring, CAL, DEV, FINAL or public
benchmark is used. The private arm receipts contain only source/data/code
hashes, device time, gradient norm and gold-free probability vectors. The
aggregate comparator passes only with **zero categorical changes** and
maximum absolute option-probability drift at most `1e-5` in both phases, plus
finite gradients and matching input hashes. The output schema rejects raw
text, gold labels and altered row rosters.

This gate costs one free BF16 GPU sequentially for two fresh model loads, 16
training examples and 64 total SELECT forwards per arm. The grouped arm needs
three FP32 backbone-gradient buffers in addition to the model and optimizer;
confirm HBM before launch. Run CPU contract tests first and use a bounded
external watchdog on the GPU. A failed gate records HOLD and ends the arm;
neither the threshold nor the schedule is revised. A passed gate only
authorizes review of a separately frozen, full-budget projection treatment.
