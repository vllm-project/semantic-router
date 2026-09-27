# Gemma 4 ~26B: three-type near-4K TRAIN numeric gate

**Status: frozen CPU preparation, GPU not launched.** This is a bounded
amendment to the [27B development-arm plan](gemma4-qwen27-development-arm-prereg-2026-09-27.md).
The official source-versus-fresh-adapter 32-prompt hidden-state identity
and the short Score TRAIN-row one-step numerical/reload gates passed earlier.
Those checks do not establish whether Choice, Noul and Score backpropagate
through Gemma's text decoder near the proposed 4,096-token limit. This gate
tests precisely that feasibility question; it generates no quality score.

## Immutable inputs and selection

- Official source: `google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, exact configuration,
  tokenizer and two shard hashes as in the earlier pinned source inspection.
  Require a complete state-key/shape match, tied output embedding and actual
  loaded parameter accounting on both independent model loads.
- Dataset: rights-clean v2 TRAIN SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
  No SELECT, CAL, DEV, JevArena, JevBench, Decision Index or other evaluation
  input is accepted. The official Qwen tokenizer SHA-256
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
  is used **only** to form the same-row no-truncation intersection, not to
  load or train Qwen.
- Deterministic selection: in the intersection of rows admitted at 4,096
  tokens by both pinned tokenizers, choose the longest Gemma input separately
  for Choice, Noul and Score. Break an equal-length tie by lexicographically
  greatest TRAIN row ID. The fixed lengths are **4,090 / 4,076 / 4,044**.
  The private lock carries exact row/group IDs, raw-line hashes, Gemma
  prompt/token hashes, task family, language, label index and option count.
  No private row ID or text appears in this public note.

The private lock is owner-held mode 0600 under an owner-held 0700 directory.
It binds source revision/shards, both tokenizers, full TRAIN file, every
relevant source hash, private launcher hash, the three ordered row specs,
single GPU ordinal, objective and optimizer values, numerical thresholds,
wall-time ceiling and no-evaluation condition. The no-GPU `meta` path
recomputes the longest-row selection and independently revalidates each
selected row. A changed source, row, tokenizer, code, launcher or lock is a
hard failure before model weight loading. Actual GPU launch additionally
requires a fresh device-ownership and mirror-hash check.

## Frozen GPU cell and stop rule

Use one freshly available ROCm GPU, locked to physical ordinal 3 and exposed
as `cuda:0` inside one offline container. Load the official BF16 checkpoint,
freeze all source weights, vision, embeddings, MoE router and packed experts.
Train only the exact 60 q/o attention LoRA modules (rank 8, alpha 16,
dropout .05; 120 A/B tensors) and a fresh shared 2,816→256 Decision head,
totalling 6,540,800 trainables. Enable non-reentrant activation checkpointing
and disable cache for long-input backward. Every step uses one distinct
locked TRAIN row, in Choice→Noul→Score order, with native option count and
no input truncation. There are **at most three optimizer updates**.

Use BF16 backbone/autocast and FP32 head/loss, AdamW with LoRA LR `2e-5`,
head LR `1e-4`, weight decay `.01`, seed `20260926`, valid-candidate CE +
`0.5 × Brier`, no replay and no scheduler. Record the unclipped gradient
norm, clip to 1.0, then update. At every row require finite logits with
shape `(1, K)` for its real K options, finite loss strictly between 0 and
`1e6`, all trainable gradients present and finite, norm in `(0, 1e6)`, all
60 LoRA B tensors with nonzero gradient and changed weight, and at least one
changed head tensor. No frozen parameter may enter the optimizer.

After the third step, save only adapter and Decision head to a new private
package. Unload the first model and independently reload the unchanged
official source and saved package. Require exact adapter and head tensor
equality, finite native logits on all three locked inputs and maximum
per-row logit drift ≤`1e-3`. The private receipt records source/lock/code/
package hashes, per-type length, options, loss, gradient/update checks, HBM
peak and reload drift. These are **TRAIN-row mechanics**, not task results.

The hard cap is **45 wall minutes / 0.75 conservative GPU-hour**; the private
launcher applies a 2,680-second container timeout with ten seconds of
termination grace and records exit status, wall time and GPU-hours even on
failure. OOM, unsupported checkpointing/backward, nonfinite value, missing
gradient, changed source, wrong device, failed reload or timeout means
**STOP**. Do not shorten a row, move to another GPU, relax a numerical
threshold, rerun with a different seed or continue to full training under
this lock. Retain the failure receipt and release the device. A pass permits
only a separately reviewed development training arm; it does not authorize
SELECT scoring, formal evaluation or publication.

## Preparation receipts

The launcher is kept private because it contains infrastructure paths. Its
SHA-256, tracked probe SHA-256, private lock SHA-256 and no-GPU meta receipt
SHA-256 are filled from their final frozen files below. Focused CPU tests
cover deterministic same-cohort longest-row selection, three native option
shapes and fixed lock values. The remote CPU container has **no GPU device
mount and no network**. This note and code must be signed and reviewed before
the one GPU cell is considered for execution.

| Artifact | SHA-256 |
| --- | --- |
| Tracked long-input probe | `868a72320a4d14a8c4e762454f538aa538bffaa7fcccebbc8a506218f5c2ab7e` |
| Private offline launcher | `a893feef140075c994ba075781428803f762ba8e1a1fd3b66f761444c2e90836` |
| Private frozen lock | `a163de3415f9487dfde94fabcf071c6adbf7b1a5e03cbf45704cf79db533ccaf` |
| Positive no-GPU meta receipt | `d3a9500fdcac696d191c8964a49b9f00354f12d72ba419b6f6a5672490437b45` |
