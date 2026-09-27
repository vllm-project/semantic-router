# Gemma 4 text LoRA admission and proposed 27B development ablation

This is a **prospective research preregistration**, not a Decision 2.0
score, model selection, or training authorization. The source is the official
general checkpoint [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. The pinned
source, two weight-shard hashes, tied output embedding, complete loaded key
set and shapes were checked in the prior [native source preflight](gemma4-native-preflight-2026-09-27.md).
Actual loaded parameters are 25,233,141,760 text and 25,805,933,872 total;
the upstream 3.8B active figure has not been independently measured. The
source's advertised 256K context is not an admitted Decision input length:
the current tested cap is 8,192 tokens.

## Text-only target plan

The pinned text decoder has 30 layers and 128 packed experts. Its 25 regular
layers have `q_proj: 2816→4096`, `o_proj: 4096→2816`; layers
5, 11, 17, 23 and 29 use 8,192-wide q/o projections. The packed expert
matrices and router are distinct from those linear attention projections.
Attach rank-8, alpha-16, dropout-0.05 LoRA to the exact 60 full-path q/o
modules **only**. Freeze the official text source, MoE router, experts,
embeddings, and vision tower. This is a deliberately small first comparator,
not a claim that q/o is optimal. A source-model `Gemma4TextModel` with the
shared dynamic-option `CandidateHead` at hidden width 2,816 and head width
256 would have 3,645,440 LoRA parameters and 2,895,360 head parameters,
6,540,800 prospective trainables. A no-GPU pinned-config/meta-model PEFT
attachment verified all 60 targets and 120 A/B trainable matrices. It did
not load training data or run an optimizer.

The Gemma encoder keeps the shared segmented option/query prompt but inserts
its official BOS token and shifts candidate/query positions exactly once.
The Qwen code path is unchanged. A fixed set of 32 synthetic **unlabeled**
schema prompts covers 11 Choice, 11 Noul and 10 Score inputs, including a
few Chinese and Spanish strings. Its token roster SHA-256 is
`86cfaff42cf37c08ece8dac11eb3c0b098ca33f19e44d82b52478d4e7b45f5eb`.
These prompts are input and numerical-identity checks, not a multilingual
evaluation or decision benchmark.

## First GPU cell: source versus fresh adapter identity, then stop

The next cell is **not yet authorized by this note**. Its private lock binds
the exact official revision, configuration/tokenizer and both shard hashes,
code and runner hashes, 32-prompt roster, GPU ordinal, zero updates, and a
20-minute / **0.3334 conservative GPU-hour** ceiling. Run two independent
BF16 loads in order on one freshly available GPU: first the official text
source, then the same source with freshly inserted, untrained LoRA. Both use
identical tokenized inputs, SDPA, no cache and no dropout. Store only the
selected hidden vectors at option endpoints and final query positions in
private receipts. Do **not** attach or score the random decision head, read
formal labels, run an optimizer, or call these outputs model quality.

The cell fails closed if the GPU or source revision differs, a weight or
loaded key/shape is absent, the output embedding is untied, the 60 target
paths/counts differ, any fresh LoRA B matrix is nonzero, vision becomes
trainable, input truncates, a vector is nonfinite, any one of the 32 readings
is missing, or source-to-adapter maximum absolute hidden drift exceeds
`1e-4`. A failed GPU enumeration, OOM, runtime error or timeout also stops
the path without searching a different threshold or checkpoint. Before any
launch, recheck current GPU occupancy and mirror/file hashes; after either
success or failure, retain exit code, timestamp, elapsed GPU-hours, receipt
and log hashes, then release the GPU. A successful identity cell admits only
the adapter wiring for a separately reviewed optimizer experiment.

The local no-GPU/meta receipt has SHA-256
`eeb337907eb291f5154a94a47d97e0606c853443d77d82f90ca15e1deb076109`.
The private launch runner is bound by SHA-256
`803ef00861e5a95bcb7d23eb22dc08b1d17026a169f01a2c9164913bef3b6e04`.
The private lock SHA-256 is
`4b649c6ca8f9ee315a2e2d84259f6e18f79084c2b4170acc5b461e4913602290`.
It binds these code SHA-256s: adapter probe
`4296ba568d03889ab05699bc3845818ffe8e68dffa8db5edfc5c6b75da5ddb54`,
Gemma adapter
`57d12e76dfded2cfc328a63d5767a24c18af3fb61efc3274407062f4bf716aa0`,
and the existing source probe
`af25a8ccc6fd9d8b27bfa65930cb6302d537eeba83b8b650730fdc37e332454c`;
the three shared prompt/normalization code hashes are also in the private
lock. A no-device, no-network image smoke verified the private lock against
all 32 official-tokenized prompts and reported zero optimizer updates.
Focused unit tests: **10 passed**; Ruff and Black passed. This repository
contains neither weight shards nor private hidden-vector receipts.

## Subsequent gates, not authorized by the identity cell

If numerical identity passes, the **next separate** one-step gate must
pre-register one exact training row, initialization seeds, optimizer and
learning rate, maximum tokens, BF16/FP32 policy, and acceptance tolerances.
It will verify finite loss, gradients and weight updates limited to the
enumerated 120 LoRA tensors plus decision head; vision/router/experts and
source parameters must remain fixed. Save, reload and compare native logits
under an exact numerical tolerance, including candidate order and all three
task shapes. No dev or formal score follows automatically from that gate.

Only then consider a matched development ablation against the **completed
official-source Qwen3.8-27B clean-v2 arm**, not a third-party Decision base.
Its immutable source is `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The existing arm used rights-clean v2 TRAIN SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
7,324 admitted no-truncation rows at 4,096 tokens, one epoch and 458 optimizer
steps; SELECT SHA-256
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
selected its checkpoint. The Gemma candidate would use the same training
source and option-head objective, rank/alpha/dropout, no formal labels and a
4,096-token ablation cap. **Before launch**, tokenise both exact models on
the same row IDs, compute the no-truncation intersection and exact admitted
token budgets, then freeze the row roster, order, optimizer-step budget,
control identity and selection rule. If the existing Qwen control does not
cover the same admitted rows or token budget, report that it is an unmatched
reference and obtain approval for a fresh matched control if warranted; do
not silently relabel the old result as matched or repeat it by default.

Use TRAIN only for fitting, the already separate SELECT for checkpoint
selection by family-macro accuracy, Brier score and earliest-step tie break,
and CAL only after selecting a model for calibration. Compare frozen selected
packages on the same typed DEV and three-task CSS pilot as **development**
evidence, reporting Choice/Noul/Score, task-wise transfer, valid coverage and
calibration. Stop for nonfinite values, source-identity failure, hidden
truncation, a changed data/compute budget, or quality no better than the
registered control under the prospective decision rule. Any formal
JevArena/JevBench evaluation requires a separately frozen package and
protocol; neither a source identity pass nor a development win qualifies the
Gemma arm for publication.

## Approved zero-step identity result

After separate review and approval, the single locked GPU cell ran on an
available device. Both independent BF16 loads completed all 32 synthetic,
unlabeled inputs. The official source and the freshly inserted **untrained**
q/o LoRA produced maximum absolute drift **0.0** across selected hidden
states, below the frozen `1e-4` tolerance. The strict adapter count and fresh
zero-B checks passed. Loaded parameters remained 25,233,141,760 text and
25,805,933,872 total, with the output embedding tied. No decision head was
scored, optimizer run, formal label read, or model quality evaluated.

The runner exited with status 0 after 111 seconds, a conservative **0.03083
GPU-hour**. Its source, adapter and comparison receipt SHA-256s are
`c82c043a0282ad2534470983146f8437ccff8f60bb6aba2dc86a11ad112b3858`,
`dc9b2dd052457785484f6195beaac5041e441f9698a0019d5c97497cc9d54457`,
and `80ed1da6f56dd68382bc1c50d3dfa99a41720fb264fb083a357b5d9e9be0b4e1`.
The timing receipt SHA-256 is
`b6eecf0986d79079e93c2a0111b9d4112a69c7713f16240eab6ce0bdbee0a44c`;
the source, adapter and comparison log SHA-256s are respectively
`1740755a6140d06dd69217f453b1540e3883ede0948670b54748f30eb387a2c2`,
`057ce08303be682dc32414e1d69ffded894e226b0f8821a9454aa305c230477e`,
and `50e241dca4f939a24a2527272efea42829d6106fc16dd96eca66b3aa45be6643`.
The selected device returned to its pre-run memory baseline and task
containers exited. Private receipts remain private. This admits only the
zero-step adapter/source identity; optimizer, transfer and score remain
untested and subject to their separate gates.
