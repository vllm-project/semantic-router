# 0.6B official posttrained source: fixed shared-head contrast

**State: prospective source-only experiment; no optimizer update has occurred.**
This contrasts the official general posttrained Qwen3-0.6B backbone with the
completed official Qwen3-0.6B Base **shared-head** control. The failed
candidate-interaction architecture arm remains a separate HOLD and contributes
no model-selection score here. This is an internal preregistration, not a
model-card or release claim.

## Source identity and exposure gate

The control source is `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; the treatment is the official, unmodified general posttrained
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. Both
source LICENSE files are byte-identical Apache-2.0. The pinned treatment
`model.safetensors` SHA-256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
`config.json` is
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
The [CPU source audit](../scripts/audit_qwen06_posttrained_source.py) pins
every weight, config, tokenizer, vocabulary, merges and LICENSE file for both
sources, plus each data partition, before GPU work.

The tokenizer JSON and tokenizer config bytes differ between official Base
and posttrained. This is **not** ignored: for every rights-clean v2 TRAIN
7,455, SELECT 700 and CAL 700 row, the native decision renderer returned
identical token IDs, candidate endpoints and option keys between the two
sources. All 8,855 rows matched. TRAIN exposure is exactly 4,094,489 input
tokens on both; SELECT and CAL have 90,505 and 89,827 respectively. The audit
receipt SHA-256 is
`269427d9be15e8bf47876e0c8bfd62bdfa730e1135eac2235e6df4b6ef8b6077`.
CAL was read for source/length/split audit only, never used to select or fit a
model. The official safetensors contain 751,632,384 parameters, of which
155,582,464 are the language-model output head. The native Decision scorer
retains the same 596,049,920-parameter Qwen3 backbone as the control and
attaches the same 1,053,184-parameter shared decision head; loaded treatment
size should be 597,103,104 parameters, subject to the GPU loader check.

## Frozen training and selection

| Item | Treatment rule |
| --- | --- |
| Initialization | Official posttrained backbone above, new shared candidate head, seed `20260926`. The head parameter tensors must exactly match the control's seed-matched fresh shared head at zero-step. The backbone is intentionally different. |
| Data | Same rights-clean v2 TRAIN 7,455, SELECT 700 and CAL 700 source groups and row order as the completed Base control, with pinned SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`, `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`, `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`. No new teacher, replay, class weights, data replacements or changed labels. |
| Exposure | One epoch, 7,455 examples, 4,094,489 renderer tokens, one GPU, microbatch 1, accumulation 16, exactly 466 updates and the same deterministic length-bucket order. Complete input is capped at 8,192 tokens without truncation. |
| Loss and optimizer | CE + `0.5` normalized Brier; AdamW decay `.01`, clipping `1.0`, FP32 parameters and head, BF16 backbone compute, gradient checkpointing. Backbone peak LR `2e-5`, head peak LR `2e-4`, `.05` warmup and cosine tail. |
| Candidate snapshots | Evaluate SELECT at exactly steps 64, 128, 192, 256, 320, 384, 448, 466. Choose highest six-family macro accuracy, then lowest macro normalized Brier, then earliest step. No alternative checkpoint search. |
| Advancement | The selected checkpoint must have **at least 562/700 correct**, six-family macro accuracy **at least `.77259`**, all 700 valid native answers, finite numerics and complete provenance. This matches the completed Base control; failure is HOLD with no typed DEV, CSS pilot, CAL scoring, formal, public or HF evaluation. |
| Budget | Zero-step and single-update technical preflight are separate. The full arm has at most **1.5 single-GPU hours**, a fixed hard stop rather than an invitation to resume or change optimizer settings. |

Each model uses the existing typed System One renderer and native probability
readout. Choice supports 2–255 named options, Noul binary, Score 2–10 ordered
levels; this experiment does not convert these to chat generation. Evaluation
adapters and scoring stay fixed. The treatment has no new architecture. The
shared head still sees causal option order, so no permutation-invariance claim
is made.

## Technical gate and downstream boundary

Before the first optimizer update, freeze the signed local code commit, exact
source/data/code/image hashes, private launch command, exclusive live GPU
reservation and the passed CPU audit above. Fresh-load Base and posttrained
with the same seed and require exact shared-head weights. The treatment must
return finite normalized Choice/Noul/Score probabilities twice, with zero
changed winners and probability drift at most `1e-4`, in at most 540 GPU
seconds. The first fixed 16 TRAIN examples then take one update after finite
per-type loss and head-gradient probes. Save a full native checkpoint, reload
it in a fresh model instance, and require 32/32 fixed SELECT items to retain
their winner with maximum probability drift at most `1e-3`, within another
180 GPU seconds. Any source, token, budget, numeric or reload miss is a
recorded HOLD, with no automatic retry.

Only if all technical gates pass may one full 466-update treatment start. It
must produce all eight SELECT reports and obey the advancement rule above. If
SELECT passes, **seal the candidate and notify the coordinating agent before
any downstream evaluation**. A separate protocol must then freeze native
typed DEV and human-transfer predictions and specify an independent transfer
check before reusing the already known JevArena v3 formal labels. The current
arm does not run that downstream work or claim a formal gain.
