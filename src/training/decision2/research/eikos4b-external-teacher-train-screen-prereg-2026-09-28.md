# External decision teacher: bounded TRAIN-only screen

**Status:** prospective source screen, before any teacher inference or student
training. This is an internal experiment note, not product-card copy. The
published [Eikos-4B model](https://huggingface.co/caiovicentino1/Eikos-4B)
is a permitted third-party **teacher and benchmark peer**, never a direct
Decision 2.0 weight initialization. Its published card describes a native
letter-logit typed decision readout. The screen below tests whether its
distributions are usable on our existing TRAIN semantics; it does not transfer
its model weights or establish a generalization gain.

## Frozen scope

| Component | Commitment |
| --- | --- |
| Teacher | `caiovicentino1/Eikos-4B@582ffb13f19a4da3f455e3db198584190bd7755b`, native `serve.Decider` PyTorch readout and release `SHA256SUMS` |
| TRAIN | Rights-clean v2, exactly 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| Pilot roster | Deterministic SHA-256 order of gold-free group/row IDs; one row per independent TRAIN group, 32 each of Choice, Noul and Score; the 96-row roster hash is frozen after a remote CPU dry run and **before** loading the teacher |
| Interface | Existing typed `state`, `instructions`, live option descriptions and ordered Score levels; no chat completion or label in the request |
| Evaluation | TRAIN hard-label agreement, gold probability and normalized Brier as descriptive source checks, with invalid/tie counts and full 96-row denominator |
| Budget | At most 15 minutes on one exclusive GPU, including source load; no retry with changed roster, prompt, calibration or source revision |

[`eikos_teacher_train_pilot.py`](eikos_teacher_train_pilot.py) verifies the
publisher release checksums, strict local TRAIN schema and file hash, the
precommitted roster, returned option keys, finite normalized probabilities and
each typed question. Its private receipt contains only aggregate scores,
source/roster/code hashes and no raw rows or per-question predictions. Any
overflow is an invalid answer. Unanticipated errors stop the screen. The
pilot cannot feed a student optimizer or be called JevArena/JevBench evidence.

The first CPU run must record the exact private input paths, source-file
checksums, container image, free GPU and roster SHA in a private execution
receipt. Only after the frozen script is mirrored from the signed local commit
may the GPU run start. All 96 native answers must be structurally valid for
this source to be considered for a later, separately registered distillation
arm; the outcome remains descriptive, since 32 groups/type are too few for a
publishable capability claim. A later arm must fix teacher coverage, KL
weight, student origin, matched token/step budget and independent development
gate in advance. Existing own-Nox or official-Qwen controls are retained; this
screen never substitutes Eikos weights for an eligible origin. Any teacher
overlap with protected task sources must be documented before a student is
promoted.

**No GPU result or distillation conclusion exists at preregistration.**

## CPU roster and runtime lock before GPU

The source code is signed as commit `36ce3428f`; the exact local and SSH
mirror script SHA-256 is
`1337811ee0fb8f41769d9ea6488d0a4c4aba6e0994a375593be89e061f201b13`.
The CPU-only dry run on the unchanged TRAIN bytes selected 96 rows from 96
groups, roster SHA-256
`18dce35a5ca58864f1b92399344fab679ec98fb7ff4ddd05ee71cfeecebb1722`.
The qualified image is `decision20-train-fast:host2` at ID
`f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`,
with PyTorch 2.12.0, Transformers 5.17.0 and FLA 0.5.2. Reserve only
physical GPU 4 on the first authorized node after a fresh live-memory check.
The second authorized node's one idle GPU does not currently have this
frozen TRAIN and source package, so transferring them for a 96-row screen
would add no useful experimental contrast. No teacher model was loaded and no
TRAIN answer scored during this lock step.
