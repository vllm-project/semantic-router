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
