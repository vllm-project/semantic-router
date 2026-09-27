# Kev-4B external teacher: TRAIN-only screen stopped

**Decision: HOLD.** The one permitted GPU invocation stopped on a native
probability-validation error. It produced no aggregate performance receipt, so
none of the preregistered Choice, Noul, or Score teacher-signal thresholds can
be evaluated. No Decision 2.0 student training or benchmark scoring follows
from this screen.

The [prospective protocol](kev4b-external-teacher-train-screen-prereg-2026-09-28.md)
fixed 96 independent rights-clean TRAIN groups, 32 per type. Before GPU use,
the exact TRAIN SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
roster SHA-256
`18dce35a5ca58864f1b92399344fab679ec98fb7ff4ddd05ee71cfeecebb1722`,
Kev model revision `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`, its
publisher source revision `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`,
30 publisher source-file hashes, and offline official Qwen base revision
`1001bb4d826a52d1f399e183466143f4da7b741b` matched. The model
fingerprint was
`8e2c7fff2ef6ad7b195443fac287fb1ae4cd83c8c9af1a12dfb743501a3ec3e9`.
The local source commit was `8cce95018b97bbf65a06c6dc5ad902588b988dcc`;
the mirrored pilot script and native adapter SHA-256 were respectively
`56e499038af0848004b4e6f8f602f146a54a8d614fdc64d85555d44b6aa64f61`
and `5439c971654a8edc5e14c314b5846fb214d6ffc6dfc6c6f9f2d00e51ac206fc9`.
The pinned runtime image ID was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.

The single model run terminated after **36.94 seconds**, an upper bound of
**0.0103 GPU-hours** on one isolated GPU. Its first fatal exception was
`ValueError: Invalid native teacher distribution` in the frozen distribution
validator. The validator had already passed type and option-key checks at that
point, but no offending probability vector or row identity was retained, so
the numerical subcause cannot be proved from this run. Static inspection found
a protocol mismatch that can cause this failure: Kev's published API rounds
each returned option probability to four decimal places and documents an
absolute sum tolerance of `0.02` for up to 255 options, while this pilot's
shared validator requires the rounded values to sum to one within `1e-5`.
The run was stopped as preregistered; the validator and roster were not changed
and the model was not retried.

The private failure receipt has mode `0600` and SHA-256
`c59da6b21acc8d566f3b6f7c004fc20a73f2396ae98866d7274d8ed95b2b2547`.
No aggregate answer receipt exists, and no raw TRAIN text, row-level output,
or target distribution was committed. The container exited and released its
GPU. The local changed-file check and training-contract tests passed before
the run.

Any further Kev signal experiment needs a separately frozen validation rule
that explicitly handles the publisher's output rounding, plus CPU tests for
different option counts. This failed run supplies no evidence for a
three-type or type-restricted distillation arm.
