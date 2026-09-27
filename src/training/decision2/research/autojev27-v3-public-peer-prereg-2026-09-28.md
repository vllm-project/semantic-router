# AutoJev 27B same-panel peer: primary-run preregistration

Status: **inference pending**. This fixes the scope and stop rules before any
typed FINAL, CSS15, or public231 AutoJev prediction is collected. The v3 labels
have already been exposed elsewhere in the project, so this will be a
version-matched comparison, not a new blind confirmation or a Decision 2.0
release result.

| Frozen item | Identity |
| --- | --- |
| Native weight | `denis-pplx/autojev-27b` HF revision `6f5b557e037f5edb25c7dc92dbc6553e5a19c015`; 26,086,635,760 loaded parameters |
| Native runtime | Upstream `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`; local `inference/autojev27.py`, SHA-256 `2075676129c663a93cac4c903ea33cb3116115d32fb7afd2fdce04a41ba3a3ec` |
| Package and runtime attestation | Earlier complete-file receipt SHA-256 `11335b3a8d222b9106cac3eb86546bdb82c9775c02d60254472b01c45eaad935`; native package fingerprint `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2`. Reverify before inference. |
| Cross-process gate | Earlier fixed 32-prompt gold-free smoke SHA-256 `3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55` passed two independent reference-runtime processes: zero changed categories, maximum option probability drift zero. The predeclared allowable maximum is 0.02; accelerated FLA failed and is excluded. Repeat or invalidate this qualification if the image or implementation changes. |
| Typed FINAL | 1,600 original items, 2,000 scored questions, prompt SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd` |
| CSS15 transfer | 6,547 items/questions, prompt SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6` |
| Public JevBench subset | 231 items, prompt SHA-256 `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`; separate report, never called an official sealed JevBench ranking |

The existing typed DEV/CSS pilot scores are *development* controls and are not
reused as formal predictions. Search the private experiment ledger and refuse
to rerun if an identical complete formal peer prediction already exists.
Reverify model/source revisions, full package inventory, calibration and
native loader; require the qualified PyTorch-reference path on one dedicated
GPU. The fixed adapter preserves all questions and native 8,192-token limit;
unsupported or over-budget answers remain invalid in the full denominator.
No truncation, omitted Score questions, manual answer repair or fallback model
is allowed. Abort on missing package bytes, changed prompt/input digests,
incomplete native output or cross-process gate failure.

Collect exactly one primary full prediction per panel, in the order typed
FINAL, CSS15, public231. Keep partial/interrupted attempts with their reason;
do not replace a complete run after seeing labels. Audit each output against
the entire gold-free input, unique IDs, question keys, package/model/source
digests and expected denominators. Record the three prediction SHA-256 values,
run timestamps, image identity and observed GPU-hours **before** any scorer
opens target files. Only after all three audits pass, score with the pinned v3
typed/CSS programs and public231 scorer, keeping the public score separate.
Report typed T, transfer H, `100*sqrt(T*H)`, coverage, per-type/task metrics
and public easy/standard/hard. Do not use these scores to select a candidate or
infer Decision 2.0 success. No model training, upload or publication is part
of this peer run.
