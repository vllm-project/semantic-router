# Sol 2B targeted-to-human continuation

This is a separate 2B development arm that starts with the frozen
targeted3024 SELECT-selected step 160 and continues on the rights-clean v2
GoEmotions curriculum. It does not replace the direct Sol 1.0-to-v2 control,
and no final benchmark or CSS15 gold is used for selection. The existing
targeted model was LoRA; the trainer requires a full Decision 2.0 source to
attach a fresh adapter, so the selected LoRA was explicitly materialized and
audited before continuing.

## Source and parity gate

The targeted step-160 adapter model SHA-256 is
`49ce326b5b116ed81397cda874f6e287e8c37126ec2f1f86e7140a4ca39ff629`.
The FP32 PEFT merge produces full-model SHA-256
`2f4bb061e0881d2d5f29da1cedee8655655ce339bacfa4f6971bcd845add5ae9`;
the receipt SHA-256 is
`eae632fba65bc3b208aa3698d2334a88cbd5a0a4b7521ade9398fa3be3aacd1c`.
The receipt binds exact source and merged file hashes and was verified before
inference. Its source fingerprint includes the pinned Decision 1.0 Sol source
and selected adapter. The same hard900 type calibration, SHA-256
`0ed1805105febba3c1639a8b230781085913d0c6e80e456a18f036f354c61227`,
was used only to compare the source model behavior.

Gold-free 16-prompt direct forward audits used both precision modes. FP32
gave zero Choice flips and maximum calibrated option-probability difference
`0.000508` (report SHA-256
`257e69d6290dae750b0523010b80512a098f8bfd2aff560cd806298debe19796`);
BF16 gave zero flips and maximum difference `0.009114` (report SHA-256
`b2346be57732b205aea1b826417928f4f318d1a6e6bcd91cb9facf4dca0164df`).
These are not exact numeric parity. A prior separate native adapter comparison
on the same gold-free sample reached `0.02013` maximum probability drift,
which also shows execution sensitivity.

On the full 1,600-item DEV prompt panel, independent BF16 native inference
with the same calibration yielded 7 Choice flips, 3 Noul threshold flips,
and no invalid pairs between unmerged and merged models. Absolute calibrated
probability differences had mean `0.00691`, p95 `0.02175`, and maximum
`0.09832`; 344/1,600 exceeded `0.01`. Score expected-value absolute
differences had mean `0.01116`, p95 `0.04133`, and maximum `0.06820`.
The comparison report SHA-256 is
`e7f4a69774924179b430c922e3cade894ab7ebf4b0138545bbb4fb944a0c7d89`;
merged native predictions SHA-256 is
`8462a39aad1ac45b8ce39e35917276328244fc3f799792705e6e67f849b28376`.
The merged source independently scored 939/1,600 (58.6875%) on DEV, against
946/1,600 (59.125%) for the selected unmerged adapter; merged report SHA-256
`6f3907450a1f6c78d91daf940aac52f533ee6832a00603b75fdb7bb02e20ec4e`.
The merged full model is therefore identified as a **distinct warm start**;
the unmerged adapter's metrics are never transferred to its continuation.

## Matched continuation protocol

The frozen rights-clean v2 TRAIN has 7,455 rows and SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
SELECT700 and CAL700 SHA-256 values are
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`;
the data manifest SHA-256 is
`61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
They add human GoEmotions labels while preserving synthetic anchors. The
data audit excludes exact/near protected-panel overlap; it is not a semantic
independence proof.

The continuation uses an isolated mirror of locally committed training code
at `d87dcc508d46d49f3fcc273c43e037d9ee2fbd8f`. It holds the direct-v2
control's rank-16 LoRA, alpha 32, dropout .05, LoRA LR `1.5e-5`, head LR
`7.5e-6`, CE + 0.5 Brier, batch 16, one epoch, 8,192-token limit, step-32
SELECT, and seed `20260926` fixed. The only intended change is the source
initialization. On new SELECT700 before optimization, the merged source was
572/700, including GoEmotions Choice 130/200 and Noul 169/200. The direct
Sol 1.0 control began at 506/700, including Choice 134/200 and Noul
169/200; most of the baseline gap comes from previously learned synthetic
targeted tasks. SELECT results across this matched pair are comparable;
earlier targeted SELECT600 results are on different items.

After SELECT freezes a complete checkpoint, fit native per-type temperatures
on CAL700, then independently run DEV1600 and CSS pilot1430. Report the same
validity, family/task breakdown, calibration and transfer metrics as the
direct-v2 control. A gain on synthetic SELECT alone is insufficient.

## Same-size public JevBench diagnostic

Before continuation, the original unmerged targeted160 adapter was run on
the pinned 231-item *public* JevBench panel with its hard900 calibration.
All 231 answers were strictly valid: easy 48/48, standard 67/72, hard 46/111;
overall 161/231 (69.697%) and tier-macro accuracy 78.166%. The Sol 1.0
baseline also scored 161/231 overall and 47/111 hard. This targeted arm has
no same-size public JevBench gain, and the public panel is not the sealed
official JevBench composite. Prediction, companion manifest, and score
SHA-256 values are respectively
`ae7e4a993c5b968f4201ad29f884f86430a312a53693f89eeceb50ca383f4bdc`,
`832aaf7ba02417bd4e0d37d54330b8fa4a44b4030d5e1fbc648d6d06cfbb125f`,
and `c0774d9d069374b777aa50fa0c770e5bcc0af9732b0026a684f2ca9ebc49d94a`.

## Frozen continuation result

Training completed all 466 updates and selected BEST step 320. The run
provenance, `BEST.json`, and `COMPLETE.json` SHA-256 values are
`9866b2d5cb94f4250262b7da2585136bffee66d81b5ac717fb2e07bfa30fb8cc`,
`6c327296742869b0c9f29e00044d82fbdb84a09e036f3e83b7348f0317b087cc`,
and `b6ca2f261248a3e2d0f0e3f47edf2db207763e09bf643d62b7d1e18e990bd2d8`.
SELECT700 was 614/700, family-macro `0.861667`. Human GoEmotions Choice
was 157/200 and Noul 177/200, slightly below the direct-v2 control's
158/200 and 178/200; the main difference was prior synthetic targeted
learning (quantized median 90/90 versus the control's 45/90).
The selected SELECT report SHA-256 is
`fdeec4470335f049c696d685010f441616255413ae638f8c55d709935206a3bf`.

The selected adapter model SHA-256 is
`e6abc21e943cb8113b9f7a63aed67fd6373ef2f2f30ae3466de34476525001af`.
Native CAL700 fitted Choice `1.184990459`, Noul `1.187191802`, Score
`0.824924662`, all interior temperatures. CAL was 602/700 (86.0%) with
post-fit ECE `0.02469`; CAL receipt SHA-256 is
`761c921403129ff7c70a43fd869f8f10641fc560ce9323b281c25724d6936d86`.

The independently calibrated native adapter was valid on every item of
DEV1600 and CSS pilot1430. DEV was 933/1,600 (58.3125%): attribute gate
252/400, rule precedence 224/400, set reconciliation 291/400, transition
table 166/400. Prediction and score SHA-256 values are
`48ae551a4eae173debb4c6aa1604ccecfe890d59dee5928355226ccd86ca3264`
and `929d9f127e0451279a0c7a7ba92a766f564b52c7b5b608c961c4f98a4353783a`.
CSS pilot was 629/1,430 (43.9860%) micro accuracy, median task macro-F1
`0.361834`: discourse 195/497, implicit hate 160/498, stance 274/435.
Prediction and score SHA-256 values are
`3a1b1bdbf26a6fd2ffb6930a592d243fdfa5ab06c2cd6a7c73659b873fad2169`
and `3344e4ea6c26f25e4a9351a3cbf1e5bb8d63aced8f5ac786c9633391ef450240`.

On the pinned public-only JevBench231 panel it was strictly valid 231/231:
easy 48/48, standard 67/72, hard 48/111; overall 163/231 (70.563%),
tier-macro 78.766%, Brier `0.22636`, pmax ECE `0.16356`.
Prediction, companion manifest, and score SHA-256 values are
`b0ad4b106ca0633b0f23bd7920cfb0cf67b14885a1a8a14bc2b3c3e4bbb3687d`,
`e7bdbf6ccf5b512b9c64d03031bb5434cb9f2388bf498596618402646989c9cd`,
and `90ed28bd6cbfca687bf3702509a2995c9b1462ecf78b95ed033c4098004d7272`.

| 2B model | DEV1600 | CSS pilot micro | CSS median macro-F1 | Public JevBench231 |
| --- | ---: | ---: | ---: | ---: |
| Sol 1.0 | 58.9375% | 38.4615% | 0.31554 | 161/231 |
| Targeted3024 BEST160 | 59.1250% | 42.5175% | 0.34763 | 161/231 |
| Direct clean-v2 BEST448 | 58.5625% | 40.3497% | 0.32337 | 164/231 |
| Targeted then clean-v2 BEST320 | 58.3125% | **43.9860%** | **0.36183** | 163/231 |

The continuation is the best 2B CSS pilot transfer arm here, adding 1.47
accuracy points and `0.01420` median macro-F1 over targeted3024 BEST160.
It loses 0.8125 DEV points versus targeted3024 and 0.625 versus Sol 1.0,
and trails the direct clean-v2 control by one public JevBench item. It
therefore fails the requested broad 1.0-overall improvement gate. The
public-only JevBench count is not an official sealed composite or a claim
of statistical significance; all comparisons are development diagnostics.
