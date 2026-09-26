# Sol 2B rights-clean GoEmotions control

This direct Decision 1.0 Sol 2B to Decision 2.0 control adds a larger,
human-labeled natural-language source to the earlier rights-clean synthetic
control. It tests whether that source improves transfer without changing the
2B architecture or optimizer. It is a development result; synthetic final
and CSS15 gold remain sealed.

## Data and source

The frozen rights-clean v2 TRAIN7455, SELECT700, CAL700, and manifest
SHA-256 values are respectively
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
`3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`,
and `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
They include 2,800 official GoEmotions TRAIN-derived human labeled rows,
plus 400 human rows each in SELECT and CAL; comment groups remain isolated.
The data manifest records source rights, IDs, split assignment, and exact/
approximate near-context audits against protected panels. The official
GoEmotions dataset is credited under CC BY 4.0. The warm-start is the
published Decision 1.0 Sol 2B revision
`0665a41108e8f0b33a9515c98311c45947b99399`; historical upstream
training-row exposure remains unknown.

The direct v2 control keeps the previous matched rank-16 LoRA protocol:
alpha 32, dropout .05, LoRA LR `1.5e-5`, head LR `7.5e-6`, CE + 0.5 Brier,
effective batch 16, one epoch, max 8,192 tokens, SELECT every 32 steps, and
seed `20260926`. SELECT and CAL were not used for gradient updates. Training
completed all 466 updates and froze BEST step 448. `BEST.json`,
`COMPLETE.json`, and run provenance SHA-256 values are
`0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420`,
`f7d2b847c2816554b13b4fdd566470c8af56fba113a0d88d85f4c6ab99fc87b4`,
and `d6026f3974020d72b821f891aa271ffd8d8be2ad264fc7e1826e2c0983d3b0c7`.

## Selection, calibration, transfer

On SELECT700 the Sol 1.0 initialization was 506/700 and the frozen BEST
was 570/700 with family-macro accuracy `0.77583`. The human GoEmotions
Choice slice increased from 134/200 to 158/200; Noul from 169/200 to
178/200. The BEST SELECT report SHA-256 is
`0539425de2781cad547fbcf76e2e449b8bce206a1e355ae951a768f689b8cc30`.

Native CAL700 fitted temperatures Choice `1.167573747`, Noul
`1.231983086`, and Score `0.632143722`, all inside the optimizer bounds.
CAL was 558/700 (79.71%) with post-fit ECE `0.03146`. The selected model
fingerprint is
`1f8aef4cfcc482cc1e5ada110106bf436dffc45c5e5c01579e5551bf1e02abce`;
calibration receipt SHA-256 is
`05c0a7a8fcf96cc7d82ea3d4e02796af097bd1a2f32c3e5bfa57b93eac2cf042`.

The independent calibrated native adapter had 100% valid predictions on
DEV1600 and CSS pilot1430. DEV was 937/1,600 (58.5625%), by family:
attribute gate 249/400, rule precedence 227/400, set reconciliation
294/400, and transition table 167/400. DEV prediction and report SHA-256
values are
`c1a8b4222ed3e3dbb938c07aa6ca487d2296eb1b3b6c7e08a0870237e202a64a`
and `555611c3bdbc74ec4a3b6717ed7fdf8981deaf6232d128e63e6790fd6473275b`.
CSS pilot was 577/1,430 (40.3497%) micro accuracy and median task
macro-F1 `0.323366`: discourse 174/497, implicit hate 150/498, stance
253/435. CSS prediction and report SHA-256 values are
`0ba91a91adbfb2b9e4fca208c1a0ff19fcae6b5ea3c4839b2fa7b55308486413`
and `e154d255b40dbf29c77f8cd2fba0dc24691ef8a56beec965a2d4db92b1b2e26d`.

The matched Sol 1.0 baseline reached DEV 58.9375%, CSS micro 38.4615%,
and CSS median macro-F1 `0.31554`. This control improves CSS accuracy and
F1 but loses 0.375 DEV points. The earlier targeted3024 BEST160 reaches
DEV 59.125%, CSS micro 42.5175%, and median macro-F1 `0.34763`; the
direct v2 control trails it on all three. The clean v1 control reaches
DEV 59.50%, CSS micro 39.51%, and median macro-F1 `0.30712`; v2 trades
synthetic accuracy for a modest CSS improvement. These development panels
do not support a broad-gain or open-SOTA selection claim.

## Same-size public JevBench diagnostic

On the pinned 231-item *public* panel, the same native calibrated checkpoint
was strictly valid on every item. It scored easy 48/48, standard 66/72,
and hard 50/111: 164/231 (70.996%) overall, tier-macro 78.904%,
Brier `0.22216`, and pmax ECE `0.15633`. The published Sol 1.0 baseline was
161/231, with 47/111 hard; the targeted3024 BEST160 arm was also 161/231,
with 46/111 hard. The +3-item overall gain is small and accompanies worse
calibration than Sol 1.0 (Brier `0.20545`, ECE `0.10394`). It does not
establish improvement on JevBench's sealed tasks or the official composite.
Prediction, companion manifest, and scorer report SHA-256 values are
`5a914b32f59532bc95ab0bb41cdbd498b561e000bcf3aeb6994aa8a58da70a5c`,
`caa09d226487014a451ae96d0193ac47a60aaf7342384967bd8aa86fe9447a3a`,
and `8720df85ff5f34e923c40cd72e9db6364b90d7c23e30de7bc4068ad9c16cf5cb`.
