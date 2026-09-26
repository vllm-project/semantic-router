# GLiNER 0.6B native human TRAIN head-only screen — preregistration

Status: **design and implementation only; zero optimizer updates**. This is a
single source-preserving development experiment, not a Decision 2.0 checkpoint
or a release evaluation. It does not repeat the earlier R111 full-backbone
64-step mixed-data recipe.

## Source and motivation

The English [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
checkpoint is pinned to revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`
and weight SHA-256 `40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997`.
Its actual **486,444,053** parameters, `span` architecture and 512-position
encoder limit were measured from the released files. The source and pinned
[GLiNER2 library](https://github.com/fastino-ai/GLiNER2) at
`55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` are Apache-2.0. Inference
uses the published `Classifier` through existing
`gliner25-native-exclusive-v3` Choice/Noul/Score projection; it never
truncates long inputs. The source's same-panel development scores are typed
DEV **652/1,600**, human CSS pilot **561/1,430** (median task macro-F1
`.31035`) and the public 231-item subset **116/231**, with 56 native
overflows. Its prior SELECT score is **372/700**, family-macro `.387193732`.

R111's 64-step full-backbone continuation on 1,024 mixed TRAIN rows raised
SELECT to 491/700 but degraded independent typed DEV from 652 to 584,
including transition 21 to 4/400. This screen tests whether a **classifier
head-only, 10× lower task LR, human-label-only** update can preserve that
source behavior while improving independently annotated transfer. It is a
different causal intervention; a SELECT win alone is insufficient. Human
TRAIN has **no Score supervision**, so this screen cannot by itself support a
complete three-type Decision 2.0 claim.

## Data and source freeze

Use only the already audited rights-clean v2 TRAIN/SELECT/CAL partitions,
7,455/700/700 rows, and exact source manifest. SHA-256 values are TRAIN
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
SELECT `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
CAL `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`,
rights manifest `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
Recheck IDs, source groups and canonical-input disjointness before writing
private training rows. Keep source text and labels private; the rights
manifest records upstream licenses and redistribution limits.

The eligible human-annotated English TRAIN sources are GoEmotions 2,800 rows,
CosmosQA 448, SNLI 272, SQuAD2 332 of 334 native-admitted, and FLUTE 120.
All others are excluded; the two SQuAD overflows are excluded without
truncation. Use the identical native classification schema builder from R111,
which disables native training-time label dropout/synthesis. Draw **512**
native-512 rows by fixed SHA-256 rank of seed
`decision2-gliner25-human-head-20260927-v1` and source group. Preserve 128
complete GoEmotions Choice/Noul group pairs (256 rows), plus one row from
each of 80 CosmosQA, 80 SNLI, 48 SQuAD2 and 48 FLUTE groups. This yields
**384 independent source groups**, Choice 336 and Noul 176, zero Score.
Within a multirow source group, choose one row by the same seeded hash.
Write the native TRAIN export and a hash/count manifest only to the private
experiment store. SELECT and CAL cannot affect the group draw.

## Fixed optimizer and gates

Freeze encoder, span representation, count modules and all other source
weights. Train only `classifier.0.{weight,bias}` and
`classifier.2.{weight,bias}`: exactly **2,101,249** trainable of 486,444,053
parameters. First verify byte-pinned source, native source/zero-update forward
parity, BF16 GPU, finite nonzero head gradients, no encoder gradients and
optimizer groups containing only the four head tensors. A private
pre-optimizer receipt must bind source/data/code/image hashes before step one.
Failure blocks optimizer launch.

Run exactly **32** updates, batch 2 × accumulation 8 (effective 16), one
epoch over 512 rows, task LR `5e-6`, encoder LR `1e-6` with an empty encoder
parameter group, AdamW weight decay `.01`, betas `.9/.999`, epsilon `1e-8`,
four warmup steps then linear decay, gradient norm cap 1, BF16, native 512
positions, seed `20260928`. Preserve exact schema; no label augmentation,
invalid-sample skipping, CAL fitting, teacher output or Jev API data. The
single fixed checkpoint is step 32; no checkpoint selection on SELECT or
downstream panels. Verify save and native reload. For identical source/candidate
SELECT, reuse the prior source prediction only if its data and adapter hashes
match; otherwise rederive it before candidate scoring.

SELECT700 is the sole training-stage gate: final candidate must gain at least
**14 correct** and **.02 family-macro accuracy** over source, lose no Choice,
Noul or Score correct answer, lose no more than `.025` accuracy in any one
of the six SELECT families, and have no additional invalid/overflow answer.
Every condition is required. Report all type/family counts and invalids.
If any condition fails, stop and keep the checkpoint private as a negative
result without typed DEV, CSS or public subset evaluation.

Only a SELECT pass permits one same-adapter independent typed DEV1,600 and
human CSS pilot1,430 comparison. Before any public-subset score, require
typed DEV at least **668/1,600** (+16 over source), nonnegative Choice,
Noul and Score deltas, no four-family loss over eight of 400 items,
CSS at least **589/1,430** (+28), and CSS median task macro-F1 at least
`.33035` (+.02). Report 400-group paired confidence intervals and task-level
CSS detail. If this fails, no public-subset or sealed evaluation occurs.
The CSS pilot tasks may share source families with TRAIN; any gain is not
automatically unseen-task transfer. Only after both panels pass may the
public 231-item diagnostic run once, distinctly labeled as a public subset.
No sealed FINAL, CSS15, or authored release labels enter this screen; even a
positive development result would need longer-input, multilingual,
calibration, robustness, release-set and package-parity gates before
publication.
