# 9B Score-cardinality residual: technical preflight and matched run

This is one diagnostic arm under the frozen
[registration](qwen35-9b-score-cardinality-residual-prereg-2026-09-28.md).
It starts from official general `Qwen/Qwen3.5-9B` Posttrained revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with a fresh native
shared head plus the zero-initialized Score-cardinality residual. The
completed shared-head control remains unchanged. No formal JevArena v3,
JevBench, or Decision Index score is attributed to this arm.

## Preflight

The first gold-free preflight command failed **before model loading** because
its shard-name predicate expected `model-*`; the pinned official source uses
`model.safetensors-*.safetensors`. All four actual shard content hashes
matched the earlier admission. The predicate was corrected in signed commit
`2b7528ffe17d9112531999c674236344261dccf2`; this changed no source,
data, model, threshold, or optimizer setting. The failure remains in the
runtime record.

The corrected preflight passed source/config/tokenizer/shard hashes,
TRAIN/SELECT/CAL partition hashes and group isolation, 7,324 TRAIN rows,
5,261 TRAIN groups, 99 three-level Score groups, 3,579,176 native TRAIN
tokens and maximum input length 4,089. Under the pinned one-GPU runtime,
the zero-initialized treatment matched the archived shared-head SELECT
baseline on **32/32 valid inputs**, with **zero categorical changes** and
maximum option-probability drift **5.9605e-8**, below the frozen 1e-6 cap.
For three deterministic TRAIN-only three-level Score inputs, its own shared
component gave identical logits and probabilities: **zero categorical
changes and zero drift**. No protected answers were read for this parity.
The zero-step receipt SHA-256 is
`34bc5d9438c1f9f27cabcad4ca63a0b3d5162ad8a57a08080057c00313f8f0f4`.

The separate fresh one-update technical run produced finite loss
**1.590555** and gradient norm **11.178686**; its saved checkpoint passed
a new-process native reload on the fixed first 32 SELECT inputs: **zero
categorical changes, zero p99 drift and zero maximum probability drift**.
The checkpoint receipt SHA-256 is
`0a38ff2b88a3a6bcbb3c6852c533862a7fb9d9e35915f6906b4beb03366dbfe6`.
The one-update SELECT value is not a performance result or alternate
checkpoint candidate. The failed first preflight, corrected preflight,
one-update process and reload occupied approximately 9.74, 35.02,
91.06 and 30.80 single-GPU seconds respectively, or 0.0463 GPU-hours
combined; Docker event start/die times were used for this accounting.

An aggregate-only reread of the 99 TRAIN three-level Score groups found
53 `stage4_ordinal`, 32 `stage4_replay_stage3_logic_score`, and 14
`stage4_dense_table` groups. They come from only two source families:
67 `legacy:stage4-general-composition-v2` and 32 `legacy:stage3_replay`.
Language tags are 51 Chinese and 48 English. Median state length is only
171 characters. This identifies a narrow mechanism and length distribution;
it cannot establish causation from cardinality alone, and no raw training
text or development labels are needed for the finding.

## Frozen 458-update treatment

The full run started from the same original official source, never the
one-update output. Its code and registration were signed in
`9e369832859106073be76f4f6ad77ac7168a1565`; the shard matcher-only
correction is the subsequent signed commit above. The remote runtime uses
an exact mirror of these files. Its data partition hashes equal the completed
control, and its 458/458 optimizer updates, 8/8 prescribed SELECT checkpoints,
and 700/700 valid SELECT answers at every checkpoint were complete. All
reported losses and gradient norms were finite; there was no resume or
alternate seed. The run used **0.793889 one-GPU-hour** including source load,
training, SELECT and checkpoint writes, under the frozen 2.0-hour cap.

| SELECT update | Correct / 700 | Family macro accuracy | Score-five / 90 |
| ---: | ---: | ---: | ---: |
| 64 | 508 | .678241 | 13 |
| 128 | 556 | .766296 | 34 |
| 192 | 594 | .823148 | 62 |
| 256 | 611 | .862778 | 87 |
| 320 | 628 | .874630 | 88 |
| 384 | 636 | .886667 | 90 |
| **448 (BEST)** | **644** | **.906667** | **90** |
| 458 | 641 | .900833 | 90 |

The frozen selector chose update **448** by family macro accuracy. Its
normalized Brier was **.071472**. The completed matched shared-head control
selected update 458 with 625/700, macro .873426 and Score-five 86/90.
The treatment passes the predeclared SELECT retention gates of at least
620/700, macro .863426, Score-five 84/90 and 700 valid. This is a
development-selection result, not evidence of improved three-level Score,
human transfer or a release score. `BEST.json`, `COMPLETE.json`, the BEST448
checkpoint receipt and training log have SHA-256 values
`0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420`,
`b8bba3294d543ee1817174ae1c95e500e7e7bbdac284a0728d754a050c858f3b`,
`523f95b934dac65111546af104a2c668b817ff0b49677927a4b862f3711ef1bf`
and `59d6875fe70d7a0eddbd96f925209991092af7b4818043754acde5c6fc1c9ff3`
respectively. The last hash is for `train-metrics.jsonl`.

The current source tree's control trainer/loss file hashes differ from the
archived control source because optional experimental head, teacher and
ordinal-loss paths were added later. The treatment used a separate copied
trainer to preserve those existing hashes. The active CE + .5 Brier,
zero-teacher, zero-ordinal optimizer path, source/data/seed/token budget,
SELECT roster and update schedule match the archived control. Code and
model-package identities are reported separately; the contrast is an
architectural treatment, not a claim of byte-identical source programs.

A fresh-process native reload of BEST448 passed the same 32 SELECT rows
(13 Choice, 14 Noul, five Score), with **zero categorical changes, zero p99
drift and zero maximum probability drift** against its original SELECT
predictions. Only the selected checkpoint was reloaded. After this gate,
CAL-only temperature fitting produced Choice .896898, Noul .832732, and
Score .05, bound to model SHA-256
`fb46e04c9947f037076dcaa488eb9d49355bdc1804d763fd874bddd21c6ebfa2`.
This calibration affects probability quality, not argmax decisions; its
report SHA-256 is
`5c7dbfb2e36d79bf8c3ef110070ee32cd00faf37b4bea02b62b8ab1eae5f8e1b`.

## Single development readout

Only frozen BEST448 and the audited CAL temperatures were used. A single
gold-free typed DEV1,600 and CSS pilot1,430 collection produced **1,600/1,600**
and **1,428/1,430** valid native answers respectively; the two invalid pilot
answers exceeded the untruncated 4,096-token cap. Before reading either
development key, the ordered prediction files, manifests, input hashes,
calibration and model identity were sealed in a private receipt, SHA-256
`ef4f29509c8e14b7484241b673c7233967ec1686a74f2729510a465ce31d836e`.
The typed and CSS prediction SHA-256 values are
`aa6cef416e43148b8c13ed7afaebe5ab798ea960c11d5180c21ad0d1e065e89d`
and `9fa8597abe0728862bc348e20e9d3731ee48bdc332bc2b24a4a2456fbbef4c29`.
The inputs matched the earlier same-panel control hashes
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
and `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
Only after this seal were the unchanged typed and CSS development scorers
used. Their reports have SHA-256 values
`bd9496995ed51edfdde62055134f4e28c4a27253860dce0ad0c262f54c0d8ca1`
and `3c7d271190ffee11a53653a47c39671ee4d28b87f3f7618cca7f6e446874fe02`.

| Development metric | Completed shared-head official control | Score-cardinality BEST448 | Change |
| --- | ---: | ---: | ---: |
| Typed four-family macro `T` | .683125 | **.808125** | +.125000 |
| Typed correct / 1,600 | 1,093 | **1,293** | +200 |
| Choice / 800 | 726 | **794** | +68 |
| Noul / 400 | 193 | **262** | +69 |
| Three-level Score / 400 | 174 | **237** | +63 |
| Typed normalized Brier ↓ | .24611 | **.15766** | −.08845 |
| Typed ECE10 ↓ | .19151 | **.08407** | −.10744 |
| CSS pilot task-median macro-F1 `H` | **.54927** | .50140 | −.04787 |
| CSS pilot correct / 1,430 | 813 | 785 | −28 |
| Proxy `100 × sqrt(T × H)` | 61.255 | **63.655** | +2.400 |

The typed family results were attribute gate **400/400**, transition table
**394/400**, rule precedence **262/400**, and set reconciliation **237/400**.
The Score-three-level confusion rows (gold levels 0/1/2, predicted columns
0/1/2) were `[85, 0, 0]`, `[62, 4, 41]`, and `[60, 0, 148]`.
Only **4 of 107** true middle-level items were recovered. The old control
never predicted the middle level, so the residual modestly changed that
behavior but did not solve it. The SELECT five-level ceiling therefore did
not transfer to the three-level rule/state mechanism.

| CSS pilot task | Completed control macro-F1 | Treatment macro-F1 | Change |
| --- | ---: | ---: | ---: |
| Discourse (497 items, 495 valid) | .54927 | **.50140** | −.04787 |
| Implicit hate (498, all valid) | .41348 | **.40605** | −.00743 |
| SemEval stance (435, all valid) | .75438 | **.73800** | −.01638 |

The decline spans all three human-label pilot tasks and particularly the
task that determines their median. It is not attributable to an increased
invalid-answer count; the old control also had two over-budget discourse
answers. This is development evidence about this one matched arm, not a
formal transfer result.

## Disposition and next discriminating variable

The prospective mechanism screen required at least **245/400** Score-three
and proxy **65.0**, while retaining Noul at least 193/400. Noul passed, but
Score missed by eight items and proxy missed by **1.345** points. It also
falls well below the unchanged own-Lux1-relative formal-candidate target of
approximately **72.326** proxy. The arm is **HOLD**. No alternate checkpoint,
threshold, residual scale, CAL choice or seed was searched after this result;
no typed FINAL, CSS15 FINAL, JevBench, Decision Index, model upload or release
claim followed.

The next *proposed*, not launched, single-variable contrast is a same-start
soft-distribution retention objective: keep official source revision,
Score-cardinality readout, 7,324 original TRAIN rows and order, 3,579,176
tokens, 458 updates, optimizer and SELECT protocol fixed, and add one
predeclared small KL term against a frozen **own Decision-1.0-Lux-9B** native
distribution on those same TRAIN inputs. This changes the training target,
not initialization or row exposure. Before any GPU work, audit teacher
weights/version, full three-type probability coverage, source overlap,
Score-middle probability quality, teacher inference length, and loss
finiteness; choose the KL weight from prior protocol, not from this opened
DEV/pilot result. The hypothesis is that soft retention will recover human
transfer while preserving the readout's typed gains. A failure would point
toward new source-disjoint human and long-state Score evidence rather than
another bias-scale search. No comparable same-start 9B soft-Lux treatment
was found in the existing experiment ledger.

All technical and training processes together consumed **0.941439
one-GPU-hour** by Docker start/die events, including the failed early file
predicate, zero-step parity, one-step/reload, full training, selected reload,
CAL, and both gold-free development collectors. CPU-only scoring consumed
no GPU-hours. The task GPU reservation was closed after the collectors;
the full run itself used 0.793889 GPU-hour by recorded start/end timestamps.
