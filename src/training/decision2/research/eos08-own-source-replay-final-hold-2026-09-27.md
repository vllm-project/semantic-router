# Eos 0.8B own-source replay: development HOLD

**Decision:** The preregistered soft replay intervention failed its fixed-step
development gate. Do not open typed FINAL, CSS15, or JevBench for this arm, and
do not package it as a Decision 2.0 release candidate. This result applies to
the fixed own-Eos initializer, rights-clean v2, 512 repeated TRAIN rows, KL
weight 0.5, and reference runtime. It does not decide a new Score-data arm.

The prospective [v2 protocol](eos08-own-source-soft-replay-v2-prereg-2026-09-27.md)
and [same-node readout amendment](eos08-native-node-readout-amendment-2026-09-27.md)
were signed before the corresponding training and soft open-panel reads. Both
arms started from our `Decision-1.0-Eos-0.8B` at revision
`3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd`, used the same audited
TRAIN/SELECT/CAL and 498-update schedule, and differed only in repeat-row
source-probability KL weight: 0 for hard control, 0.5 for soft treatment.
The original failed cross-process BF16 v1 gate remains a separate negative
record; the deterministic v2 runtime did not retroactively rescue it.

## Sealed open-development readout

The typed DEV has 1,600 answers; CSS pilot has 1,430 across three tasks. T is
typed family-macro accuracy, H is the CSS task-median macro-F1, and the
development proxy is `100 sqrt(T × H)`. These labels were open before this
experiment and do not provide blind release evidence.

| Native package | Typed correct | Choice | Noul | Score | T | H | Proxy | Typed Brier | Score Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Own Eos 1.0 | 792 | 510 | 197 | 85 | .495000 | .192001 | 30.8286 | .361777 | .662235 |
| Hard fixed498 | 797 | 510 | 202 | 85 | .498125 | .216212 | 32.8177 | .430316 | .779280 |
| Soft fixed498 | 802 | 512 | 205 | 85 | .501250 | .217000 | 32.9805 | .422290 | .775549 |
| Hard SELECT-BEST480 | 800 | 512 | 203 | 85 | .500000 | .215602 | 32.8331 | .431740 | .779405 |
| Soft SELECT-BEST448 | 799 | 512 | 202 | 85 | .499375 | .218209 | 33.0104 | .418590 | .773668 |

The causal, fixed498 soft-minus-hard proxy delta was **+0.1627**, short of the
frozen +1.0 criterion. Typed Brier improved only **0.00803**, short of the
alternative 0.01 criterion. Both arms had zero invalid answers. Score stayed
85/400 in every row, with every model predicting level 0 on all 400 Score
questions; the gold distribution is 85/107/208 over levels 0/1/2. The
SELECT-BEST comparison is diagnostic and cannot replace the failed fixed498
gate. Both continued packages gained CSS pilot proxy over Eos 1.0 but worsened
typed and Score Brier; their visible result is not a first-release claim.

After both BEST identities were frozen, type temperatures were fitted on
CAL700. The hard BEST temperatures were Choice 1.16457, Noul 0.83939, Score
0.72771; the soft BEST temperatures were 1.05038, 0.72656, 0.63841. Native
CAL inference left accuracy and H unchanged. Hard/soft typed Brier became
.432187/.426772, respectively, versus raw .431740/.418590; Score Brier became
.786170/.786298 versus raw .779405/.773668. CSS pilot task-median full-sum
Brier improved from .970004/.941565 to .927149/.928438. These distinct Brier
conventions must not be mixed. CAL did not repair the Score collapse.

## Reproducibility and resource receipt

- Source model revision as above. Hard fixed498, hard BEST480, soft fixed498,
  and soft BEST448 package SHA-256 fingerprints are respectively
  `571194df1779e1a2277ec984853116bb58304992a9f54464a34d2ca0122f61ce`,
  `bd180952d0b423f37f40a24f9f870900321a7aaa5fbabde485c5d4ced8f6abbf`,
  `4dccf9faf87997b4444ab5b341f9966808fef11b9fb486228c566c5f42ec13d3`,
  and `d96270c24d6b9c5858ada43dfbec5250178faeb4b94a15a224f8ae56d2ba999f`.
- All four packages passed full SELECT700 and frozen32 native parity: zero
  category or input changes, zero probability drift against the frozen
  reference. Same-node hard and Eos 1.0 DEV/CSS outputs reproduced their
  earlier node's canonical per-answer JSON exactly after omitting timing and
  usage metadata. All raw and calibrated panels had zero invalid or
  over-budget answers.
- Typed/CSS prompt SHA-256: `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
  / `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
  Gold SHA-256: `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`
  / `9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`.
  The source mirror was bound to `aef2ee065`; the image digest was
  `sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
- Measured hard/soft optimizer GPU time was 1.076/1.010 GPU-hours. The
  same-node parity, raw/CAL open-panel readout, and Eos 1.0 control occupied
  a task-reserved GPU for 0.620 GPU-hours under its 1.0 GPU-hour cap; the
  reservation was released. Earlier failed smokes and transfer diagnostics
  are accounted separately in their original receipts, not erased.
- The private final receipt, including 41 scored prediction/artifact hashes,
  code hashes, parity manifests, and measured readout time, has SHA-256
  `4cbcb80f956438bf5ca2669e5dc5ee825708abe1253f5f402e0174fcd0e13fb9`.
  Raw predictions, private paths, labels, and training text remain outside
  this source note.

**Next experiment:** a new prospective, group-disjoint Score mechanism and
calibration-data arm is needed to test whether the level-0 collapse is caused
by the present Score mix. It must start from an eligible initializer with a
matched control and new development protocol; this failed KL weight is not
adjusted from the observed DEV outcome.
