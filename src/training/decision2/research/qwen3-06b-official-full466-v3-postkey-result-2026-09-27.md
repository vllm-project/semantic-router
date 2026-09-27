# Official Qwen3 0.6B BEST466: post-key v3 same-panel result

**Status:** full gold-free formal prediction and same-panel scoring completed;
the package and public release gates remain separate. This is a prospective
comparison after project-level access to the formal answer keys, not an
original blind-test claim. The selected checkpoint and calibration were locked
before the candidate predictions. The prior-generation and open-peer
predictions were reused only after their full-panel seals and source hashes
matched the new roster. No candidate score was read before all three new
prediction files were sealed.

## Frozen identity and chronology

| Item | Frozen value |
| --- | --- |
| Initialization | Official `Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Selected full checkpoint | Step 466; 597,103,104 loaded parameters; fingerprint `5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab` |
| CAL | 700 disjoint records; fit SHA-256 `dc29fc12c65f2cf0a676f547360703fde5e846b1df0cd4fb557f830fb3d13da5` |
| Native adapter | `decision2-typed-benchmark-adapter-v2-calibrated`; source SHA-256 `33dae46ec1aa812f0a75365e2757a60f17b8d9474fdd15d9a45524437c1b7d56`; max length 8,192, no truncation |
| Roster | SHA-256 `21ed1cfdf78a744c6bed62ca42c3a67d02376bc4d1360b56bba84e7cd9fc75c2` |
| Pre-inference candidate lock | SHA-256 `e3b2c7876d71eeaf2b656cf723f5fd346a9b4e4aeb9ef53b5b5819bb27f9f17a`; created 11:50:02 UTC |
| Complete prediction seal | SHA-256 `3bd2e8e2ee992ef170bfccbf801d142cdecddad317e7186bd6bc0c9cb46ec33c`; created 11:56:47 UTC |

The typed, human-transfer and public prompt SHA-256 values were respectively
`e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`,
`7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`,
and `642d3fac1b6521fe33df72f9228e4e364b7be7ea277893207f97da5bc75ddd`.
The TRAIN-to-formal-panel overlap audit reported zero same-record, exact-state,
normalized-state or near-text collisions for all three panels; its receipt
SHA-256 was `2501199805ac94397f15776224a2d7c5f6befffbea9ac19218dce7d968826493`.
This audit does not establish absence of semantic or pretraining overlap.

Candidate gold-free prediction SHA-256 values were `e7380958e5273ad89eaf2c2b80cf8812111d88ac351de6bf1bfa06cfaf718552`
(typed), `ef521866fc50d8fb1e2f21705b44141a0fac04fe0e11595b68d872f3e3e512ef`
(human transfer), and `2cf2ab0a2ba097ecb71ea02124d0674fc7d11e64fc29b89d41a09a40390b0244`
(public). The sealed candidate outputs contain 1,600 typed original items /
2,000 answer slots, 6,547 human-transfer items, and 231 public items. Typed and
public inference had zero invalid or over-budget answers. Human transfer had
15 over-budget answers; all 15 count as failures in the full denominator.

The single GPU reservation spanned 11:51:18–11:57:01 UTC, at most **0.0953
GPU-hours** including audit and container startup gaps. GPU reservation was
released after the prediction seal. The runtime image identity was
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

## JevArena v3 same-panel scores

`T` is the four-family macro accuracy on typed FINAL and `H` is the median
macro-F1 over 15 human-transfer tasks. The frozen scalar is
`100 × sqrt(T × H)`; invalid and absent answers stay in the denominators.

| Model | Loaded parameters | T | H | v3 scalar | Typed valid | Human-transfer valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official-Qwen 0.6B BEST466 | 597,103,104 | 0.309063 | 0.480094 | **38.5200** | 2,000/2,000 | 6,532/6,547 |
| Decision-1.0-Kai-0.6B | 571,909,635 | 0.361875 | 0.356907 | 35.9383 | 2,000/2,000 | See sealed control report |
| Bosun-v3.1-0.6B | 606,131,200 | 0.433750 | 0.342161 | 38.5243 | 2,000/2,000 | See sealed peer report |

The candidate is **+2.5817** points above Kai1 by point estimate and
**-0.0043** below the near-size Index-selected Bosun peer. The paired
5,000-replicate bootstrap (seed `20260927`; independent typed groups and
resampled human-transfer tasks/items) gives 95% intervals of
**[-2.0317, +7.8464]** for candidate minus Kai1 and **[-6.3498, +2.2105]**
for candidate minus Bosun. Neither interval excludes zero. The corresponding
private paired report SHA-256 values are
`ff365136819ad13a20e0d17dd7e9104dbc71bad134b0c773829fab7b86094a6d`
and `200ce03bf4e87ed10d9701a427f2341ed27676b3a9ebc9419773d7c95b399876`.

The aggregate masks a large capability tradeoff:

| Typed FINAL type | Candidate | Kai1 | Bosun |
| --- | ---: | ---: | ---: |
| Choice | 109/800 (13.63%) | 277/800 (34.63%) | 365/800 (45.63%) |
| Noul | 458/800 (57.25%) | 404/800 (50.50%) | 541/800 (67.63%) |
| Score | 80/400 (20.00%) | 98/400 (24.50%) | 83/400 (20.75%) |

At the four typed family level, the candidate scores 0.0550 on constraint
competition, 0.3813 on evidence join, 0.6000 on exception stack, and 0.2000
on resource ledger. The same Kai1 values are 0.1475, 0.5000, 0.5550, and
0.2450. The candidate's human-transfer median improves, but typed Choice and
Score regress. Its typed Brier is 0.3535, ECE10 is 0.1785, and Score MAE is
1.1828; calibration transport concerns from development remain relevant.

## Public JevBench 231, separately reported

This is the reproducible public subset, not the upstream closed leaderboard.
The public scorer required the native companion manifest because this
collector places model identity in the manifest rather than each row; a first
scorer call without that manifest failed before writing a report, and the
manifest-bound call passed. No predictions were changed.

| Model | Correct / 231 | Valid / 231 | Easy | Standard | Hard | Tier-macro accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official-Qwen 0.6B BEST466 | **143** | 231 | 48/48 | 51/72 | 44/111 | **0.70158** |
| Decision-1.0-Kai-0.6B | 114 | 187 | 45/48 | 39/72 | 30/111 | 0.58315 |
| Bosun-v3.1-0.6B | 133 | 231 | 47/48 | 50/72 | 36/111 | 0.66598 |

The candidate public score report SHA-256 is
`17075505c67717fa3efe78760995ce53d994fa52c2cd183d0621b7025196289f`.
Candidate typed and human-transfer score report SHA-256 values are
`1cdb39ee11ee027e49233093b08c2b09b5e09684a421f3ed94c228f802c2229b`
and `80e7e66fe529b610defdc5c50f28965f112a21ec7ee431f1b43a360c066d17f4`.

## Decision and next discriminating experiment

The selected official-base model clears the earlier **development** gate and
has a higher post-key same-panel v3 point estimate than Kai1, stronger human
transfer, and a stronger public JevBench score. The v3 paired interval does
not demonstrate a certain positive aggregate gain, while Choice falls by
21.0 percentage points and Score by 4.5 points. Describe any first release
with those limits; do not say it established a blind or statistically certain
overall win. A user-facing release still needs a clean model package,
round-trip download and native-output parity on the exact downloadable
revision, and a product model card. These were outside this comparison.

For the next optimization arm, freeze a same-base, matched-token targeted
Choice/Score curriculum with soft replay, then screen on independent
development items before another formal prediction lock. Its key question is
whether constraint/evidence and ordinal decisions recover without giving up
the observed human-transfer gain. Do not select that arm on these already
keyed formal labels.
