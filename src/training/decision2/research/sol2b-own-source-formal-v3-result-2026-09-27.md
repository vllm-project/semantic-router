# Own-Sol 2B: JevArena v3 and public231 result

**Decision: HOLD.** The previously qualified BEST160 unmerged PEFT package did
not pass its frozen first-release gate. Its development proxy had exceeded our
Sol 1.0 control by 2.19 points, but its prospective formal JevArena v3 score
was **1.62 points lower** on the matched panel. No 2B model was uploaded or
published from this arm. This result supersedes the advancement-only decision
in the [development receipt](sol2b-own-source-adapter-package-result-2026-09-27.md),
without changing that receipt's development scores.

## Exact comparison

The candidate directly starts from our `Decision-1.0-Sol-2B` at revision
`0665a41108e8f0b33a9515c98311c45947b99399`, using the existing BEST160
checkpoint and original CAL. The package manifest SHA-256 is
`a917ffe0cc64aa6e551825f0d4458d16deffe56ab0bcb451bd8f91687c953d8a`;
loaded parameters are 1,900,750,144. Its unmerged package matched the original
source on all 3,030 development predictions: zero changed categories or
probability drift. The matched own-1.0 control used its pinned native loader;
the selected same-size open peer was `Mapika/decider-2b` at revision
`533964dae8be954c5b5e19fa4948e48408094c1e`, with its native `system_one`
adapter. The peer's historical Decision Index score was used only to select
the opponent; all scores below are fresh same-panel runs.

JevArena v3 contains 1,600 typed original items, which require 2,000 scored
answers across four independent families, and 6,547 human-labeled transfer
items across 15 tasks. `T` is the four-family macro accuracy, `H` is the task
median macro-F1, and the frozen score is `100 × sqrt(T × H)`. Invalid,
over-budget and missing answers remain in the denominator. The JevBench
public231 subset is separate and does not enter v3.

| Matched model | Typed `T` | Transfer `H` | v3 score | JevBench public231 |
| --- | ---: | ---: | ---: | ---: |
| Decider 2B, pinned public revision | 0.583125 | 0.420180 | **49.4992** | **175/231** |
| Own Sol 1.0, pinned revision | 0.421875 | 0.492462 | 45.5804 | 161/231 |
| DEV2.0-2B BEST160 unmerged package | 0.403125 | 0.479368 | 43.9596 | 162/231 |

Candidate minus own Sol 1.0 v3 score: **−1.6208**, paired 95% CI
**[−2.8790, +0.5372]**. Candidate minus the Decider peer: **−5.5396**,
paired 95% CI **[−12.3095, −2.1503]**. The registered release threshold was
at least +2.0 points over own 1.0 and a positive lower confidence bound;
neither was met. Intervals use 5,000 paired bootstrap draws (seed 20260927),
resampling typed independent groups within family and transfer tasks/items.
The comparison is **prospective post-key, same-panel evidence**: v3 labels had
been accessed in unrelated prior work, so it is not a never-opened blind test.

The candidate had 917/2,000 valid typed answers correct versus 961 for own
Sol 1.0. By family, the candidate improved constraint competition
(0.185 vs 0.145), but declined on evidence connection (0.680 vs 0.715),
exception handling (0.405 vs 0.4425) and resource/Score decisions
(0.3425 vs 0.385). Its transfer micro accuracy increased (0.5062 vs 0.5007)
while the task-median macro-F1 declined (0.4794 vs 0.4925). Its typed Brier
score was worse (0.3534 vs 0.3453), while typed ECE was slightly lower
(0.2406 vs 0.2548). These are real tradeoffs, not a broad improvement.
On public231, candidate/own/peer tier-correct counts were easy 48/48/48,
standard 67/66/64 of 72, and hard 47/47/63 of 111. The candidate's +1
public item is not a material benchmark advance. This is a public-only
JevBench replication, not an upstream closed-set rank.

## Freeze, input limits and reproducibility

The candidate lock was created before this arm's formal predictions (SHA-256
`0c9232fc069e04d94b9b219abf9036087b3b737f05ca9b2e7b4d2a30a198cfb2`).
Frozen gold-free prompt SHA-256 values: typed
`e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`,
transfer `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`,
and public231 `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`.
All nine complete model × panel prediction files were sealed before scoring.
The corrected gold-free seal SHA-256 is
`106a83bbda1b3d3d5626e1864f2a4fb85b95546586379613ab5ff7baf2ff1a7c`;
the later digest-only score-input lock SHA-256 is
`cc1d12afa138f4a07010169394a93f29263412273e9ab7413c142fbec577269f`.
The private aggregate result receipt SHA-256 is
`bd61f6c2d6398e759d615c7c8a6bd05d871010cfa13d708afd676a9f08e25e79`.
Raw predictions, labels and paths stay private.

The package's native input limit produced 18 over-budget transfer answers;
all are invalid, with zero truncation. Own Sol 1.0's native 16,384-token
limit produced four over-budget transfer items. Its collector stopped at the
first such item. A gold-free tokenizer audit identified exactly four, then
the unchanged native collector ran the remaining in-budget inputs. A
separately hashed stitch retained every original native answer and inserted
four explicit invalid answers, restoring the full 6,547-row denominator.
No prompt, gold, probability, model or calibration was changed.

The first seal implementation failed before any scoring because it sought
`adapter_version` in every package prediction row; the package collector
records that field in its signed per-panel manifest. The frozen checker was
left unchanged. A separately hashed corrected seal validates that manifest,
every row's model/input identity, the own-1.0 over-budget continuation and
the original frozen code hashes. Its source SHA-256 is
`07927842302b79ff354f9c593088e4814fbabc8742bd4eaf87ad6bff7f682816`;
the over-budget helper source SHA-256 is
`52d166706935558b381e659f9026bc4b3be3a7e103cab14c163edab25fc20684`.

All actual inference used the pinned one-GPU runtime image and original native
adapters; component scoring and the paired comparison used the frozen scorer.
Docker start/die events account for 546 candidate GPU-seconds, 375 peer
GPU-seconds, and 311 own-control GPU-seconds: **0.3422 GPU-hour** total for
this formal panel, excluding CPU seal/scoring. The existing development work
consumed approximately 0.17 GPU-hour. The independent local
`make test-training-contracts` check passed after the gold-free repair code.

## Next discriminating experiment

Retain this exact model/package and the failed formal result as a control.
On our own Sol 1.0 source, compare a **balanced typed/transfer training mix
with soft-distribution replay** against the existing targeted3024 arm under
equal data-access, token and step budgets. Preregister the training sources,
language and Score balance, replay loss, selection rule and stop gate before
starting GPUs. Use TRAIN-disjoint SELECT/CAL, typed DEV and CSS pilot to
choose; then seek genuinely new, source-disjoint confirmation because the
present v3 key has been accessed. Do not choose a checkpoint by re-scoring
this revealed v3 panel or transfer the BEST160 score to a new weight pack.
