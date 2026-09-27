# Decision 2.0 4B: JevArena v3 first-release result

**Decision under the policy frozen before labels: HOLD.** The candidate improves
the paired aggregate score, but its Score accuracy falls 2.75 percentage points
below Decision 1.0 Nox. The predeclared maximum drop was 2.00 points. This
record preserves that negative decision; it is not a model card or a release
approval. A separate [post-key tradeoff review](decision2-4b-v3-postkey-tradeoff-review-2026-09-27.md)
addresses the user's later preference without changing this result.

## Frozen comparison

The sealed JevArena v3 first-release panel contains 1,600 typed items in 400
independent four-variant groups and 6,547 human-labeled records across 15
transfer tasks. Each model answered the same prompts through its pinned native
adapter. The typed items contain **2,000 scored answers** because one family
asks two questions per item. The headline is `100 × sqrt(T × H)`, where `T` is
the four-family macro accuracy and `H` is the median of 15 task macro-F1 scores.
Invalid and missing answers count as wrong. The public JevBench subset is a
separate 231-item cross-check and does not enter the headline.

| Model | Loaded parameters | T | H | JevArena v3 | Public JevBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llm-semantic-router/DEV2.0-4B` checkpoint 0232 | 4,205,751,296 | 0.750000 | 0.523704 | 62.6720 | 195/231 |
| `llm-semantic-router/Decision-1.0-Nox-4B` | 4,208,383,488 | 0.614375 | 0.519046 | 56.4702 | 173/231 |
| `jaredpalmer/kev-4b` (open control) | 4,207,062,528 | 0.719688 | 0.487400 | 59.2263 | 175/231 |

The measured Nox/candidate parameter ratio is 1.000626, within the frozen
1.25 same-size limit. The paired 4B-minus-Nox headline difference is
**+6.2018 points**, with a 95% interval of **[+2.3882, +10.7653]** from exactly
5,000 paired bootstrap replicates, seed `20260927`. Typed independent groups
were sampled within family; transfer tasks and then records within task were
sampled in paired draws. `T` differs by +0.135625, interval
[+0.105000, +0.165625]. `H` differs by +0.004658, interval
[-0.056113, +0.080811]; the transfer axis alone has no established positive
effect at this confidence level.

### Typed task families and release guardrail

| Type | Decision 2.0 4B | Nox 4B | Difference | Predeclared limit |
| --- | ---: | ---: | ---: | ---: |
| Choice | 778/800 · 97.25% | 552/800 · 69.00% | +28.25 pp | At least -2.00 pp |
| Noul | 655/800 · 81.875% | 653/800 · 81.625% | +0.25 pp | At least -2.00 pp |
| **Score** | **167/400 · 41.75%** | **178/400 · 44.50%** | **-2.75 pp** | **At least -2.00 pp: FAIL** |

The Score miss is 11 answers; the frozen limit permits at most 8 additional
misses out of 400. The typed four-family macro values are not the raw
2,000-answer micro accuracies (80.00% candidate and 69.15% Nox).

### Human-labeled transfer tasks

These are macro-F1 scores including invalid and missing answers as misses.
The candidate wins 7 tasks and loses 8; the median shifts only slightly.

| Task | Decision 2.0 4B | Nox 4B | Difference |
| --- | ---: | ---: | ---: |
| `conv_go_awry` | 0.499248 | 0.455796 | +0.043452 |
| `emotion` | 0.594317 | 0.497068 | +0.097249 |
| `flute` | 0.833337 | 0.580677 | +0.252660 |
| `ibc` | 0.428542 | 0.542695 | -0.114153 |
| `indian_english_dialect` | 0.455462 | 0.307972 | +0.147489 |
| `media_ideology` | 0.439446 | 0.442811 | -0.003365 |
| `mrf` | 0.755456 | 0.760781 | -0.005325 |
| `persuasion` | 0.511413 | 0.530598 | -0.019185 |
| `raop` | 0.593080 | 0.575593 | +0.017487 |
| `reddit_humor` | 0.582944 | 0.462366 | +0.120579 |
| `talklife` | 0.259275 | 0.296482 | -0.037207 |
| `tempowic` | 0.523704 | 0.621274 | -0.097570 |
| `tropes` | 0.032164 | 0.066959 | -0.034795 |
| `wiki_corpus` | 0.535846 | 0.519046 | +0.016800 |
| `wiki_politeness` | 0.567260 | 0.569592 | -0.002332 |

The largest task losses are `ibc` and `tempowic`. The long-input `tropes`
task has 114 records, four explicit native over-budget invalid answers for
each model, and very low macro-F1 for both. The over-budget handling was
recorded before labels were opened and those records remained in the
denominator.

### Reliability and public cross-check

| Measure | Decision 2.0 4B | Nox 4B | Frozen rule or interpretation |
| --- | ---: | ---: | --- |
| Typed invalid/missing | 0/2,000 answers | 0/2,000 | Equal, below the 2% floor |
| Transfer invalid/missing | 4/6,547 | 4/6,547 | Equal, below the 2% floor |
| Typed probability coverage | 2,000/2,000 answers | 2,000/2,000 | Complete |
| Typed Brier | 0.130543 | 0.204949 | Lower is better |
| Coverage-adjusted Brier using actual 2,000-answer denominator | 0.130543 | 0.204949 | Candidate is 0.074406 lower; within the +0.03 limit |
| JevBench easy | 48/48 | 48/48 | Public subset only |
| JevBench standard | 69/72 | 66/72 | Public subset only |
| JevBench hard | 78/111 | 59/111 | Public subset only |

The public 231-item ranking here covers only these three frozen models; it is
not an official rank on JevBench's inaccessible sealed questions. No global
Pareto claim follows from a three-model panel.

## Chronology and reproducibility bindings

The signed candidate lock was recorded at 05:45:43 UTC on 2026-09-27. All
nine model-by-panel prediction digests were sealed by 06:22:16; the raw
prediction list was sealed at 06:26:33, the gold-free audit at 06:27:04, and
the pre-key receipt at 06:33:02. First label access was logged at 06:33:50.
The score reports were written after the pre-key receipt. The timestamp log,
receipt chain, score-to-prediction digests, typed/CSS gold digests, model IDs,
adapter IDs and frozen source hashes were checked read-only. The log is a
hash-bound declaration corroborated by file times, not an external trusted
timestamp service.

| Evidence | SHA-256 |
| --- | --- |
| Frozen plan | `cd579fbe2aec27053f10e0185d5d8360472b0a5bef0504f9ca96aeee55b49b19` |
| Candidate lock | `5e7c70f8963d36671fccc85b0abdd8c924eefe1ac0896fc397e069cce66f1535` |
| Comparator roster | `3398514d46d4fdf834e89d894cbcdf8f7c5faa216b0bfb8c9cd7dd645171277d` |
| Numeric policy | `b0d7d432d673e9600400d859146304e6c058763a24816d310d68bd817d36aaf4` |
| Raw prediction hash list | `8aa72bbe34cdcbb9d40e2626c5e2c3f4a92d79c1e68c81f98e5ac5e2a8ad504c` |
| Gold-free audit | `1e4e78ad9750e3cf92f05089afe284c918be082eafb4ace424f9515e94d89612` |
| Pre-key chronology | `d1d11957a75e865210ad29af8d2cf87c5add9c5e995596ec197290928a1a0427` |
| Pre-key receipt | `40eeb4a7300ae36e27c6a01fded8471f6be7410145bb48716d482208aa44adab` |
| Timestamp log | `472c4656125b4775f1addf1bcaf13f5167044e4e45f605e5ae93f8cbb19dd1d6` |
| Typed gold | `707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e` |
| Transfer gold | `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4` |
| Candidate typed/CSS/public reports | `9c29887927c09cbac1237a2593a8bf001be80094a1d563d04ca56ce4c45e9325` / `67d28baa68ad0297ed36ecb445d8b7546244e0eb8f430e1c17da16bbe2803d25` / `811321502282f6ffbba4406852064e2ab5cd10b818ddf530121524147c571dae` |
| Nox typed/CSS/public reports | `7cdeef81750756de7f858c08d73443a72f0b77ae50c423d896ee2df462ccbc62` / `a73d620fd5ee191c93d4595f108d5e39d668d846dcf0b6aff3d2fe8ecd3e039b` / `790aeb2c3dfab3f852036cbb7858493e2be16a958957858ff7f2f988b35bdd0a` |
| Kev typed/CSS/public reports | `eaf23c1ac6060b8e2dbdd486f6771bf349b0b406ede20ef3c37193cd93e9e2c4` / `9574a9186f467ae1bb782ad4eb9d271cad1af2a1ec0dc198954ef36c2f6c398e` / `0a0be0cb513d807bc5aa980dca835f0f323d1b13fe3f2fe2d6d61d7b7d4ffb9a` |
| 4B-vs-Nox paired report | `d2757956e784bc5c27615fcf787c6bdf6fc33ca9cf3d5ba0db0c2107d5396d23` |

The candidate's frozen model, native adapter and calibration digests are,
respectively, `7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`,
`8be9f2006b881a41cf733f87d9364fd9e00cb01b5d34951afe7ac636e8f53e1d`,
and `6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`.
Its typed, transfer and public native manifest SHA-256 values are
`cb04d20df4f3ed7dd1f5e336fce20bfdedd05132ebb3d78901f566b8353ad1e1`,
`8e8bec051fd823fe0a3acf56861bb9a535bc2f4e251b3c39b69c5a586dde8b25`,
and `bec1eb920e87022d2ab92fea5fbf9909c7a44c7e875b807bd9870279502807b4`.
All three native manifests match the frozen model ID, revision, prediction
digest and adapter digest. The paired report's scorer-source hashes, gold
digests and two models' prediction digests match the pre-key bindings.

## Implementation defect distinct from the negative outcome

The frozen typed scorer correctly reports `items=1600` and
`overall.n=2000`: one typed family has two scored questions per item. The
current v3 ranker incorrectly requires `overall.n==1600`, so it rejects every
complete typed report. The current publication numeric gate also rejects
`probability_n=2000` because it assumes an upper bound of 1,600; its stated
coverage-adjusted Brier denominator is likewise 1,600. Applying that formula
to 2,000 complete probability answers would be nonsensical. The table above
shows the sensitivity result using the actual 2,000-answer denominator; it is
not a retroactive change to the frozen release gate. The ranking and package
gate remain blocked pending a separately reviewed protocol erratum. The Score
regression independently requires **HOLD** even after such a correction.

The appropriate next experiment is a new, preregistered Score-focused 4B
candidate under the same-size control, preserving Choice and Noul gains and
testing whether the Score loss can be removed. These sealed results must not
be used to choose among checkpoints or alter the original threshold.
