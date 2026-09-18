---
title: Balance development and validation — September 18, 2026
description: Two Balance development revisions followed by frozen 136-case validation, with a separate 124-case non-GPQA scope, measured usage, simulated costs, and explicit quality limitations.
---

# Balance development and validation — September 18, 2026

**The development match did not generalize to the frozen validation.** Balance
R2 matched the strongest observed single, Flash, on 25 reused development
cases. On the subsequent 136-case validation, R2 scored **31.00% weighted
accuracy versus GLM's 38.89%**, while reducing simulated subject cost by
77.16% (76.88% on a cache-neutral basis). Excluding the 12 prior-label-seen
GPQA retests, the separate 124-case scope also retained a quality deficit:
25.67% for R2 versus 33.33% for GLM.

The goal of matching or exceeding the strongest single model **has not been
met**. These results demonstrate a measured quality/cost tradeoff under the
frozen protocol, not capability equivalence, billed production savings, or a
complete sr-bench 1.0 score. The development results and reusable R2 recipe
remain below so the full two-round loop can be inspected.

Requests, response usage, and latency were measured from real executions.
**Prices were a size-based simulation for this deployment, not provider billing
or measured GPU cost.** Every monetary value below applies those frozen prices
to recorded usage. See the [sr-bench guide](./sr-bench) for the reusable
dataset, CLI, Dashboard, and reporting workflow.

## Frozen validation: the development match did not generalize

R2 was frozen before this validation and was not changed after observing its
outcomes. The three single models each generated once on 136 cases; the same
frozen R2 then generated once on the identical cases. Both jobs completed: **544
subject calls and 39 separate Flash judge calls, 583 calls in total**. There
were no application generation retries and no resumed development attempt.

The dataset is a custom subset of the standard profile. Its 12 GPQA cases
remain prior-label-seen retests; they are not unseen validation. The other 124
cases were disjoint from the development and previously dispatched cases
within the recorded preparation audit. That narrower held-out scope is shown
separately; it is not a claim that benchmark content was absent from model
pretraining or all previous exposure.

| Benchmark | Cases in full validation | Weight in 136-case scope | Cases in non-GPQA scope | Weight in 124-case scope |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro | 100 | 2/9 | 100 | 1/3 |
| SimpleQA Verified | 20 | 2/9 | 20 | 1/3 |
| GPQA Diamond | 12 | 1/3 | 0 | Excluded |
| ARC-AGI-2 | 4 | 2/9 | 4 | 1/3 |
| Total | 136 | 1 | 124 | 1 |

The validation retained the 4,096-token ceiling, concurrency four, 600-second
request deadline, 45-second idle deadline, 1,300-second case deadline, native
request profiles and prices used below. Each job had a 14,400-second run bound
and an $8 simulated-price dispatch budget. The latter is a reservation/stop
policy, not a guaranteed provider invoice cap. Final-channel-only scoring,
fixed denominators and output-limit failures were unchanged.

### Quality and the primary comparator

The **predeclared comparison rule** chooses the strongest observed single by
exact weighted benchmark accuracy. On these validation cases that single is
**GLM**, with `7/18`, not Flash, whose micro accuracy is higher. Weighted and
micro accuracy answer different questions because benchmark sample counts
differ. The same rule also selects GLM in the 124-case scope; there is no tie.

**All 136 cases, including GPQA retests**

| Target | Correct / planned | Weighted accuracy | Micro accuracy | Micro Wilson 95% interval | Output-limit cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM | 82/136 | 38.89% | 60.29% | 51.90% to 68.13% | 25 |
| Flash | 87/136 | 35.67% | 63.97% | 55.62% to 71.55% | 40 |
| Qwen27 | 71/136 | 29.44% | 52.21% | 43.87% to 60.42% | 52 |
| Balance R2 | 74/136 | 31.00% | 54.41% | 46.03% to 62.55% | 50 |

**The 124-case scope excluding GPQA**

| Target | Correct / planned | Weighted accuracy | Micro accuracy | Micro Wilson 95% interval | Output-limit cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM | 76/124 | 33.33% | 61.29% | 52.50% to 69.40% | 21 |
| Flash | 82/124 | 32.67% | 66.13% | 57.43% to 73.86% | 34 |
| Qwen27 | 66/124 | 23.33% | 53.23% | 44.48% to 61.78% | 46 |
| Balance R2 | 69/124 | 25.67% | 55.65% | 46.86% to 64.09% | 44 |

R2 therefore **did not meet the goal of matching or exceeding the strongest
single model**: its weighted deficit is 7.89 percentage points over 136 cases
and 7.67 points over the 124 non-GPQA cases. Completing the jobs and reducing
cost do not qualify that quality goal. The reported Wilson intervals apply
to ordinary micro accuracy, not to weighted accuracy or equivalence.

Per-benchmark correct counts retain every planned case. Parentheses show the
number stopped at the fixed output limit; these are included as incorrect,
not removed from the denominator.

| Target | MMLU-Pro | SimpleQA | GPQA retest | ARC-AGI-2 |
| --- | ---: | ---: | ---: | ---: |
| GLM | 70/100 (11 capped) | 6/20 (6 capped) | 6/12 (4 capped) | 0/4 (4 capped) |
| Flash | 78/100 (19 capped) | 4/20 (11 capped) | 5/12 (6 capped) | 0/4 (4 capped) |
| Qwen27 | 65/100 (30 capped) | 1/20 (12 capped) | 5/12 (6 capped) | 0/4 (4 capped) |
| Balance R2 | 67/100 (28 capped) | 2/20 (12 capped) | 5/12 (6 capped) | 0/4 (4 capped) |

All 16 ARC subject responses in this validation reached the output ceiling.
Those zeros are outcomes of this 4K protocol; they do not establish
unconstrained puzzle-solving ability. In particular, a scope with four ARC
cases carrying 2/9 or 1/3 of the weight has substantial uncertainty.

### Subject cost, judge cost and the cache counterfactual

All amounts remain **USD-equivalent at the frozen size-based simulated
prices**, calculated from real returned usage. Subject cost excludes the
Flash judge. No validation receipt required accounting correction: original
receipts, independent four-bucket recomputation and application reports agreed.
The earlier development corrections are retained separately below.

**136-case cost scope**

| Target | Subject cost | Judge calls | Judge cost | Total cost | Cache-neutral subject cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM | 0.67338720 | 14 | 0.00809705 | 0.68148425 | 0.67338720 |
| Flash | 0.50208535 | 9 | 0.00366470 | 0.50575005 | 0.49877295 |
| Qwen27 | 0.08846550 | 8 | 0.00300495 | 0.09147045 | 0.08793630 |
| Balance R2 | 0.15381303 | 8 | 0.00355810 | 0.15737113 | 0.15571815 |

**124-case cost scope, excluding GPQA**

| Target | Subject cost | Judge calls | Judge cost | Total cost | Cache-neutral subject cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM | 0.58210560 | 14 | 0.00809705 | 0.59020265 | 0.58210560 |
| Flash | 0.44099640 | 9 | 0.00366470 | 0.44466110 | 0.43768400 |
| Qwen27 | 0.07972330 | 8 | 0.00300495 | 0.08272825 | 0.07919410 |
| Balance R2 | 0.14503903 | 8 | 0.00355810 | 0.14859713 | 0.14694415 |

The 136-case single-model matrix cost 1.26393805 in subject calls plus
0.01476670 for judging. R2 cost 0.15381303 plus 0.00355810. Together these
validation jobs cost **1.43607588 USD-equivalent**. The 124-case tables are
subsets of those same calls, not additional runs or additional expense.

R2 used cache reads where earlier runs had written cache entries. Its
cache-neutral counterfactual prices every prompt token as fresh input, plus
output at the same model rate; it is neither billed spend nor a measured
cache-free run. Keeping both views avoids attributing every cache discount to
routing quality. R2 is cheaper than GLM and Flash but more expensive than
Qwen27 in both scopes.

### Paired comparisons against all three singles

The primary comparison is R2 versus GLM; the other two comparisons remain
visible. Differences and intervals below are percentage points of **weighted**
accuracy. The conservative paired Hoeffding interval uses the frozen weights
and independent-case assumption described in the development analysis.
Bootstrap intervals are diagnostics only. Both exclude strongest-single
selection, source contamination and tuning-selection uncertainty. An interval
covering zero does not establish equivalence or non-inferiority.

**136 paired cases**

| R2 versus | Wins / losses / ties | Weighted change, pp | Conservative 95%, pp | Bootstrap diagnostic 95%, pp | Subject saving | Cache-neutral saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM (primary) | 9 / 17 / 110 | -7.89 | -50.46 to +34.69 | -18.33 to +3.11 | 77.16% | 76.88% |
| Flash | 1 / 14 / 121 | -4.67 | -47.24 to +37.91 | -9.33 to -0.44 | 69.37% | 68.78% |
| Qwen27 | 6 / 3 / 127 | +1.56 | -41.02 to +44.13 | -0.44 to +4.44 | -73.87% | -77.08% |

**124 paired non-GPQA cases**

| R2 versus | Wins / losses / ties | Weighted change, pp | Conservative 95%, pp | Bootstrap diagnostic 95%, pp | Subject saving | Cache-neutral saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM (primary) | 8 / 15 / 101 | -7.67 | -58.08 to +42.74 | -15.33 to +0.00 | 75.08% | 74.76% |
| Flash | 1 / 14 / 109 | -7.00 | -57.41 to +43.41 | -13.33 to -0.33 | 67.11% | 66.43% |
| Qwen27 | 6 / 3 / 115 | +2.33 | -48.08 to +52.74 | -0.67 to +6.33 | -81.93% | -85.55% |

### Measured latency, wall time and routing distribution

Subject request latency excludes the separate judge call. TTFT is the time
to the first nonempty streamed final or reasoning token. A sum of concurrent
request durations is not elapsed job time.

| Target / 136 cases | Subject p50, s | Subject p95, s | TTFT p50, s | Subject request-time sum, s |
| --- | ---: | ---: | ---: | ---: |
| GLM | 34.73 | 298.56 | 0.231 | 12524.92 |
| Flash | 7.96 | 40.51 | 0.139 | 2344.52 |
| Qwen27 | 19.22 | 79.30 | 0.079 | 4977.27 |
| Balance R2 | 17.75 | 90.37 | 0.138 | 5257.09 |

The single-model matrix took **5,066.215 seconds (84m26s)** for 408 subject
cells and 31 judge calls; R2 took **1,370.506 seconds (22m51s)** for 136 subject
cells and eight judge calls. These different job sizes are not a throughput
speedup comparison. The interleaved matrix provides no separate wall time for
each single model, nor was a standalone wall time measured for the 124-case
subset. R2's p95 request latency was worse than both Flash and Qwen27.

A build lasting **49.85 seconds**, from 18:51:22.931 to 18:52:12.777 UTC,
overlapped late single-model validation on a shared host. CPU, disk or network
contention was possible. No causal slowdown was established, and no post hoc
latency correction was applied. This is a limitation on timing comparisons,
not evidence that the build changed model quality or token accounting.

| Benchmark | R2 Qwen27 calls | R2 Flash calls | R2 GLM calls |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 86 | 14 | 0 |
| SimpleQA | 16 | 4 | 0 |
| GPQA retest | 12 | 0 | 0 |
| ARC-AGI-2 | 4 | 0 | 0 |
| Total | 118 | 18 | 0 |

R2's recorded decisions were `medium` 96, `factual_guard` 18, `reasoning`
15 and `simple` seven. Excluding GPQA, selections were Qwen27 106 and Flash
18; decisions were `medium` 85, `factual_guard` 18, `reasoning` 14 and `simple`
seven. These distributions describe executed requests, not a preview or a
per-question oracle. No validation result was used to change the frozen R2.

<details>
<summary>Exclusive token buckets, per-bucket costs and 124-case timing</summary>

Fresh input, cache read, cache write and output are disjoint buckets. The
judge rows contain only Flash judge calls. Each cost bucket uses the selected
model's frozen rate; the Balance rows sum those per-call model costs.

**136 cases: tokens**

| Target / role | Calls | Fresh input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM / subject | 136 | 45,900 | 0 | 0 | 171,752 |
| GLM / judge | 14 | 4,255 | 0 | 0 | 2,734 |
| Flash / subject | 136 | 32,455 | 0 | 20,384 | 238,168 |
| Flash / judge | 9 | 1,936 | 0 | 0 | 1,234 |
| Qwen27 / subject | 136 | 31,671 | 0 | 21,168 | 275,508 |
| Qwen27 / judge | 8 | 1,644 | 0 | 0 | 993 |
| Balance R2 / subject | 136 | 31,671 | 21,168 | 0 | 262,929 |
| Balance R2 / judge | 8 | 1,712 | 0 | 0 | 1,254 |

**136 cases: simulated cost by token bucket**

| Target / role | Fresh input cost | Cache-read cost | Cache-write cost | Output cost | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM / subject | 0.05508000 | 0.00000000 | 0.00000000 | 0.61830720 | 0.67338720 |
| GLM / judge | 0.00276575 | 0.00000000 | 0.00000000 | 0.00533130 | 0.00809705 |
| Flash / subject | 0.02109575 | 0.00000000 | 0.01656200 | 0.46442760 | 0.50208535 |
| Flash / judge | 0.00125840 | 0.00000000 | 0.00000000 | 0.00240630 | 0.00366470 |
| Qwen27 / subject | 0.00316710 | 0.00000000 | 0.00264600 | 0.08265240 | 0.08846550 |
| Qwen27 / judge | 0.00106860 | 0.00000000 | 0.00000000 | 0.00193635 | 0.00300495 |
| Balance R2 / subject | 0.00616350 | 0.00021168 | 0.00000000 | 0.14743785 | 0.15381303 |
| Balance R2 / judge | 0.00111280 | 0.00000000 | 0.00000000 | 0.00244530 | 0.00355810 |

**124 non-GPQA cases: tokens**

| Target / role | Calls | Fresh input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM / subject | 124 | 43,332 | 0 | 0 | 147,252 |
| GLM / judge | 14 | 4,255 | 0 | 0 | 2,734 |
| Flash / subject | 124 | 29,321 | 0 | 20,384 | 207,885 |
| Flash / judge | 9 | 1,936 | 0 | 0 | 1,234 |
| Qwen27 / subject | 124 | 28,537 | 0 | 21,168 | 247,412 |
| Qwen27 / judge | 8 | 1,644 | 0 | 0 | 993 |
| Balance R2 / subject | 124 | 28,537 | 21,168 | 0 | 234,727 |
| Balance R2 / judge | 8 | 1,712 | 0 | 0 | 1,254 |

**124 non-GPQA cases: simulated cost by token bucket**

| Target / role | Fresh input cost | Cache-read cost | Cache-write cost | Output cost | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM / subject | 0.05199840 | 0.00000000 | 0.00000000 | 0.53010720 | 0.58210560 |
| GLM / judge | 0.00276575 | 0.00000000 | 0.00000000 | 0.00533130 | 0.00809705 |
| Flash / subject | 0.01905865 | 0.00000000 | 0.01656200 | 0.40537575 | 0.44099640 |
| Flash / judge | 0.00125840 | 0.00000000 | 0.00000000 | 0.00240630 | 0.00366470 |
| Qwen27 / subject | 0.00285370 | 0.00000000 | 0.00264600 | 0.07422360 | 0.07972330 |
| Qwen27 / judge | 0.00106860 | 0.00000000 | 0.00000000 | 0.00193635 | 0.00300495 |
| Balance R2 / subject | 0.00585010 | 0.00021168 | 0.00000000 | 0.13897725 | 0.14503903 |
| Balance R2 / judge | 0.00111280 | 0.00000000 | 0.00000000 | 0.00244530 | 0.00355810 |

**Request timing within the 124-case scope**

| Target | Subject p50, s | Subject p95, s | TTFT p50, s | Subject request-time sum, s |
| --- | ---: | ---: | ---: | ---: |
| GLM | 31.76 | 298.52 | 0.230 | 10743.69 |
| Flash | 7.67 | 40.58 | 0.141 | 2047.71 |
| Qwen27 | 19.22 | 79.38 | 0.080 | 4491.59 |
| Balance R2 | 16.50 | 88.03 | 0.138 | 4615.58 |

</details>

### Validation evidence and limits

Independent read-only audits checked **7,906 assertions over all 439 calls in
the single-model matrix**, then **2,737 assertions over all 144 R2 calls**. All
passed.
Each terminal report was read once for this audit; scoring and accounting
were recomputed from retained streams and journal records. Checks covered
unique planned cells, final-versus-reasoning separation, strict answer grading,
truncation, Flash judge identity and payload, native client request parameters,
actual selected models, per-response R2 hash acknowledgements, exclusive
usage buckets, frozen prices, fixed denominators, Wilson intervals and paired
statistics. They issued no model requests and changed no original evidence.

The audit does not human-adjudicate judge semantics, prove hidden provider
retry behavior, attest model weight files, or remove benchmark contamination.
The raw prompts, references, responses, private endpoints, host identities and
credentials remain private. Fingerprints below identify the evidence without
redistributing that content. The inherited GLM qualification failures remain
applicable: **64K strict-format failure**, **900K semantic result 4/6**, and
`semantic_quality_passed=false`.

<details>
<summary>Frozen validation and audit fingerprints</summary>

| Artifact | SHA-256 |
| --- | ---: |
| Common canonical cases | 715d02c4da7cb18ef843494bf009270c097b555e81ead487aa7d8a6c67ad7ff7 |
| Three-single validation plan | c704611603b8954bc35c09a83a9a7a209431519686ebddf3ecc83c69981170a8 |
| R2 validation plan | c26a833d85bb25c5a22284f81c950312d051daedeb7143c90f63a929947211da |
| R2 active runtime | 4f1d0400dd7803c798f84806048afd1316ae3e8adcebcbff661b7005efa82a90 |
| Prepared dataset bytes | 358c3307ab3937585269920cadb4e26bc6d3ee7bdf190ec1e47c740df210fcf2 |

The run protocol retained source commit
`5f5bbdb80b30f277dbf3661907a1a0dfe853a132`. Subsequent product deployment does not
retroactively change this runner provenance. The source/runtime fingerprints
identify this experiment, not a portable provider configuration.

</details>

## Development scope and frozen protocol

The study ran a complete three-single-model matrix once, followed by current
Balance (R0), optimization round one (R1), and optimization round two (R2), on
the same 25 cases. This produced 150 subject generations and 20 separately
accounted judge calls: 170 retained calls in total. All four runs completed.

| Benchmark | Cases per target | Weight in this four-benchmark subset |
| --- | ---: | ---: |
| MMLU-Pro | 14 | 2/9 |
| GPQA Diamond | 4 | 1/3 |
| SimpleQA Verified | 5 | 2/9 |
| ARC-AGI-2 | 2 | 2/9 |
| **Total** | **25** | **1** |

The subset weights normalize the corresponding frozen sr-bench weights. A raw
15/25 score is 60% micro accuracy; it is not the 53.65% weighted score reported
for Flash, R0, and R2. Five of the nine sr-bench benchmarks are absent, so this
study has no full sr-bench score.

The shared protocol used concurrency four, a 4,096-token output ceiling, a
600-second request deadline, a 45-second idle deadline, a 1,300-second case
deadline, and a 5,400-second run deadline. Effective native request profiles
used temperature 1.0, top-p 0.95, seed 42, and one completion. GLM used its
declared `max` reasoning profile; Flash and Qwen27 used their declared `xhigh`
thinking profiles. Target-specific parameters overrode global defaults.

Only final-channel content was graded. Reasoning content was retained
separately and never used as an answer. A generation ending at the output
ceiling counted as incorrect and remained in the planned denominator, even if
its partial final text appeared parseable. SimpleQA used a frozen Flash judge;
judge requests saw the subject's final answer, not its reasoning.

The client journal records one subject dispatch per planned cell, with no
application generation retries. Direct router receipts acknowledge one
inference call for each routed request. These receipts do not independently
establish whether an upstream provider retried internally or identify the
backend weight files.

## The two development changes

| Revision | Change from the previous stage | Observed backend selections |
| --- | --- | --- |
| R0 | Existing Balance recipe, using its existing general/reasoning quality indices and decision pools. | Flash 13; Qwen27 12 |
| R1 | Added a distinct local 4K development benchmark/index populated from the three single-model results, then bound Balance's quality selection to that index. Existing imported quality records remained separate. Candidate pools and decision objectives stayed unchanged. | Flash 24; Qwen27 1 |
| R2 | Retained the local index; added a higher-priority factual-care guard with the quality-oriented pool. Ordinary reasoning and medium decisions considered all three models, used quality tolerance `0.10`, and prioritized cost before latency. | Flash 7; Qwen27 18 |

R2's guard combines care intent with factual-check or high-stakes-domain
signals. Routing rules did not match benchmark names, task IDs, or answer
labels. The local quality index was nevertheless fitted on these same
development cases; changing general rules does not remove that tuning bias.
Learning remained enabled with adaptation disabled and conversation protection
enabled. Preview snapshots were diagnostic; reported scores came from live
execution, not offline replay.

R1 gained one correct MMLU-Pro answer, but its cache-neutral cost was higher
than Flash's. R2 restored the original observed score and reduced cost under
the chosen simulated prices. This motivates independent validation; it does
not justify selecting R2 as a proven best policy.

The worker captured each recipe from the management API with stable source,
generated-runtime, and active-runtime hash checks. Live responses acknowledged
the corresponding frozen runtime identity:

| Revision | Frozen active runtime SHA-256 |
| --- | --- |
| R0 | `8e129a3735c4bf382de07176666a542361794f3b5fefa626294c3e0f1d685826` |
| R1 | `c968575f4460d1d5ad7cb5892613e5900b429fe505da2545f554c490857a5b6f` |
| R2 | `4f1d0400dd7803c798f84806048afd1316ae3e8adcebcbff661b7005efa82a90` |

The common case-set SHA-256 is
`b219b3846303addbb519eae0995f64589fd5649ea7511b763f1f8c80b62d8a2e`.
These fingerprints identify retained evidence; they do not expose private
deployment configuration or redistribute benchmark questions and answers.

### Reuse the final development recipe

Download the [R2 configuration fragment](/files/sr-bench/balance-r2-development-20260918.yaml).
It contains the complete captured `balance` recipe, including its signals,
projections, decisions, selectors, and plugins; the three local development
quality records and their benchmark/index declarations; and the captured
Learning settings. The recipe body is identical to the retained R2 snapshot.
Imported evaluation records are omitted so they cannot be mistaken for this
study's local measurements.

Merge evaluation declarations by ID, records by benchmark/profile/model, and
the recipe by name into an operator-owned canonical configuration. Deep-merge
`global.router.learning`; preserve unrelated global settings. Supply providers,
matching model cards and names, native request profiles, prices, listeners, and
entrypoint wiring for the intended environment. Those deployment details are
deliberately absent. This fragment is not a standalone configuration or a
recommended production default, and it does not replace the repository's
default Balance recipe.

Validate and plan the merged configuration through the config API, record its
new active runtime hash, and run preview plus an independent evaluation before
use. The full R2 runtime hash above identifies the original deployment, not a
new merged configuration. The fragment's own SHA-256 is
`e565da7fa6a7c1e70aa0da8368681e622cd854487e4644cde1764f85e14fcd64`.

## Development quality, cost, and latency

Costs are USD-equivalent at simulated prices. “Subject” excludes judge calls.
Latency percentiles measure individual subject requests, not end-to-end run
duration. All rows include all 25 planned cases.

| Target | Correct | Weighted score | Output-limit cases | Corrected subject cost | Judge cost | Subject p50 / p95, seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM | 11/25 | 44.44% | 5 | 0.15480840 | 0.00369460 | 54.51 / 303.37 |
| Flash | 15/25 | 53.65% | 6 | 0.08774285 | 0.00331695 | 7.00 / 40.45 |
| Qwen27 | 14/25 | 49.21% | 9 | 0.01680630 | 0.00056615 | 25.11 / 97.06 |
| Balance R0 | 15/25 | 53.65% | 7 | 0.04717511 | 0.00405470 | 7.11 / 116.15 |
| Balance R1 | 16/25 | 55.24% | 7 | 0.08481297 | 0.00354640 | 8.33 / 52.81 |
| Balance R2 | 15/25 | 53.65% | 8 | 0.04098496 | 0.00547040 | 23.83 / 113.34 |

| Target | MMLU-Pro /14 | GPQA /4 | SimpleQA /5 | ARC-AGI-2 /2 |
| --- | ---: | ---: | ---: | ---: |
| GLM | 7 | 4 | 0 | 0 |
| Flash | 10 | 4 | 1 | 0 |
| Qwen27 | 10 | 4 | 0 | 0 |
| Balance R0 | 10 | 4 | 1 | 0 |
| Balance R1 | 11 | 4 | 1 | 0 |
| Balance R2 | 10 | 4 | 1 | 0 |

All 12 ARC-AGI-2 subject generations reached the output ceiling. Their zeros
are quality outcomes under this 4K protocol, not missing rows; this experiment
provides little evidence about unconstrained ARC capability.

The three-single-model matrix took 18m37s in total. R0 took 4m06s, R1 2m28s,
and R2 4m40s. The matrix interleaved targets, so its wall time cannot be assigned
to individual single models. R2's request p50 and p95 were worse than Flash's;
its cost reduction did not translate into a latency improvement.

## Development paired comparisons and uncertainty

Flash was the unique strongest observed single model by the exact frozen
weighted score, `169/315`. The comparator is an actual single model, not a
per-question oracle. The general tie policy selects maximum exact weighted
quality, then the least complete known subject cost, then stable target ID;
unknown cost among tied strongest models prevents a cheapest-best savings claim.

| Balance revision vs Flash | Wins / losses / ties | Weighted quality change, pp | Conservative 95% interval, pp | Corrected subject saving | Cache-neutral subject saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 0 / 0 / 25 | 0.00 | −69.71 to +69.71 | 46.23% | 44.14% |
| R1 | 1 / 0 / 24 | +1.59 | −68.13 to +71.30 | 3.34% | −4.76% |
| R2 | 0 / 0 / 25 | 0.00 | −69.71 to +69.71 | 53.29% | 51.32% |

The primary interval is the conservative weighted paired Hoeffding bound:
the observed difference plus or minus
`sqrt(2 × ln(2 / 0.05) × sum(weight² / cases))`, clipped to `[-1, 1]`.
It assumes independent cases within the frozen benchmark strata. It excludes
uncertainty from choosing the strongest model, tuning two revisions, repeated
development use, and contamination. Its width makes this study inconclusive
for a capability-equivalence or non-inferiority claim.

The retained paired bootstrap is diagnostic only. It returns `[0, 0]` for R0
and R2 because no correctness indicators differ on these cases, and
`[0, +4.76]` percentage points for R1. A zero-width resampling interval on 25
reused cases is not proof of equivalence. Per-target and per-benchmark reports
also retain Wilson intervals for their ordinary accuracy denominators.

## Development accounting and the cache confound

The four token buckets below are exclusive: fresh input, cache read, cache
write, and output. Prompt totals include all three input buckets exactly once.
These are subject tokens; the 20 judge calls are accounted separately.

| Target | Fresh input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| GLM | 15,712 | 0 | 0 | 37,765 |
| Flash | 7,471 | 0 | 9,408 | 38,586 |
| Qwen27 | 5,903 | 0 | 10,976 | 49,480 |
| Balance R0 | 5,903 | 10,976 | 0 | 40,075 |
| Balance R1 | 7,471 | 9,408 | 0 | 40,880 |
| Balance R2 | 5,903 | 10,976 | 0 | 44,303 |

Frozen simulated rates, in USD-equivalent per million tokens:

| Model | Fresh input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| GLM | 1.20 | 0.120 | 1.5000 | 3.60 |
| Flash | 0.65 | 0.065 | 0.8125 | 1.95 |
| Qwen27 | 0.10 | 0.010 | 0.1250 | 0.30 |

The original parser classified cache-write usage as fresh input in four
baseline calls. Offline reconciliation used only their saved streams,
appended versioned accounting corrections, and preserved the original calls,
results, and manifests. Flash's subject cost changed from 0.08621405 to
0.08774285; Qwen27's changed from 0.01653190 to 0.01680630. The increase was
0.00180320 in total. No R0/R1/R2 cost required correction, and reconciliation
issued no model requests. The tables use the qualified corrected view.

Sequential runs wrote and later read caches. To make that confound visible,
the cache-neutral counterfactual charges every prompt token at its model's
fresh-input rate, then adds output at its output rate:

`neutral cost = (fresh + read + write) × input rate + output × output rate`

| Target | Cache-neutral subject cost |
| --- | ---: |
| GLM | 0.15480840 |
| Flash | 0.08621405 |
| Qwen27 | 0.01653190 |
| Balance R0 | 0.04816295 |
| Balance R1 | 0.09031665 |
| Balance R2 | 0.04197280 |

This counterfactual is neither billed cost nor a measured cache-free execution.
It does not remove cache effects on latency, model variability, or run order.
R1's negative cache-neutral saving is retained instead of claiming a uniform
cost improvement across both optimization rounds.

The included 170-call study totals 0.43233059 subject cost plus 0.02064920 judge
cost, or **0.45297979 USD-equivalent** after correction. This is not the total
project expense. Earlier failed attempts, pilot/UI requests, setup work, and
old Insights are outside this comparison; excluding them does not make their
cost zero or turn them into formal benchmark costs.

## Evidence and remaining acceptance boundaries

An independent audit performed 4,042 checks against all 170 saved streams and
their journal records, including final-only grading, output-limit treatment,
exclusive usage buckets, price identities, dispatch counts, model selection,
and frozen recipe acknowledgements. A second audit performed 885 checks across
the four application reports, four correction receipts, and three comparisons.
All checks passed. Original call-receipt digests remained unchanged for all
170 calls, and repeating reconciliation returned the same correction identity.
These audit counts describe evidence verification, not additional generations.

This establishes the bounded CLI development loop and report arithmetic for
the four included benchmarks. The separately audited validation above adds
held-out evidence within its disclosed scope, and shows that R2 did not meet
the quality target. Neither phase establishes full benchmark completion or
Dashboard lifecycle acceptance by itself. The separate live Dashboard evidence
is recorded below; setup, preflight and mocked UI tests do not substitute for it.

The following qualifications remain attached to these results:

- The 25 development cases were reused for tuning. They remain distinct from
  the 124-case non-GPQA held-out scope and the 12 GPQA validation retests above.
  Neither sample represents a measured production workload distribution.
- GPQA is a retest after earlier project labels were seen. A locally disjoint
  GPQA sample must still carry that prior-label disclosure.
- GLM's earlier **64K strict-format qualification failed**. Its **900K semantic
  qualification scored 4/6 and failed**; `semantic_quality_passed=false` remains
  in force. Completing this 4K study supersedes neither limitation.
- The SimpleQA judge's final-only inputs and verdict extraction were audited;
  its semantic decisions were not independently human-adjudicated.
- Runtime/model acknowledgements identify the recorded serving configuration.
  They do not independently verify model weights or every provider-side
  transformation of native request parameters.
- The earlier aborted evaluation campaign was not resumed. This development
  study does not claim completion of its 48,920-generation formal protocol.

Further optimization needs a new development cycle and fresh independent
validation. Using these published validation outcomes to change the recipe
would make those cases development evidence for that later revision. Larger
development matrices can improve routing estimates, but must remain distinct
from the evidence used to judge a frozen candidate.

## Adapter and lifecycle acceptance

All nine dataset adapters were prepared from pinned sources for the smoke,
quick, and standard profiles in the [sr-bench guide](./sr-bench). Gated HLE
access used an authorized local account; credentials and original dataset
content are not part of this publication. The 25-case, four-benchmark development
study above exercised MMLU-Pro and GPQA final-answer grading and SimpleQA's
fixed judge. All 12 ARC-AGI-2 subject responses reached their output ceiling,
so those model runs did not reach the puzzle-output grader. The additional
Flash smokes below exercised the remaining five subject-execution paths.

These are functional acceptance observations, not a uniform capability
comparison. The code and agent smokes used a 4,096-token ceiling; HLE used a
separately frozen 16,384-token ceiling. All used concurrency two. A completed
run can contain incorrect or truncated answers: completion describes the
planned denominator and retained evidence, not task success.

| Benchmark | Completed cases; correct | Subject / auxiliary calls | Subject cost | Auxiliary cost | Run wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| LiveCodeBench | 2; 1 | 2 / 0 | 0.01491035 | 0 | 42.67 s |
| SciCode | 1; 0 | 2 / 0 | 0.01602640 | 0 | 76.84 s |
| Terminal-Bench 2.1 | 1; 0 | 17 / 0 | 0.08673873 | 0 | 100.56 s |
| τ³ | 3; 2 | 41 / 21 | 0.07429266 | 0.02434185 | 87.55 s |
| HLE | 4; 0 | 4 / 0 | 0.12880855 | 0 | 347.66 s |

Costs use the same frozen, size-based **simulated** Flash rates shown above;
they are not invoices. τ³'s 21 auxiliary calls were simulator calls. Its
observed tasks needed no LLM judge call. HLE made no judge call because every
subject response was truncated. The five completed smokes contain 66 subject
calls and 21 simulator calls, totaling **0.34511854 USD-equivalent**. They are
separate from the 170-call Balance comparison and from failed or UI attempts.

Recorded exclusive usage buckets, with auxiliary usage kept separate:

| Benchmark / role | Fresh input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| LiveCodeBench / subject | 1,171 | 0 | 0 | 7,256 |
| SciCode / subject | 1,988 | 0 | 0 | 7,556 |
| Terminal-Bench 2.1 / subject | 14,215 | 257,152 | 58,016 | 6,998 |
| τ³ / subject | 29,015 | 271,264 | 28,224 | 7,625 |
| τ³ / simulator | 18,148 | 39,200 | 4,704 | 3,167 |
| HLE / subject | 1,559 | 0 | 0 | 65,536 |

The retained execution evidence has the following boundaries:

- **ARC-AGI-2:** the 12 development subject responses counted as incorrect after
  truncation. Separate zero-LLM controls passed the official complete output
  grids for each of the two sampled puzzles to the installed grader; both
  passed. Changing one output cell to another valid color made each puzzle
  fail, preserving the rule that all test grids must match. These four controls
  exercise the actual grader without changing model results or adding model
  capability evidence.
- **LiveCodeBench:** one generated solution passed all 34 sandbox tests. The
  other response reached its output ceiling and counted as incorrect without
  running that solution's grader.
- **SciCode:** the model completed the first subproblem and reached the output
  ceiling on the second. Its model run never reached the complete H5 grader.
  Separate zero-LLM controls used an official development reference: all three
  subproblems passed, while a deliberately incorrect final subproblem failed.
  Both controls used the actual pinned grader and H5 data in bounded,
  network-disabled containers, with successful cleanup. These controls validate
  grading behavior; they add no model capability score and do not replace the
  truncated model run.
- **Terminal-Bench 2.1:** the sole task was a reviewed, benign path-tracing
  exercise. The actual Harbor verifier completed and returned reward zero;
  its receipt recorded no verifier exception or retry. The subject reached
  its output ceiling, and owned container cleanup completed. This is evidence
  of a real verifier path, not successful task completion.
- **τ³:** the original attempt failed during simulator initialization because
  required pinned assets were missing, before any subject, simulator, or judge
  call. That failed parent remains unchanged. Explicit setup filled those
  assets at the same upstream commit; all three domains then initialized with
  network and model calls disabled. A reviewed `recover-plan` selected exactly
  the three undispatched cells. One separately authorized child inherited the
  frozen cases and limits, while capturing its new runner provenance. Airline
  and retail returned reward one; telecom reached its step limit and returned
  zero. All three real trajectories and rewards were retained. No previously
  sent generation was retried.
- **HLE:** all four subjects returned `finish_reason=length` at 16,384 output
  tokens. Each counted as incorrect in the four-case denominator, and its
  complete usage was retained. There were **zero judge calls in that model
  run**. A separate reference-judge control, described below, exercised the
  installed judge without repeating subject generation or changing the four
  truncated outcomes. The configured judge is not a claim of official HLE
  leaderboard equivalence.

Independent saved-stream audits passed 339 checks for the first three smokes
and 1,008 checks for the τ³ child and HLE, covering all 87 calls. Checks included
response identity, final-only content, native request parameters, exclusive
four-bucket costs, single dispatch receipts, output limits, and concurrency.
No audit issued inference requests. This evidence covers all nine subject
adapter paths, with the grader limitations above; it is **not** nine fully
qualified model-to-grader paths or a complete sr-bench score. Dashboard
lifecycle acceptance is reported separately below. The frozen capability
validation above is separately audited and retains its observed quality gap.

### Independent HLE reference-judge conformance

A separate, explicitly authorized control used one benign programming case's
private official reference as its positive fixture and a deliberately wrong
final answer as its negative fixture. An operator extension supplied those
fixture finals to the unchanged installed HLE grading function. Fixture receipt
events were not model calls. Only the function's two Flash judge requests went
through the ordinary instrumented request and accounting path.

Both controls passed: the reference was judged `correct`, and the wrong answer
was judged `incorrect`. There were **zero subject generations and exactly two
judge calls**, with no retry. The frozen control used concurrency one, a
2,048-token output ceiling, 60-second request and 30-second idle deadlines,
a 180-second run bound, and a 0.10 USD-equivalent simulated-price budget.
Both judge streams finished normally before their limits.

| Fixture | Expected / observed verdict | Fresh input tokens | Output tokens | Simulated judge cost | Request latency / TTFT, seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Official reference | correct / correct | 585 | 117 | 0.00060840 | 1.224 / 0.113 |
| Deliberately wrong final | incorrect / incorrect | 557 | 144 | 0.00064285 | 1.603 / 0.231 |
| Total | Two controls passed | 1,142 | 261 | 0.00125125 | Run wall time: 2.839 |

Cache-read and cache-write tokens were both zero. The exclusive cost buckets
were 0.00074230 fresh input, zero cache read, zero cache write, and 0.00050895
output. These are actual judge tokens at the same simulated Flash rates, not
an invoice. This separate **0.00125125 USD-equivalent control expense** is
excluded from the model capability runs, their savings comparisons, and the
five-smoke total above.

An independent audit passed 341 checks over the saved streams, frozen native
parameters, final-only verdicts, expected fixture outcomes, call counts,
exclusive usage and costs. It verified that the original four-case HLE run's
logical journal digest was unchanged. The isolated control service stopped
cleanly after its exact process identity was checked; both owned processes
exited, and the shared acceptance slot was released.

The frozen control plan is
`268e5507569d06d73a82e24c95af9d37e3f717661c34c4537f6f66434e21ab23`.
It retained the original `5f5bbdb80` runner; the installed HLE grading module
was byte-identical to the subsequent `9e3922e58` product deployment. This is
bounded evidence that the installed reference-judge path executes and
separates these two fixtures. It does not human-validate all judge decisions,
prove official leaderboard parity, repair the truncated model run, or add a
model capability score. The original question, reference and responses remain
private.

## Public Dashboard acceptance

The matching Dashboard and worker were deployed after the frozen validation
finished. Actual Chromium tests used the authenticated public HTTPS application,
with fresh browser contexts and no test retries. Seven zero-generation scenarios
passed: persisted single-model runs, arbitrary two- and three-revision
comparisons with URL persistence and CSV export, final recipe download, all six
pages of the 136-question dataset with coverage and question details, separate
112-case single-model and MoM plan reviews, and the completed 136-case comparison.
Desktop and mobile screenshots were reviewed. The tests observed no failed API
or asset requests, page errors, or unrelated configuration-compiler downloads.
Two conditional active-run reload scenarios were skipped: one had no configured
active run, and the other run was already terminal. Neither is reported as a
successful active-run test.

A separate synthetic lifecycle exercised the real create/review/start UI with
one Flash target, concurrency one and a 512-token output bound. The user-visible
cancel action followed the first observed durable call. The 16-case parent
retained six completed cases, one cancelled call and nine undispatched cases.
Its completed receipts account for **0.00134875 USD-equivalent**, while its total
cost remains unknown because the cancelled call supplied no complete usage.
The cancelled parent and its status survived a browser reload.

The UI then selected exactly one never-dispatched case and created one linked
recovery child. That child completed one call, with a denominator of one and its
own **0.00013975 USD-equivalent** cost: 116 fresh input and 33 output tokens,
with no cache-read or cache-write tokens. Its 0.431-second run wall time and
separate parent lineage remained visible after reload on desktop and mobile.
The child inherited neither the parent's cost nor its full denominator. No
previously dispatched cell was repeated, and these functional expenses are
excluded from all capability and optimization comparisons.

The original paid browser test stopped after the child completed because its
strict progress-object assertion omitted the valid `running: 0` field. That
failed test and its once-only submission receipt were preserved. The assertion
was corrected, and a separate read-only browser continuation verified the
existing parent and child, costs, lineage, reload and unchanged run inventory.
It issued no mutations or model calls. An independent raw-evidence audit passed
127 checks over the eight subject dispatches, native request parameters,
final-only scoring, token buckets, denominators and absence of duplicate calls.
No active or ambiguous-dispatch calls remained.

The retained pre-child evidence proves the parent's progress and dispatched
case identities. A complete historical parent-report digest was not saved
before child creation, so this acceptance does not claim a byte-for-byte
comparison of every parent field across that boundary. Current parent costs
were independently recomputed from its own receipts. The failed browser attempt
is not relabeled as a pass; the combined observed workflow and read-only
reconciliation establish the functional result.
