---
title: Balance development study — September 18, 2026
description: A bounded sr-bench study of three single models, the current Balance recipe, and two routing revisions, with corrected accounting and explicit uncertainty.
---

# Balance development study — September 18, 2026

The final development revision, R2, matched Flash's observed weighted score on
25 development cases while using 53.29% less token-equivalent subject cost at
the frozen simulated prices. Its cache-neutral saving was 51.32%. These are
small-sample development observations: they do not establish capability
equivalence, production savings, or a complete sr-bench 1.0 score.

Requests, response usage, and latency were measured from real executions.
**Prices were a size-based simulation for this deployment, not provider billing
or measured GPU cost.** Every monetary value below applies those frozen prices
to recorded usage. See the [sr-bench guide](./sr-bench.md) for the reusable
dataset, CLI, Dashboard, and reporting workflow.

## Scope and frozen protocol

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

## Quality, cost, and latency

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

## Paired comparisons and uncertainty

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

## Four-bucket accounting and the cache confound

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
the four included benchmarks. It does not establish all-adapter execution,
independent holdout quality, full benchmark completion, or Dashboard lifecycle
acceptance. Those release checks remain separately tracked; setup, preflight,
and mocked UI tests cannot substitute for live acceptance.

The following qualifications remain attached to these results:

- These are reused development cases. No holdout result is included here, and
  no production workload distribution is represented by 25 benchmark cases.
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

The next capability decision requires a frozen final recipe, prespecified
quality and cost criteria, and independent validation without tuning against
its outcomes. Larger development matrices can improve routing estimates, but
they must remain distinct from that validation evidence.
