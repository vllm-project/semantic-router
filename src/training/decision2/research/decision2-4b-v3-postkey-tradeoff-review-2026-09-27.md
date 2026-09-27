# Decision 2.0 4B: post-key tradeoff review

This review follows the user's **post-key** instruction to prioritize a
statistically supported aggregate improvement for a first release while
explicitly disclosing narrower regressions. It does not amend the original
pre-key policy or turn the [frozen 4B HOLD](decision2-4b-jevarena-v3-first-release-hold-2026-09-27.md)
into a preregistered PASS. All numbers below come from the same immutable
JevArena v3 predictions and reports identified in that record; no checkpoint
or sealed-panel selection was repeated.

## Criteria worth retaining

A composite-first publication choice still needs:

1. The frozen candidate, same-size Decision 1.0 baseline, open control,
   prompts, native adapters and scorer; complete prediction and gold digests;
   no training/evaluation overlap or post-key checkpoint selection.
2. A positive *paired* 95% lower bound for the predeclared composite score,
   with the original 5,000-draw, seed-`20260927` grouped bootstrap. Report both
   component axes and their separate intervals. Keep the original requirement
   that the `T` and `H` point estimates exceed the matched 1.0 model.
3. The frozen invalid/missing limits, complete probability coverage, and a
   coverage-adjusted Brier guardrail calculated over the actual number of
   scored answers. State any protocol correction as an erratum and independently
   review its arithmetic before release.
4. The same frozen native package and adapter for the 8,147 original items and
   separate public 231 questions, measured loaded parameter count, full package
   parity and byte-inventory checks, training-data rights and provenance, and
   complete task/type reporting. Public JevBench is not an official sealed
   JevBench rank or part of the v3 composite.
5. A visible regression disclosure for each Choice/Noul/Score type and each
   human task. A post-key policy choice cannot erase the original Score
   failure, or describe it as an independently preregistered non-regression.

These retain the primary protections against a selective or mismatched
comparison. Dropping the Score-slice cap is a product tradeoff made *after*
seeing this result. It is not confirmatory evidence that Score is preserved.

## Independent numeric assessment under that choice

| Check | Evidence | Assessment |
| --- | --- | --- |
| Same-size fair comparison | 4,205,751,296 candidate vs 4,208,383,488 Nox parameters; ratio 1.000626 | Meets size rule |
| Two component points | `T` 0.750000 vs 0.614375; `H` 0.523704 vs 0.519046 | Both positive |
| Composite paired evidence | 62.6720 vs 56.4702; difference +6.2018 points, 95% CI [+2.3882, +10.7653] | Positive lower bound |
| Individual transfer certainty | `H` difference +0.004658, 95% CI [-0.056113, +0.080811] | Do not claim a significant transfer gain |
| Invalid/missing | Typed 0/2,000 answers each; transfer 4/6,547 each | Within the original guardrail |
| Probability quality | Coverage 2,000/2,000 each; adjusted Brier 0.130543 vs 0.204949 using the correct 2,000-answer denominator | Within the original +0.03 margin as a sensitivity result; implementation erratum pending |
| Public JevBench cross-check | Candidate 195/231 vs Nox 173/231, with easy 48/48, standard 69/72, hard 78/111 | Complete, separately reported |
| Score retention | Candidate 167/400 vs Nox 178/400; -2.75 pp | **Fails original -2.00 pp cap; disclose prominently** |

The candidate's 15 human tasks split 7 improved and 8 regressed; the most
material macro-F1 losses are `ibc` (-0.114153) and `tempowic` (-0.097570).
The typed Score deficit and the uncertainty of the transfer gain constrain
the claim even though the composite interval is positive. The open Kev control
scored 59.2263 on the same T/H panel and 175/231 on the public subset; this
does not establish a broader open-model Pareto frontier.

## Release interpretation and remaining checks

**Numeric composite-first assessment: eligible for a disclosed, post-key
tradeoff decision. Publication is not yet technically cleared.** The current
ranker treats 1,600 typed items as 1,600 scored answers, although they expand
to 2,000; the current package numeric gate also rejects 2,000 accepted
probabilities and uses 1,600 in its Brier adjustment. A reviewed, versioned
erratum must fix those accounting defects without changing raw predictions,
the T/H formula, bootstrap, frozen score reports or the original HOLD record.
Then the corrected rank/card generator, rights/overlap review, full package
parity, final byte check and publication bundle must all pass. Until those
checks complete, the accurate status is **post-key release candidate, not
published**.

If released under this explicit user decision, the card should state:
“JevArena v3 aggregate improves by 6.20 points over the same-size Decision
1.0 Nox on the frozen two-axis panel (paired 95% CI +2.39 to +10.77). Score
accuracy falls from 44.50% to 41.75% and fails the original pre-registered
non-regression cap; human transfer median rises only 0.47 percentage points
with an interval spanning zero. This first release follows a post-key
aggregate-first tradeoff decision.” The card must also name the 8 of 15
regressing human tasks and keep the public JevBench result separate.
