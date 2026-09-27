# Decision 2.0 first-release aggregate policy amendment

**Recorded 2026-09-27 09:07 UTC, while the own-Nox-origin 4B arm was still
training and before that arm used typed DEV, CSS pilot, typed FINAL, CSS15 or
public JevBench labels.** This is an owner-directed change to release policy,
not a claim that the original arm was preregistered with these thresholds.
The original arm and its selection rule remain preserved at their first
commit. Its SELECT-only checkpoint choice, CAL-only temperatures, frozen
training budget, and failed preflights are unchanged.

The owner set a first-release objective of a substantial **combined**
improvement over the corresponding Decision 1.0 size, approaching strong
same-size open models selected from Decision Index 0.2.1. Individual typed,
transfer, language, robustness or calibration regressions do not
automatically disqualify a candidate; every measured regression must appear
in the release evaluation and reader-facing model card. The same principle
applies to subsequent sizes. The 27B has no size-matched 1.0; its initial
target is the first tier of verified near-27B open models on our same panel.
Index scores are used only to choose comparators, never inserted into a
JevArena or public JevBench rank.

For the ongoing 4B arm, the earlier development Score-loss cap is retained
in its historical result and reported as pass/fail, but is **not** a hard
publication veto under this later owner decision. Development promotion
still requires complete typed coverage, CSS pilot invalidity no worse than
the frozen 1.0 source, and the originally fixed combined DEV/pilot threshold
of 56.0. Report Score and every other slice regardless. A failed original
guardrail cannot be rewritten as passed. The formal first-release judgment
uses actual frozen candidate weights on JevArena v3 and a separate public231
rerun against same-panel 1.0/open peers, paired uncertainty and a full package
readback/parity check. A candidate altered after already opened v3 labels
cannot claim a virgin blind test; that exposure and any later independent
confirmation must be disclosed.

The public card shows same-panel rank charts and a model-by-task matrix for
JevArena v3, plus the public JevBench rank chart. Do not show Pareto plots.
Retain actual loaded parameter counts in the results table. Internal audit
and gate artifacts stay private; the card explains capabilities and limits
in product language.
