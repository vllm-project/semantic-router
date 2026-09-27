# JevArena v3 first-release decision rules

**Prospective status:** written before the typed FINAL or CSS evaluation
labels are opened for Decision 2.0. This document specifies how a first
release decision will use the two sealed v3 axes. It is not a score report or
an authorization to open either panel. The final signed freeze declaration
must bind the exact scorer and paired-comparison hashes, models, prompts,
calibration, package manifests, thresholds and prediction output paths.

## Panel, comparators and score

- Typed FINAL has 1,600 Choice/Noul/Score answers in about 400 independent
  related groups. `T` is the preregistered family-macro accuracy; invalid,
  missing and over-budget answers are wrong.
- CSS evaluation has 6,547 human-labeled answers in 15 tasks. `H` is the
  median task macro-F1, with invalid and missing answers retained as misses.
- The v3 capability score is `100 * sqrt(T * H)`. Neither the public 231-item
  JevBench result nor future authored items enter this score. The JevBench
  public result is required as a separately labeled first-release cross-check.
- Each proposed Decision 2.0 model is paired with the frozen native Decision
  1.0 model for its actual size. The comparison is rerun on exactly the same
  v3 panels. A model size with no credible corresponding 1.0 comparator must
  name that limitation before scoring and use a predeclared nearest-size
  comparator; it may not claim an equal-size win.

## Numeric first-release gate

For a candidate to be labeled a **validated Decision 2.0 improvement**:

1. Both `T` and `H` point estimates must be strictly greater than the paired
   1.0 comparator. The v3 aggregate score must also be greater.
2. The 95% paired interval for the **candidate minus 1.0 v3 score** must have
   a lower bound above zero. Use 5,000 fixed-seed replicates. Preserve paired
   model answers, sample typed independent groups within family, and sample
   CSS tasks then records within each sampled task. Report axis intervals as
   well; do not infer a significant gain on an individual axis from the
   aggregate interval.
3. No typed Choice, Noul or Score family accuracy may be more than 0.02
   absolute below the corresponding 1.0 family. This guards against a large
   slice regression hidden by the macro score.
4. On each sealed axis, the invalid-or-missing fraction must be at most
   `max(0.02, comparator_fraction + 0.01)`; the score still counts all such
   answers as wrong. Typed Brier must be at most the 1.0 typed Brier plus
   0.03, using the same normalized probability policy. Failed probability
   validity cannot be treated as a favorable Brier omission.
5. Every required 8,147-item prediction panel must be complete, hashed and
   attributable to one frozen native model/calibration package. Packaging
   parity must pass its previously specified categorical and probability
   drift limits. The public JevBench run must use the same frozen model and
   adapter, score all 231 eligible items and publish easy/standard/hard.

If a gate fails, mark that size `HOLD` and keep the immutable result; a later
training or adapter candidate needs a new preregistered run and cannot reuse
the sealed panel for selection. Do not tune these thresholds after viewing
FINAL labels or publish only the favorable subset of axes.

## Mandatory context around a pass

Publish the 15 CSS task rows, three typed slices, paired point differences and
intervals, coverage and probability metrics, long-input and paired-order
diagnostics, and the language panel with its independent source-group counts.
The existing small multilingual panel is a diagnostic and cannot establish a
universal multilingual improvement; report regressions plainly. Rank and
parameter Pareto plots must use actual loaded parameters and one benchmark
edition per plot. A v3 pass says only that the frozen first-release protocol
was met, not that the model is universally best or has passed the future
authored v3.1 expansion.

## Current 4B candidate, without opening FINAL

The clean-v2 Eikos native SemIf package selected checkpoint 0232. Its already
completed gold-free selected-LoRA/merged-package parity receipts cover the
full DEV 1,600 and CSS pilot 1,430 prompts with zero categorical mismatches
and zero option-probability drift. The combined receipt reports gate pass.
These establish package consistency only. The DEV, pilot and public results
are development evidence; this model has no v3 `T`, `H` or paired interval.
The signed freeze must verify all package and receipt hashes anew, and pair it
with native Decision 1.0 Nox 4B on the sealed panels.
