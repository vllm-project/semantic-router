# JevArena v3 answer-count erratum and post-key policy record

**Status:** discovered after the first 4B candidate's FINAL labels were opened.
This is a visible correction, not a prospective change to that candidate's
frozen protocol. Preserve the original pre-key plan, freeze, source mirror,
raw predictions, scored reports, paired intervals and failed ranking log byte
for byte. The original 4B release decision failed its predeclared Score-slice
floor and remains recorded as a strict-protocol **HOLD**.

## Count defect

The typed FINAL panel has **1,600 input items** in about 400 independent
four-variant groups. The `evidence_join` family asks both Choice and Noul on
each of its 400 items. The frozen typed scorer correctly evaluates **2,000
answers**: 800 Choice, 800 Noul and 400 Score; by family the counts are 400,
400, 800 and 400. The ranker incorrectly asserted that `overall.n == 1600`
despite receiving a valid `items == 1600`, `overall.n == 2000` score report.
The release gate and card probability table likewise used 1,600 as the
denominator for answer-level invalidity and coverage-adjusted Brier. Its
synthetic unit fixtures reflected that mistake. These count assertions are
not the capability formula: `T` remains the scorer's mean of four family
accuracies, `H` remains median task macro-F1, and the v3 score remains
`100*sqrt(T*H)`. The paired bootstrap already counts question-level answers
within each independent typed group; its seed and 5,000 draws are unchanged.

The corrected source uses 2,000 for answer-level validity and Brier, checks
the exact family/type answer shape, and shows `1,600 items / 2,000 answers`
in the card. The existing 8,147-item sealed panel count is unchanged. This
new source and gate policy have different hashes and apply to **future fresh
pre-key freezes only**. They must never be silently substituted into the
original freeze or described as the original preregistered implementation.

For the already frozen 4B run, the separately signed
`scripts/rank_postkey_answer_count_v3.py` may produce a **post-key diagnostic**:
it imports the unchanged frozen ranker, first reproduces its exact count
failure, lets that ranker verify its own original source hashes, plan,
freeze, roster and score-report identities, then replaces only its typed
count/shape validator in memory. It writes an exclusive private sidecar with
the original failure-log SHA, frozen ranker SHA, erratum-script SHA, freeze
SHA and unchanged report SHAs. That sidecar is not a strict release PASS and
does not modify any frozen file.

## User-directed aggregate-priority amendment

After the original blind scores were visible, the project owner requested
that an initial release prioritize a substantial **composite** gain while
allowing an explicit Choice/Noul/Score tradeoff. This is a **post-key policy
change** and cannot be called preregistered. The amended rule recorded in the
research log is: 2.0 minus paired 1.0 JevArena v3 score at least **+3.0
points**, with the 5,000-draw paired 95% interval lower bound above zero;
publish all per-type and per-task regressions. Retain complete 1,600/6,547
raw panels, the same frozen model and comparator, the same T/H formula and
bootstrap, JevBench public 231 as a separate report, answer-level invalidity
and Brier safeguards using 2,000 typed answers, and full native package
parity. This replaces the original requirements that both `T` and `H`
individually rise and that no typed type fall more than 0.02, for a separately
labeled *user-directed post-key release decision*. The original floor failure
remains visible. Future candidate freezes may prospectively adopt this rule.

A versioned amended-release verifier must bind the original strict HOLD
receipt, the owner's dated direction, this amendment text/hash, the original
pre-key freeze and source digests, the signed count-diagnostic code, every
unchanged scored report and prediction hash, paired CI and public report,
native model identity, rights/provenance and package parity. It must emit a
distinct amended decision receipt rather than altering the original gate.
The current `publication.bundle_arena_v3` cannot accept that receipt directly:
both `assemble` and `verify` require a strict `release-gate/1` PASS, and the
external audit checks the original policy/source hashes. A separate versioned
staging **and verification** path is required before any amended-policy HF
publication. Merely changing a `status` field or bypassing the SHA checks
would not be a valid repair.
