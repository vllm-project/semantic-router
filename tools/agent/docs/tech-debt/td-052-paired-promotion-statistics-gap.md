# TD052: Paired Promotion Statistics Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

Comparisons can diagnose clear gate failures and point regressions, but they
must return `unavailable` rather than `pass` for promotion until this gap is
closed.

## Scope

- private case alignment between baseline and candidate
- paired delta confidence intervals and non-inferiority tests
- slice, multiplicity, and missing-pair policy
- sequential and repeated-seed comparison evidence

## Summary

The current comparison contract validates workload, benchmark, seed, profile,
baseline linkage, and treatment-factor compatibility, then computes aggregate
metric deltas. It has no case-aligned baseline/candidate observations in the
public report and therefore cannot estimate a paired delta distribution.
Separate confidence intervals on two aggregates would not be an equivalent
test.

## Evidence

- Private execution records retain opaque case IDs, but `compare_reports`
  receives only aggregate reports.
- The comparison rejects self-comparison, a mismatched baseline, incompatible
  snapshots, and a profile whose treatment factor did not change.
- Aggregate regressions may fail a candidate; aggregate non-regression cannot
  produce a passing promotion verdict.

## Why It Matters

Ignoring pairing loses power, mishandles correlated cases, and can conceal
coverage shifts. Point estimates also provide no uncertainty, non-inferiority
margin, or protection against repeated metric and slice testing.

## Desired End State

The comparison service reads authorized private records, joins an immutable
case universe, applies a declared missing-pair policy, and emits paired effect
estimates with confidence bounds and pre-registered promotion margins. Public
output contains aggregate statistics only.

## Exit Criteria

- Define a versioned comparison manifest with primary metrics, direction,
  margin, alpha, multiplicity method, slice plan, and missing-pair policy.
- Join baseline and candidate records by opaque case and repetition IDs and
  verify identical grading/workload lineage.
- Implement deterministic paired bootstrap or an appropriate paired test,
  repeated-seed aggregation, effect bounds, and effective sample counts.
- Add known-distribution, missingness, zero-variance, and regression fixtures.
- Permit `pass` only when every required comparison has sufficient paired
  evidence and its bound satisfies the registered margin.
