# 0.6B encoder track, Milestone 6: amendment b (Score-level guard; post-readout, disclosed)

Written 2026-09-29 ~06:25 UTC+8, before any M6 formal run, and before the readouts of `m6-mx-s2`,
`m6-mx-s3` and the m6-mx family soup existed (both seeds were still training). It amends §3 of
[`m6-prereg-2026-09-29.md`](m6-prereg-2026-09-29.md). All other rules, including the formal successor
rule in §4, are unchanged. It follows the precedent of the accepted M4e amendment, which also replaced
a development guard after a readout, before any formal run, with disclosure.

## What had been read

Development readouts only (typed DEV + CSS pilot), with no formal panel:

- the seeds `m6-cx-s1..s3` and `m6-mx-s1`;
- the family soup `m6-cx-soup`;
- `m6-zcx-soup` (Z + m6-cx, 12 seeds), built early because it is the first greedy step whenever
  m6-cx's family Q exceeds m6-mx's;
- the M4/M5 reference soups.

## The problem

The §3 guard "Score predictions use ≥ 3 distinct levels" was meant to stop a near-constant (collapsed)
typed-DEV Score from reaching the formal runner. As written, it counts levels, not their use, and a
handful of items at the margin decides it:

| Soup | Typed-DEV Score predictions by level | Modal share | ≥ 3 levels |
| --- | --- | ---: | --- |
| released `m4-t-a7-soup` | 0: 384, 2: 16 | .960 | fail |
| `m4-v2-soup` | 0: 219, 1: 181 | .547 | fail |
| `m5-a-soup` | 0: 284, 1: 7, 2: 109 | .710 | pass |
| `m5-x-soup` | 0: 368, 1: 2, 2: 30 | .920 | pass |
| `m5-z-soup` | 0: 323, 1: 3, 2: 74 | .807 | pass (on 3 items) |
| `m6-cx-soup` | 0: 6, 2: 394 | .985 | fail |
| `m6-zcx-soup` | 0: 235, 2: 165 | .588 | fail (on 0 items at level 1) |

Z passes on 3 of 400 predictions. `m6-zcx-soup`, whose predictions split 59/41 over two levels, fails.
The released model fails too, although its Score is the most constant of all.

## Amended guard (replaces "≥ 3 distinct levels" everywhere in §3 and §4)

**Score is not near-constant:** the modal predicted typed-DEV Score level covers **≤ 90%** of the
400 Score predictions.

The other guards are unchanged:

- typed-DEV Choice ≥ 255;
- typed-DEV Score ≥ 101;
- SELECT and CAL Noul accuracy ≥ .85.

The verdicts that matter are robust to the threshold:

- `m6-zcx-soup` passes for any threshold ≥ .59.
- `m6-cx-soup` fails for any threshold < .985.
- Z passes for any threshold ≥ .81, so S0 = Z (unless the m6-mx soup passes with a higher Q) for
  every threshold in [.81, .985).

Under the amended guard, X fails (.920) and the V2 soup passes (Choice 385, Score 111, Noul .910 / .928).
Neither changes S0.

## Procedure

- The greedy of §3 is re-run from the start under the amended guard; eligibility, order, acceptance
  rule and the ≤ 4-step cap are unchanged.
- The finalists of §4 use the amended guard.
- The results report both outcomes:
  - under the original rule, `m6-zcx-soup` is rejected at step 1 (guards fail on the level count);
  - under the amendment, the greedy continues from `m6-zcx-soup`.
- The near-constant Score of the m6-cx family is a finding and is disclosed.
  - Seed levels: s2 is at 100% level 2, s3 at 90.2%.
  - One explanation is own-Lux KL on the human Score rows (arm (a) trained them on gold). The
    recipe changed in other ways too, so this is not attributed.
