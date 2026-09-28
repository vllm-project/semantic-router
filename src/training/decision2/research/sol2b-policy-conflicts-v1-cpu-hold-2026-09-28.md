# Sol 2B policy-conflict source v1: CPU admission screen

**HOLD. No source admission, GPU training, formal scoring, calibration, or
checkpoint selection.** This is the first frozen candidate from the
[prospective Sol 2B data-arm protocol](sol2b-formal-gap-next-arm-2026-09-28.md),
not a completed model experiment. Its fixed state-removed shortcut screen
failed Noul. Thresholds and the generated roster were not changed in
response. The private candidate and blind-review packet remain outside the
public repository.

## Candidate and checks

The deterministic generator has 512 structured situations, 128 in each of
eligibility, permission, scheduling and inventory. Each situation produces
one native Choice, Noul and three-level Score request: 1,536 TRAIN candidate
rows total. The policy oracle evaluates a credential gate, a posted-versus-
voided resource ledger, a hold/release exception, and conditional versus full
completion. An independently expressed ordered-rule oracle agrees on the
tested state grid and all generated candidates. The correct answer is
computed before rendering the prompt. The 48-group review packet contains 12
groups per family and excludes answer and oracle fields. Structural validation
is **not** semantic blind review: independent reviewers have not supplied
the required 44/48 unambiguous verdict.

| Frozen check | Observed | Protocol gate | Outcome |
| --- | ---: | ---: | --- |
| Independent groups / native rows | 512 / 1,536 | 512 / 1,536 | Pass |
| Choice A/B labels | 256 / 256 | balanced | Pass |
| Noul false/true labels | 256 / 256 | balanced | Pass |
| Score 0/1/2 labels | 172 / 172 / 168 | balanced | Pass |
| Blind packet groups | 48, 12/family | 48, 12/family | Structural pass; semantic review pending |
| Native source tokens | 1,004,169 | 20–25% of 4,194,465 control tokens | 23.9403%, pass |
| Choice / Noul / Score token shares | 32.95% / 32.87% / 34.19% | each within ±5 points of one third | Pass |
| Maximum native input length | 676 tokens | no truncation | Recorded, not an independent gate |

The exact archived control TRAIN hash was verified before tokenization. The
token count uses the same segmented native input path and the original Sol
tokenizer. The control contains 7,455 rows and 4,194,465 native tokens. All
2,800 existing human-labeled rows and 516 old Score rows are retained by the
prospective replacement rule. There are 3,551 whole replaceable groups,
containing 4,139 rows and 3,583,360 tokens. This establishes **capacity**
for 1,536 replacement rows and 1,004,169 new tokens. It does **not** establish
an exact 7,455-slot, ±1%-token, group-atomic replacement roster: none was
selected after the earlier shortcut failure. No training-data token budget
claim beyond this capacity check is made.

The separately generated 128-group diagnostic is group-disjoint from the
candidate. A four-fold bag-of-words classifier sees only the question text
and option descriptions, never the policy state, source or group ID. The
baseline is the fixed global majority rate for each type, as frozen in the
protocol. A pass requires no more than five percentage points above that
baseline.

| Type | Fixed majority | State-removed accuracy | Difference | Outcome |
| --- | ---: | ---: | ---: | --- |
| Choice | 50.00% | 47.66% | −2.34 points | Pass |
| Noul | 50.00% | **67.19%** | **+17.19 points** | **Fail** |
| Score | 34.38% | 18.75% | −15.62 points | Pass |

Noul question mode and target-label construction are associated with this
leakage: the answer can be predicted too often without reading the state.
The candidate therefore cannot test the stated policy-reasoning hypothesis.
A later redesigned source would require a fresh prospective registration and
new independent diagnostic; changing this candidate's label schedule or
threshold after seeing this result would invalidate the current screen.

## Reproduction and remaining gates

Public source and CPU checks are
[`build_sol2b_policy_conflicts.py`](../training/data/build_sol2b_policy_conflicts.py),
[`audit_sol2b_policy_conflicts.py`](../training/data/audit_sol2b_policy_conflicts.py)
and their unit tests. Six focused tests pass, including native row schema,
full generator oracle reexecution, option-order logic, deterministic output,
blind-packet answer exclusion and this exact failed shortcut regression.
The private TRAIN candidate SHA-256 is
`5e1d3422b1ee289dc6cbd1a9bb608095e4123d812b7f39455820265413141f44`;
the blind packet SHA-256 is
`198ef71aabd07e5339547f026f73a2285548216aed2f1f7a21b2dc9c3fc97fc9`;
the complete private CPU audit SHA-256 is
`90e8a55f9978e5075592f5d538b9ebfacb4c928a90d3d9fcc1750e186913c0de`.
No protected prompts, answers, private locations or raw rows are included
here.

Independent semantic blind review, rights review, overlap quarantine,
exact replacement roster and zero-step source parity remain unperformed.
They are not waived by the token and schema passes. This v1 source must not
be uploaded as an admitted training set or used for any model run.
