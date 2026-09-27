# ~27B teacher contrast: TRAIN-only whole-group capacity envelope

**Decision: HOLD.** This is a gold-free CPU feasibility result for a *new*
prospective schedule rule. It is not a restart, reinterpretation, or passing
result for the existing [frozen teacher contrast](qwen38-27b-train-teacher-residual-screen-2026-09-28.md),
whose greedy and exact-subset source-quota admissions both ended
`WHOLE_GROUP_QUOTA`. Neither receipt changes. No new 27B weight, optimizer,
teacher vector, selector label, development result, formal key, public-set
result, or HF artifact was used or produced here. GPU-hours: **0**.

## Frozen-input, answer-free capacity result

The CPU checker used the exact rights-clean v2 TRAIN file digest
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
the official Qwen3.8-27B source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and its pinned native
tokenizer files. It projected only input and provenance fields from TRAIN;
it did not inspect target labels. Token counts use the current native
segmented Choice/Noul/Score prompt, with no truncation. The aggregate-only
private receipt is SHA-256
`74a996796b5b42b746d4e83a23b7a5cde70eca13760f363f5be178b44db53816`.

| Capacity at 4,096 tokens | Count |
| --- | ---: |
| Frozen TRAIN rows | 7,455 |
| Admitted rows | 7,324 |
| Admitted Score rows | 507 |
| Score groups partially cut by the context cap | 0 |
| Choice target for a 2,560-row schedule | 1,026 |
| Noul target for the same schedule | 1,027 |
| Fixed strict source/type quota buckets without an exact whole-group subset | 1 Noul bucket |
| Minimum pooled whole-group source-quota L1 shift at unchanged type totals | 2 rows |
| Largest individual source-quota shift in that solution | 1 row |

Thus a **separate future protocol** could retain all 507 admitted Score rows,
the exact 1,026/1,027 Choice/Noul targets, the same 2,560 examples and 160
updates, but replace strict per-source exact quotas with this prospective
deterministic rule: choose source/type whole-group counts that minimize total
absolute deviation from the original largest-remainder source quotas; break
ties in fixed source-hash order. The 2-row aggregate drift corresponds to
one source gaining one row and another losing one row. This is the smallest
possible deviation under those constraints, not evidence that a model will
improve. The A/B treatment, if independently admitted later, would remain the
previously defined matched `.02` hard-versus-soft auxiliary target contrast
on identical rows and a single final checkpoint. No new arm is authorized by
this capacity calculation.

**Group-scope caveat:** the frozen selector defines whole groups inside each
source/type bucket. The parent TRAIN has 1,400 group IDs spanning task types;
neither the failed old selector nor this minimal envelope proves that every
cross-type variant of a case would be selected together. That scope must be
declared in any new lock and checked for leakage and answer cues. The matching
A/B row set alone does not establish source-disjoint transfer.

## Remaining admission gates

An existing candidate protected-prompt inventory has all five required role
names; its 32 referenced files are present and byte hashes match. This is
only file/role availability. The new schedule has **not** been materialized,
the prompt-only row schema and candidate overlap have **not** been checked,
and its teacher mask and source-specific rights have **not** been reattested.
The previous exact-quota CPU run stopped before the teacher-mask and protected
inventory checks. Therefore a complete source/data/rights/teacher/overlap
admission path is **unproven** and remains HOLD.

Before any device reservation, create and sign a distinct versioned new-arm
lock that preserves the old failed receipts; materialize deterministic selected
IDs and order in a private manifest; revalidate pinned teacher row/option
identity, mask minima, full rights scope, and protected-prompt exact/near plus
reviewed semantic overlap; and run the documented zero-step and one-step
native parity gates. Missing or failed checks stop this version. Do not select
another source quota, teacher threshold, seed, checkpoint, or development
panel after reading outcomes.

Reproduction: `training.data.audit_27b_quota_envelope` in this branch is the
CPU checker. Its output excludes row IDs, source names, private paths and raw
text. Synthetic group-capacity tests and the repository changed-path training
checks pass. This receipt neither revises the old experiment nor changes any
published performance number.
