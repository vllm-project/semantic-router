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

## Protected-inventory schema follow-up

The available candidate inventory has 32 unique roles and 23,414 rows; all
32 referenced files exist and match their manifest hashes. The five required
27B evaluation role names and expected row counts are present: typed DEV
1,600, CSS pilot 1,430, typed FINAL 1,600, CSS15 6,547 and public subset 231.
**It is not a valid gold-free inventory for the strict 27B admission parser.**
Four role files contain nested answer-like keys: the public Decision Bench
view on 30 rows, CSS15 on 6,547 rows, CSS pilot on 1,430 rows and typed FINAL
on 400 rows. This audit counted key names only, never inspected values or
printed raw prompts. The parser stops at `PROTECTED_NESTED_GOLD_FIELD`.
Key-name rejection alone does not prove these fields contain gold: the native
prompt contract uses a CSS `questions.label` as a question ID and permits one
exact `state.target={entity,item}` task-input shape. Conversely, the optional
Decision Bench role's `state.answer` still lacks source-specific input
attestation and must remain excluded from any admission claim.
The inventory also lacks the rights-clean TRAIN, SELECT and CAL roles required
by new training-source audits. No overlap result can be inferred from a
file-hash PASS. A future version must use role-specific input-only projections,
prove that the projected prompt still covers the actual native inference
surface, add the three partition roles, and pass the strict parser before
candidate overlap is considered. GPU and release status remain HOLD.

### Prospective input-only projection, unexecuted on protected data

`training.data.plan_goldfree_inventory` is a local CPU-only construction
prototype with six synthetic tests. For the five required native evaluator
roles it requires pinned sealed-prompt bytes, calls the existing native
`load_gold_free` loader, preserves the full `state` and `questions` input
content as text, and records the native `input_digest` in a private sidecar.
It does **not** edit the model request or delete legitimate task fields. For
rights-clean TRAIN/SELECT/CAL it selects only the four hashed input fields,
checks each source `input_sha256`, then projects input content to text; no
target field is accessed by the projection. Exact eight-role counts, IDs,
top-level field whitelist, text-only schema and absence of nested answer-like
JSON keys are asserted. The synthetic tests include legitimate question-ID
and target-entity inputs, an answer-bearing prompt rejection, count and hash
failures, and a nonserializable target sentinel.

This is **schema feasibility, not an inventory or overlap PASS**. No real
sealed prompt or partition was opened by this prototype, and the candidate
32-role inventory was not replaced. The three rights-clean role files and
their pinned source hashes still need private construction and a complete
source-level audit. Optional roles need separate native-source validation.
The current 27B overlap checker only scans `state`, so it cannot certify the
whole native question/option surface from this projection; a newly versioned
full-input-surface screen and reviewed semantic/source overlap are required.
The newer Choice source auditor scans all input text fields but pins the old
manifest digest, so it too needs a prospective manifest/version lock before
use. Existing exact-quota HOLD receipts and teacher/data rights gates remain
unchanged. GPU-hours for this follow-up: **0**.
