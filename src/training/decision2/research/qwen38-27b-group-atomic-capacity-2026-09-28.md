# 27B group-atomic capacity after near-state exclusions

**Status: a CPU-only feasibility witness for a future version; no training
admission.** The original 2,560-row pooled candidate remains HOLD, and the
two older exact-quota arms remain failed. This calculation does not change
their frozen Score minimum of 460 rows or any previous receipt.

The input-only capacity code SHA-256 is
`2f37205402d7315648e937564a7158f8cadfdfec8209874baddb43709360b24b`.
It ran in the pinned CPU image
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`
from an exact local-source mirror. The private, mode-0600 witness receipt
SHA-256 is
`809912b45c75aa7b630fd8fb40cafe898dc55a7201cc47cb9084484b0abb134d`.
Its selected-row/input/token identity digest is
`0743da3f1ed18cc8eb824c41c736d92b23fb3d8a738e1b3401ff830229424be5`.
The selected IDs and source inputs are private and are not reproduced here.
GPU-hours: **0**.

## Measured capacity

An exact/SimHash near-state screen compared the full 7,455-row pinned TRAIN
against the gold-free SELECT and CAL input projections. It found 60 TRAIN
rows in 101 near pairs with CAL and 44 TRAIN rows in 60 near pairs with
SELECT; the union is **83 rows** spanning **61 source groups**. Excluding the
*entire* affected source groups and groups with any native input above 4,096
tokens leaves 5,200 groups and 7,219 rows. A self-check on the previous
2,560-row candidate reproduced its 52 SELECT and 89 CAL near-state pairs.
Near pairs are conservative signals rather than confirmed duplicate labels.
Neither SELECT nor CAL answers were read.
The private receipt saves all 101 CAL and 60 SELECT bounded-pair ID tuples,
and the exact 83 excluded TRAIN IDs. Their respective pair-list identity
digests are `052d1101b0bbfb8a093d797431af613b1e398c67676560b450dcd2f644295a7c`
and `a82db9a3ef147bb973cd12e8a2feda0b4ac351a8f7c19dfc08a625752a8cc703`.
Thus the 83-row exclusion is traceable to saved pair identities under the
specified bounded algorithm; it is not a claim that every pair is a semantic
duplicate. The selected 2,560-row witness is reproducible from the pinned
inputs and seed 96. Searching 128 seeds to find it was exploratory capacity
analysis, so that seed cannot retroactively become the old arm's selection
rule.

| Requirement / capacity | Result |
| --- | ---: |
| Eligible Score rows after whole-group exclusion | **419** |
| Eligible Score teacher mask | 174 |
| Eligible three-level Score teacher mask | **63** |
| Old frozen Score-row minimum | 460: **impossible after hard exclusion** |
| Future sensitivity floor | 400: feasible as a *new*, predeclared version only |
| Deterministic whole-group witness | 2,560 rows, 160 nominal updates at 16 rows/update |
| Witness Choice / Noul / Score | 1,201 / 940 / 419 |
| Witness total teacher mask | 971; human Choice+Noul 797, Score 174, three-level Score 63 |
| Native raw token exposure | 1,244,573 vs prior 1,244,036, +0.043% |
| Eight-token padded exposure | 1,253,880 vs prior 1,253,048, +0.066% |

The witness is one of 128 deterministic source-group hash orders, selected
solely for proximity to the old raw/padded token budget after verifying the
frozen teacher mask. This is **feasibility search**, not a prospective
training-selector result. The 1,201/940 Choice/Noul mixture differs from the
earlier roughly even type balance; it requires a new explicit objective and
source-quota decision. Eight-token padding is not actual dynamic microbatch
padding, and 160 is only the nominal update count at accumulation 16.

## Boundary before a new training arm

The old Score≥460 arm cannot be repaired by relabeling this witness. A new
arm, if chosen, must preregister the Score≥400 floor, type/source mix,
group-atomic selection, token/step and dynamic-padding budgets, single
candidate/seed rule, teacher-mask criteria, stop conditions and control.
It must then perform an independent complete native-input overlap screen for
the **new selected witness** across all eight core roles, source/semantic
review of remaining rows, rights/use review, and zero-step/one-step native
model parity. This capacity check alone does not establish those conditions.
No optimizer, SELECT/CAL scoring, DEV, formal JevArena, JevBench, package or
Hugging Face upload was run or authorized by this receipt.
