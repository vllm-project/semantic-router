# Eos 0.8B FEVEROUS v2: full TRAIN annotation metadata

**Decision: metadata gate passed; student TRAIN and release remain HOLD.** This
is a CPU-only source inventory under the separately frozen
[v2 protocol](eos08-feverous-source-screen-v2-prereg-2026-09-28.md). It is not
a model score, training result, source-disjoint transfer proof, or permission to
download the 9.9 GB Wikipedia dump. No GPU, optimizer, model inference,
protected answer key or publisher dev/test split was used. The strict v1
empty-label failure remains recorded [separately](eos08-feverous-source-screen-v1-hold-2026-09-28.md).

The publisher [Zenodo TRAIN annotation file](https://zenodo.org/records/4911508)
matched its 175,493,294-byte size and published MD5
`d8d4634760dad714b4cc30e43d25e589`. Local SHA-256 is
`7fd77957a43e03ff22644d8e14ad0755b0246a9848c74e8a042b015eef92a9a0`.
All 71,292 complete records were read. The one empty-label placeholder also
has a nonnumeric ID; v2 explicitly quarantines it **before** labeled-row ID
validation. All 71,291 remaining records have unique numeric IDs and one of
the three publisher labels. This ordering is a faithful implementation of the
predeclared quarantine, not another policy amendment. There is one repeated
normalized claim among labeled records; candidate selection keeps at most one.

| Publisher relation | Labeled TRAIN rows | Rows with a complete sentence-only evidence set | Distinct referenced pages for those sets | Deterministic candidate roster |
| --- | ---: | ---: | ---: | ---: |
| REFUTES | 27,215 | 13,210 | 10,808 | 64 |
| NOT ENOUGH INFO | 2,241 | 1,065 | 1,249 | 64 |
| SUPPORTS | 41,835 | 16,560 | 13,194 | 64 |

The text-only candidate pool has 15,294 page-linked components under shared
referenced-page union. A fixed SHA-256 ordering of source IDs yielded a
balanced 192-record roster with no repeated normalized claim or shared
referenced page. Its ordered ID-list SHA-256 is
`1b66aa1e17f59201fb4cc7650e4d913172c2e0f4c7b4d3fa8e089e001b5c54e6`;
the IDs themselves remain private. The provisional metadata floor therefore
passes. The audit program emits aggregates and hashes only:
[`audit_feverous_train_metadata.py`](../training/data/audit_feverous_train_metadata.py).

**Admission remains HOLD** because these are *references* to sentences, not
the full publisher page text. We have not measured native Eos tokens, checked
whether all required evidence lies within an unhighlighted ≤8,192-token
rendering, or met the fixed ≥155,681 total replacement-token gate. We have not
resolved whether FEVEROUS's NEI verdict remains valid under the finite pages
we would supply. More fundamentally, `REFUTES / NOT ENOUGH INFO / SUPPORTS`
are categorical logical relations; placing NEI between refutation and support
on a numerical `Score` axis is a **training hypothesis**, not a validated
ordinal measurement. A fixed signed-evidence rubric, including what expected
level and distance-sensitive losses mean, and an independent answer-blind
source-page review protocol must be committed before even sparse page-body
acquisition; the actual review must pass before training. Original article-specific
Wikipedia rights, page/claim
overlap with control and protected sets, 24-row blind semantic review, and
claim-only/evidence-deletion shortcuts are also pending. FEVEROUS and
VitaminC both descend from Wikipedia; distinct dataset names do not establish
independent source identity. A 1 MiB ZIP tail inspection showed 544 Wikipedia
JSONL members but opened **no page bodies**; it does not count as a long-input
or rights check. The next decision is to accept or reject the native rubric
with an answer-blind protocol, not GPU training or a 0.8B release claim.
