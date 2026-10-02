# IB4 phase 1: licence-clean rows for Index gap families (`m6/ib4/p1/`)

> **Status: RELEASE-SAFE PENDING C1.** Every data audit passed (see `status.json`). The custodian's C1 content recheck
> has been requested; until it passes, do not train a C1-scored model on these rows.

Four families, native Decision 2.0 rows (TRAIN 9,459; DEV 782, role `select`):

| Family | Task | Source (licence) | TRAIN | DEV |
| --- | --- | --- | ---: | ---: |
| `sqa2` | Noul: does the passage support the proposed answer to the question? | SQuAD 2.0 train (CC BY-SA 4.0) | 3,694 | 398 |
| `isarc2` | Noul: is the author being sarcastic? (en + ar) — **in-distribution** | iSarcasmEval train (MIT) | 2,636 | 282 |
| `sentfin3` | Choice-3: headline sentiment toward the named entity (negative / neutral / positive) | SEntFiN 1.0 (MIT) | 2,661 | 102 |
| `fc_pick` | Noul: should the assistant call the candidate function for this request? | Glaive function-calling v2 (Apache-2.0) | 468 | 0 |

- **Balance.** Yes = no in every Noul family; one row per class in `sentfin3`. Labels are also balanced inside each
  declared cell: per question in `sqa2` (gold answer vs another answer from the same passage), per language and length
  in `isarc2`, per entity in `sentfin3` and per candidate name in `fc_pick`. So the answer, the entity or the function
  name alone does not predict the label (G4 checked).
- **`sentfin3` replaces IB1 `sentfin`.** An arm that mixes `sentfin3` must drop IB1 `sentfin`, which reused the same
  headlines with two classes.
- **In-distribution.** `isarc2` comes from the train split of an Index benchmark's own source (the test files are
  never read). Keep it separable by `family`.
- **Index isolation.** No row of the Decision Index suite is included. Every group with an exact, normalized or 13-gram
  match, or a shared URL or host, was dropped, and the guards' positive controls passed. A row-level contamination
  audit against the full private Index panel found no duplicate or partial row (planted control 200 / 200). No
  Jev-derived repository is a source.
- **Dedupe.** Rows whose state duplicates an IB1-r3, IB2 or IB3-r2 row were removed before balancing, except for the
  replaced IB1 `sentfin` family.

Licences and attribution are in `license-registry-ib4.json`. `sqa2` contains Wikipedia text (CC BY-SA); share-alike
terms apply to that family. The preregistration and its amendments are in `v2/data/records/ib4-prereg-2026-10-02.md`
(branch `xunzhuo/decision-2-training-data-ib4`).
