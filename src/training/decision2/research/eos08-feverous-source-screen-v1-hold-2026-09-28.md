# Eos 0.8B: FEVEROUS range screen v1 result

**Decision: v1 HOLD.** The frozen [small source screen](eos08-feverous-source-screen-prereg-2026-09-28.md)
encountered one complete publisher TRAIN row with an empty label. Its strict
source-label gate was frozen before parsing the range sample, so the apparent
three-label and text-evidence coverage cannot be promoted to v1 admission.
This is a source-only CPU finding, not a new Decision model score. No GPU,
optimizer, protected key, model prediction or full Wikipedia archive was used.

The publisher [Zenodo TRAIN file](https://zenodo.org/records/4911508) reports
175,493,294 bytes and MD5 `d8d4634760dad714b4cc30e43d25e589`. We read
four deterministic 1 MiB HTTPS ranges with verified 206 and exact
`Content-Range`; the four private byte-range SHA-256 values, in offset order,
are:

1. `fb9a5e4e2adbd2dd6c222a0412b87442298ce01391f4b5df1cd29b9d799d7473`
2. `4e58e805d6b9bec416caaf61359fa8caafa2fcfdbbe1954a9df1154ea35cc1ee`
3. `d01f48ab9d136c3a6d1b4ee97563000135f336974132a89eb3e2ea090f0ceec0`
4. `392c9fff061193e3e4c22b706414e34185f37418db688b39d0922e8baafa4c19`

There were 1,680 complete JSONL rows after dropping boundary fragments, of
which 1,679 had one of the three declared labels and one had an empty label.
No source text, row IDs or page titles leave the private range files.

| Publisher label | All labeled sampled rows | At least one complete sentence-only evidence set | Distinct referenced pages in sentence-only sets |
| --- | ---: | ---: | ---: |
| REFUTES | 626 | 282 | 291 |
| NOT ENOUGH INFO | 57 | 27 | 29 |
| SUPPORTS | 996 | 391 | 472 |

The 700 sentence-only eligible rows are **41.7% of the labeled range sample**;
the 27 text-only NEI rows meet the predeclared provisional count/page floors.
This is a fixed four-quartile byte-range probe, not a random unbiased estimate
of the full source. Element-kind record occurrences were sentence 1,179,
cell 1,001, header-cell 89, caption 80 and list item 25; a record can have
multiple kinds. Dropping a table-bearing evidence requirement would change
the task and is prohibited.

This result does not establish any full-document native length, the required
155,681 replacement-token exposure, 192 source-disjoint complete page groups,
NEI meaning under finite displayed pages, article-specific redistribution
rights, or protected-set disjointness. Training and release remain HOLD.
Because the unlabeled row is a source anomaly, a **separately versioned** CPU
screen may predeclare its quarantine and rerun metadata accounting; v1 must
stay recorded as a failure. No GPU experiment is warranted from this receipt.

Reproduce aggregates with
[`audit_feverous_score_ranges.py`](../training/data/audit_feverous_score_ranges.py)
on the pinned private ranges; synthetic-only unit tests are
[`test_audit_feverous_score_ranges.py`](../training/data/tests/test_audit_feverous_score_ranges.py).
