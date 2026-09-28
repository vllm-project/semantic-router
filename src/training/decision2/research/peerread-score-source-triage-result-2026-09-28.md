# PeerRead ordinal Score source triage v1: HOLD

The [prospective protocol](peerread-score-source-triage-prereg-2026-09-28.md)
was run as a CPU-only publisher TRAIN inventory. The checked-out publisher
revision was `9bb37751781a900cee9e74ec3105997732c8e8e5`; the exact local and
remote audit script SHA-256 was
`e9569fb978ac7d282a38b150e49829cf9707e05ac0985b5b86141a4ac5715757`.
No publisher DEV/TEST review records, Decision protected labels, model outputs,
or training job were opened. **Zero rows admitted; zero GPU-hours.**

| Publisher TRAIN section | Paper review files | Reviews with text and strict numeric `RECOMMENDATION` | Papers with those reviews | Scale observed | Section license file |
| --- | ---: | ---: | ---: | --- | --- |
| ACL 2017 | 123 | 248 | 123 | 1–5 | Present, CC BY 4.0 text |
| CoNLL 2016 | 19 | 33 | 19 | 1–5 | Present, CC BY 4.0 text |
| ICLR 2017 | 349 | 2,166 unfiltered | 349 | 1–10 | **Absent** |

The ACL and CoNLL sections have 281 numerical review ratings in aggregate,
below the frozen 400-review source floor even before paper/source overlap and
evidence-quality checks. The ICLR count is **not** an official-review count:
its TRAIN files include 5,940 review/comment records, and this first inventory
has not separated assigned reviews, meta-reviews, public comments and later
annotations. Its section has no license file in the publisher repository;
the publisher explicitly warns that venue sections have different terms. It
cannot pass the frozen section-specific use/retention gate on this evidence.

The paper reports that annotators could infer only about half of aspect scores
from review text in their feasibility sample. Therefore even the sections with
numeric labels require answer-blind evidence-sufficiency review before a
faithful native Score mapping. The current HOLD is a source-quality/rights
decision, not evidence that PeerRead ratings would fail in a future model.

The aggregate-only receipt SHA-256 is
`e1d93b35f399f1ff2266e0ed5a35e1116a09b1756d0829fc05de91d4bf5c9da6`.
Exact aggregate TRAIN byte hashes are retained in that receipt;
the audit program emits no review text, paper IDs, annotator IDs or individual
rating records. Do not relax the 400-review floor or combine 1–5 and 1–10
scales to make this v1 screen pass. A distinct prospective arm would need
section-specific rights, official-review identity, and an independently
reviewed Score rubric before any model training.
