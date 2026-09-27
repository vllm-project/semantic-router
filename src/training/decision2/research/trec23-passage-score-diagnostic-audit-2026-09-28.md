# TREC DL 2023 passage judgments: Score diagnostic source gate

**Disposition: HOLD; CPU source screen only.** This is a possible public-source
evaluation diagnostic for ordinal Score, not training material, a sealed test,
or a substitute for Decision 2.0's native Choice/Noul/Score panel. No model
predictions or evaluation scores were read or produced in this audit.

## Publisher semantics and pinned inputs

[NIST's 2023 DL track page](https://trec.nist.gov/data/deep2023.html)
defines passage labels as 0 not relevant, 1 related but **not answering**,
2 highly relevant, and 3 perfect. In binary passage retrieval, level 1 is
nonrelevant; it must not be quietly mapped to a positive answer. Unjudged
passages have no label and cannot be supplied as grade 0. These labels test
*answer relevance*, not general quality or severity. The [publisher corpus
specification](https://github.com/microsoft/msmarco/blob/master/TREC-Deep-Learning.md)
ties passage IDs to byte offsets in the MS MARCO v2 passage corpus; native
four-level Score rendering is therefore feasible without treating the task as
chat generation. Four ordered criteria are accepted by the [System One API](https://docs.typesafe.ai/api).

| Publisher input | Private pinned SHA-256 | Screen |
| --- | --- | --- |
| 2023 passage qrels with near-duplicate propagation | `5a98a95d5714b00c5066593719e5eb7ca93dd13588d0facf7f426a4c9d6b19b8` | 22,327 judged rows / 82 queries |
| 2023 query text | `56b763823a2a9027b136f42dc5d9c429112198b40c075b1c5df63672e8e3cb4b` | 700 publisher queries; only 82 judged |
| NIST duplicate equivalence classes | `57ddba8918693c8532e220932f4cd9b21ad9c384341fc28ff8b9eba5df3fe32d` | Complete ID-to-representative scan |

Qrels grades 0/1/2/3 have 13,866/4,372/2,259/1,830 rows. The official
duplicate classes collapse these to **18,878 query × near-duplicate-class
groups**; 3,449 qrels rows are propagated copies. Eight groups (16 rows)
have conflicting grades. A future diagnostic should quarantine the entire
eight groups before any metric, freeze that exclusion rule, and compute
uncertainty over the 82 independent query groups. It should report ordinal
error and per-grade accuracy because an overall exact-match score is
dominated by levels 0 and 1. The present audit does not resolve those eight
human-judgment conflicts or manufacture consensus grades.

## Text, rights, overlap, and limits

The [MS MARCO terms](https://github.com/microsoft/msmarco#terms-and-conditions)
permit noncommercial research use while granting no underlying document IP
license. Exact passage text remains in a private experiment directory and
must not be copied into the public repository, gist, model card, or dataset
download. This public benchmark may appear in model pretraining, so
source-disjointness from our finetuning records alone cannot establish a
never-seen test. It remains supplemental.

`training/data/audit_trec23_score.py` fail-closes on the three publisher
hashes, parses the 70-member TAR by HTTP byte ranges, verifies gzip member
passage IDs against uncompressed byte offsets, joins judged queries and
passages, and renders ordered Score criteria through the local native
segmented-option encoder. The protected-role scan reads only prompt fields
from the exact pinned inventory; it never reads answer keys. The known
inventory hash is `c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2`.
The protected manifest path is not verified in this task, so overlap remains
**HOLD**. The existing lexical near screen also omits protected leaves over
800 characters; a full-length/semantic review would still be needed if a
candidate were considered for a benchmark.

The bounded exact-text screen used shards 00, 10, 20, and 40, whose publisher
gzip member bytes were pinned in the private receipt. All **1,604** judged
rows in those members joined to exact passage IDs and uncompressed byte
offsets, spanning all 82 query groups, 1,178 originating documents, and
grades 0/1/2/3 = 1,000/310/149/145. This is only **7.2%** of the qrels and
does not define an evaluation set. Passage length in characters was median
303, p90 448, p99 13,774, maximum 178,458; a few very long records could
exceed small-model context limits. Native token length and overflow require
an exact tokenizer revision on the same private CPU run, with no truncation.
None was pinned in this audit. None of the 82 judged queries contained
non-ASCII characters, while 435 joined passages did; ASCII is not a language
identifier. This source supplies no multilingual diagnostic claim.

## Decision

Do not add this source to training or JevArena v3/v3.1. It can be revisited as
an external Score diagnostic only after exact full text join, conflicting
class exclusion, pinned protected-prompt overlap including long leaves,
source rights and evaluation custody, and native length checks are complete.
Never assign model scores to the bounded four-shard preflight. All work here
used CPU only and zero GPU-hours.
