# Decision Index 0.2.1 input rebuild audit

Audit date: 2026-09-27 UTC. This note identifies the input files and row
selection needed for an independent **0.2.1** run. It is not a Decision 2.0
score or a claim of organizer acceptance. Keep source rows and model results
private under their upstream terms.

## Pinned evidence and physical input files

| Item | Identity and count | Status |
| --- | --- | --- |
| [Reproduction kit](https://github.com/apolinario/decision-index/tree/19ad28ec9485493cc4f7fc07d91c178f948e6434) | Git `19ad28ec9485493cc4f7fc07d91c178f948e6434`; supports 0.1 and 0.2, not 0.2.1 | Read and inspected |
| 0.2 `selected-rows.jsonl.gz` | 124,971 physical base rows; lab gzip SHA-256 `25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656`; **uncompressed** SHA-256 `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` | Exact kit rebuild target; 0.2.1 Space declares the same base corpus SHA |
| 0.2 `added-rows.jsonl.gz` | 30,419 physical rows for seven added benchmarks; **uncompressed** SHA-256 `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` | Exact kit rebuild target; the Space says 0.2.1 inherits 0.2's seven added benchmarks, but does not independently publish their row-file hash |
| Common exclusions | 442 run IDs; `excluded-questions.json` SHA-256 `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` | Unchanged across editions; apply at scoring, not by editing source rows |
| 0.2 subset files | ACOS `3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45`; ToolRet/BRIGHT `9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df` | Shipped in the kit and hash checked |
| Current [0.2.1 Space bundle](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4) | Git `ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4`; `index-v0.2.1.json` SHA-256 `5444deeacd9bd6ea9e8ccf008f99f1223fe1d43af6e40739259aec804284ec55`; `methodology-v0.2.1.json` SHA-256 `235384612203690889a4d82ff22dab16fd3c65a6c8f5db0ef28b361f9ed9f665` | Read via HF CLI at the pinned revision |

The kit's [0.2 manifest](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/hub/0.2/manifest.json)
and [`editions.py`](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/editions.py)
provide the row and subset hashes. Gzip headers can vary after rebuilding;
verify the **uncompressed** hashes. The 0.2.1 Space's `suite.corpus_sha256`
refers to the base file alone, not a combined base-plus-added digest.

### Complete run denominator

The Space reports **120,340 scheduled / 119,898 scoreable base requests**.
Its `suite.added.requests` separately reports **30,419** requests for the
seven benchmarks included in the index. The complete 0.2.1 run therefore
needs **150,759 scheduled / 150,317 scoreable requests**. This matches the
kit's [`pipeline.py`](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/pipeline.py)
completion rule: `base_scoreable + added_requests` (0.2: 120,615 + 30,419).
The 0.2.1 delta of 717 applies to the base requests only. A run of just
119,898 requests would omit seven headline benchmarks and is incomplete.

## Reconstructing the 717 removed base requests

Build and hash-verify both 0.2 row files first. For ToolRet and BRIGHT, apply
the common 442 exclusions and the kit's ACOS subset, then group retrieval rows
by `_evaluation.group_id`. Their `scoring.qrels` and `scoring.retrieved_ids`
are part of the frozen row payload. Mark a group answerable iff any retrieved
candidate has `qrels.get(candidate_id, 0) > 0`. Keep every chunk of an
answerable group and discard every chunk of an unanswerable group. The 0.2.1
methodology requires **685/1,000 ToolRet** and **220/550 BRIGHT** scored
groups, removing 315 and 330 base requests respectively. Record and hash the
sorted removed `run_id` lists; reject the port if counts or group completeness
disagree. Candidate recall and the conditional ranking score must both remain
visible, because the latter alone is not end-to-end retrieval performance.

The kit's deterministic [Home appliance generator](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/suite/build/home_appliance.py)
creates 80 dev and 160 test rows. An independent CPU audit of the pinned
generator found that canonical `state` equality partitions test into 136
states: exactly **24 pairs** of duplicate test states (one copy removed per
pair). Another **48 singleton** test states equal a dev state; these do not
overlap the 24 pairs. This yields exactly **88** retained test rows as stated
in the 0.2.1 methodology. Canonical JSON means sorted keys and compact
separators as in that generator's `canon` function.

**Exact-row blocker:** the public 0.2.1 Space says to retain one copy of each
Home test pair but does not publish the retained `run_id`s, a list hash, or a
deterministic first/last-copy rule. The duplicate states' question and option
payloads are not byte identical; selecting the other copy can change a
model's predictions and score. A port may use first-in-frozen-file as a
*provisional* rule and compare published benchmark metrics, but must not
claim exact organizer parity until the chosen IDs are confirmed by the
maintainer or equivalent row-level evidence. The same rule should be frozen
before evaluating Decision 2.0.

## Acquisition and access preflight

The kit [quickstart](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/README.md)
rebuilds the source suite from Git revisions, HF dataset revisions, and
content-hashed HTTP releases. Budget about **7 GB of downloads and 17 GB of
workspace**; allow more for model results, caches, and repeated attempts. No
selected/added row files were found in the two authorized experiment nodes'
task directories during this audit, and the published 0.2 suite dataset ID
was not accessible, so do not assume a ready shared copy. The 0.2 kit's
`suite rebuild` then `suite import` is the documented route.

The [source-by-source licence table](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/docs/suite.md)
is authoritative for the 43 upstream datasets and purpose-built cases.
Access and handling points relevant to a private noncommercial evaluation:

- HLE is gated. A **dry-run** with an authenticated HF CLI at pinned revision
  `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29` resolved its 274.3 MB test
  parquet without downloading it. Another machine or credential still needs
  the same terms/access check.
- RAGTruth includes MS MARCO noncommercial contexts and Yelp contexts that
  forbid redistribution; its built rows and predictions stay private.
- GPQA asks that questions not be posted in plain text; BBH carries the
  BIG-bench training canary. HoVer/Wikipedia and SGD carry share-alike terms;
  ANLI is CC BY-NC 4.0; POP909's underlying data is research-use.
- ARC is requested without an explicit Hub revision in the kit, but final
  output-file hashes are fixed. If upstream changes, the rebuild must fail
  rather than silently using moved input. HTTP acquisitions and the major Git
  sources have pinned content hashes or commits in the kit.

For a private rebuild on an authorized SSH node, keep credentials in an
environment variable without command-line token arguments or log printing:

```sh
# Run inside the pinned kit checkout and an authenticated private environment.
export HF_HUB_DISABLE_XET=1
python -m decision_index suite rebuild --edition 0.2 --work "$PRIVATE_WORK"
python -m decision_index suite import \
  --edition 0.2 \
  --rows "$PRIVATE_WORK/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
  --added-rows "$PRIVATE_WORK/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz" \
  --dir "$PRIVATE_SUITE"
python -m decision_index suite verify --edition 0.2 --dir "$PRIVATE_SUITE"
```

Before any 0.2.1 model run, require: both uncompressed row hashes above,
exclusion and subset hashes, a 717-ID filter receipt including the 24
retained Home IDs, exact 150,317 scoreable-count assertion, no source/train
overlap, and a pinned scorer that replays published 0.2.1 per-benchmark,
area, and headline values at display precision. The *published aggregates*
alone cannot resolve the Home retained IDs or establish row-level inference
parity. A full model run must include the seven added benchmarks and preserve
native Choice/Noul decision outputs; Score is not exercised by this external
index.
