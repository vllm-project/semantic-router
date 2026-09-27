# ANLI as an open three-level Score development diagnostic

**Decision: bounded lexical screen passed; ANLI is eligible only as an open
development diagnostic.** This is a CPU-only source and prompt screen. It is
not a model evaluation, training-data admission, blind release result, or
JevArena score. GPU use: **0 GPU-hours**.

The candidate is the three public development rounds of
[`facebook/anli`](https://huggingface.co/datasets/facebook/anli/blob/main/README.md),
revision `8e4813d81f46d313dac7892e1c28076917cfcdf9`, licensed CC BY-NC
4.0. Its publisher defines labels `0 = entailment`, `1 = neutral`, and
`2 = contradiction`. The proposed native System One `Score` rubric reverses
the numerical order to `0 = evidence contradicts`, `1 = insufficient evidence`,
and `2 = evidence supports`. The original dev labels were counted in aggregate
for this audit; they were never placed in the constructed model input. The
publisher's explanation field and row IDs are also absent from model inputs.

| Round | Rows | Contradiction / neutral / entailment | Normalized premise groups | Complete native Score prompt tokens, median / p90 / p99 / max |
| --- | ---: | ---: | ---: | ---: |
| R1 | 1,000 | 333 / 333 / 334 | 844 | 224 / 248 / 277 / 324 |
| R2 | 1,000 | 333 / 333 / 334 | 872 | 222 / 246 / 265 / 340 |
| R3 | 1,200 | 396 / 402 / 402 | 1,127 | 207 / 253 / 292 / 336 |

Across all rounds there are 3,200 rows and 2,842 distinct normalized premise
groups; one premise group crosses rounds. There are no repeated normalized
premise/hypothesis pairs and no contradictory labels assigned to such a pair.
All 3,200 rows contain a nonempty publisher explanation, which this diagnostic
does not use. The complete native Score requests were constructed with the
existing System One adapter and segmented option encoder, using the pinned
official Qwen3.8-27B tokenizer. No prompt exceeds 4,096 tokens and no input
was truncated. This is a useful **short evidence-relation** stress test, not a
long-context test. Confidence intervals should resample premise groups rather
than treating repeated hypotheses over a premise as independent.

The three source parquet SHA-256 values, in R1/R2/R3 order, are
`72e27463177b4363be80f1fc6ccdaab44ddaeb65db58c2280f94690e15468334`,
`43e4673665decf0b0e8487e55f98285423cb356b985e206fe5998defae2e38fa`,
and `61775ec09351f6011ce4dc9ea313f457bba6e11d7665d34d95c111665023a83e`.
The tokenizer is pinned to official revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; the audit receipt records
individual tokenizer-file hashes. The initial CPU receipt, before a complete
protected inventory was available, had SHA-256
`38f176a5d7bf70c7835c9d6d8209e74cfedfc2985c937a58addf2c8e82701fee`
and correctly returned `HOLD_MISSING_PROTECTED_INVENTORY`. The later
aggregate-only overlap receipt has SHA-256
`05e9f0dd4c9754d8c44d404096b5e10794959d1f82b2fad47e0b9b7231bbe49f`.
Neither receipt or this note contains source text, row-level IDs, targets,
explanations, predictions, or private infrastructure.

## Why the source is informative, and what it cannot prove

ANLI was collected through adversarial human/model interaction. The
[original paper](https://aclanthology.org/2020.acl-main.441.pdf) describes R1
and R2 Wikipedia contexts sampled through HotpotQA; R3 also draws on news,
fiction, spoken transcripts, WikiHow, RTE5 and Wikipedia. This diversity makes
it a plausible **cross-source NLI** check after SNLI/OCNLI development. It
does not establish complete corpus independence: R1/R2 reuse HotpotQA source
passages, R3 has several upstream text sources, and training or pretraining
material may reuse those passages. A lexical no-match would not prove semantic
or original-corpus disjointness. The public development labels also prevent
any claim that a later score is an untouched test.

The mapping itself requires a blinded semantic review of neutral examples:
`neutral` means neither entailment nor contradiction under the NLI annotation
scheme, while a System One user may interpret “insufficient evidence” somewhat
differently. This source tests one three-level evidence relation, not rule
priority, state update, arbitrary Score rubrics, multilingual transfer, or
open-ended response quality. A development comparison must report each round,
three-class accuracy/macro-F1, class confusion and calibration separately.

## Eight-role input-only overlap screen

A later, versioned **strict input-only** protected inventory was supplied and
verified at manifest SHA-256
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
The loader checked the manifest schema, every relative role-file hash, exact
role row count and projected input-only row schema before comparison. The
screen covered eight required core roles and 20,263 role rows in aggregate;
some roles may represent related material, so this is not an independent
question count.

| Reference role | Projected input rows | Exact raw / normalized / bounded near / same ID matches |
| --- | ---: | ---: |
| Rights-clean TRAIN | 7,455 | 0 / 0 / 0 / 0 |
| Rights-clean SELECT | 700 | 0 / 0 / 0 / 0 |
| Rights-clean CAL | 700 | 0 / 0 / 0 / 0 |
| Typed DEV | 1,600 | 0 / 0 / 0 / 0 |
| Typed FINAL, input only | 1,600 | 0 / 0 / 0 / 0 |
| Human-transfer pilot, input only | 1,430 | 0 / 0 / 0 / 0 |
| Human-transfer final, input only | 6,547 | 0 / 0 / 0 / 0 |
| Public supplemental prompts | 231 | 0 / 0 / 0 / 0 |

The comparison used only ANLI premise/hypothesis source text and all
projected reference input fields; publisher labels and explanations, and
protected answer keys, were not inputs to the overlap algorithm. Exact
matching considers input spans of at least 20 normalized characters. The
bounded near method uses 64-bit SimHash candidates and confirms similar
strings with Hamming distance at most 8, relative length difference at most
8%, and SequenceMatcher ratio at least 0.94. It is not an exhaustive semantic
or paraphrase detector. Twenty-seven unreviewed optional inventory roles were
excluded. A zero among the eight core roles cannot certify source-corpus
independence, especially given the HotpotQA and other publisher context
sources described above.

The next bounded action is a blinded rubric-alignment review, particularly for
`neutral`, before relying on the mapped Score diagnostic. Any subsequent model
scores remain **open development** results. ANLI must never be folded into the
sealed JevArena main score or labeled an independent untouched release test.

Reproduction: `training/data/audit_anli_score_diagnostic.py` and its synthetic
contract tests. The script checks pinned source and tokenizer bytes, native
prompt lengths, grouped source statistics and strict protected-role presence,
hashes and input schema; it runs no model and requires no GPU.
