# 4B first-release rights and prompt-overlap audit

This is a read-only audit of the candidate's frozen clean-v2 TRAIN against the
gold-free prompts used for the prospective 8,147-item JevArena v3 first release
and the 231-item public JevBench reproduction. It does not inspect evaluation
answers, predictions, or model scores. The detailed receipt is private; this
document contains only aggregate findings and content hashes.

## Frozen inputs and receipt

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| clean-v2 TRAIN | 7,455 | `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| clean-v2 rights manifest | — | `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8` |
| typed FINAL, prompt-only | 1,600 | `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd` |
| CSS15, prompt-only | 6,547 | `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6` |
| JevBench public subset, prompt-only | 231 | `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` |
| candidate package provenance | — | `11bc47f01214bc322165e27adff9a941a9dfc33eec1d20941fc5072d4fa7bd73` |
| audit script | — | `2588ea1adfa2ea1bc2b572b2473aea782e6a4f03735d3f2f9a74679982820e21` |
| private detailed audit receipt | — | `9d66daaac33121e290b45db036cc713fc86a3a9e59577a46ffd1129c2f0cb2a8` |

The script verified each pinned input hash and row count, rejected any prompt
row with fields other than `id`, `state`, and `questions`, and verified that the
candidate package provenance binds the exact TRAIN and rights-manifest hashes.
The receipt has private read permissions and retains matched identifiers if any
are found; this public summary contains none.

## Observed overlap

| Prompt panel | Shared row IDs | Exact state | Normalized state | Approximate near match |
| --- | ---: | ---: | ---: | ---: |
| typed FINAL (1,600) | 0 | 0 | 0 | 0 |
| CSS15 (6,547) | 0 | 0 | 0 | 0 |
| JevBench public subset (231) | 0 | 0 | 0 | 0 |

Exact checks use raw and NFKC/case-folded/whitespace-normalized state hashes.
The near check uses context only: an eight-band, 64-bit SimHash candidate index,
Hamming distance at most eight, relative length difference at most eight
percent, and final string similarity at least 0.94. This is an approximate
screen and can miss paraphrases or shared underlying reasoning templates.
No evaluation gold or candidate predictions were read.

The training set includes 120 FLUTE official TRAIN examples; CSS15 contains
500 FLUTE evaluation items. Their original record IDs have zero intersection.
This is source/task-family overlap, so a FLUTE result still cannot be presented
as independent *cross-source* transfer. The ID finding does not establish that
the base model was not exposed to the source during pretraining.

## Rights scope

The frozen rights manifest marks the exact TRAIN as eligible for a scoped
release of trained weights and model cards. It explicitly excludes publishing
raw source rows, SELECT/CAL rows, and individual text predictions. The TRAIN
provenance is 2,837 internally generated rows, 644 Stage3 replay rows (internal,
CLINC150 CC BY 3.0, and BANKING77 CC BY 4.0), 448 CosmosQA rows (CC BY 4.0),
272 SNLI and 334 SQuAD 2.0 rows (CC BY-SA 4.0), 120 FLUTE rows (AFL 3.0), and
2,800 GoEmotions official TRAIN rows (CC BY 4.0). These sum to 7,455.

Release packaging must attribute all named datasets, retain source licenses
and notices, document uncertainty around CC BY-SA model-weight derivation,
and separately review the base model's license and training lineage. This
audit confirms the candidate's manifest and package linkage, not a legal
determination or absence of undisclosed pretraining overlap.

## Reproduction and decision boundary

The read-only auditor and focused tests are in
`src/training/decision2/training/data/`. It consumes the already-frozen files
and writes only a new private receipt. It does not alter TRAIN, prompts,
labels, predictions, model packages, or score reports. For the pinned inputs,
this audit found no direct train-to-evaluation context leakage. Subsequent
releases with different TRAIN, prompts, or model packages require a fresh audit
and cannot inherit this result.
