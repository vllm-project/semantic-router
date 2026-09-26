# Three-level Score SELECT r1: frozen blind-review candidate

**Status: frozen, unreviewed, BLOCK_FOR_MODEL_INFERENCE.** This is a
checkpoint-selection diagnostic. It is not TRAIN data, a release benchmark, or
evidence that any Decision 2.0 model improved. Neither an independent review,
key comparison, nor model run had occurred when this note was signed.

The prospective method is
`score-three-level-select-method-2026-09-27.md` (SHA-256
`9bfc968c9eef41fc1aee0938928d8b59bbf7dab76618aba18acf47dc43547ec6`).
The exact source commit used for the private freeze is
`f540067f84f9c3a339e1bf8d5e36bab607be76b4`. The frozen source contains a
separate mechanical oracle and a builder that emits an isolated, gold-free
reviewer packet. Case facts, source records, salt, answer key, and joins remain
private.

| Freeze check | Result |
| --- | ---: |
| Independently generated source groups | 80 |
| Counterfactual rows | 240; one each at levels 0, 1 and 2 per group |
| Operations | waiver precedence; inclusive interval coverage; independent evidence quorum; allocation caps |
| Source groups per operation | 16 English, 4 Chinese |
| Protected reference roles / prompt rows | 29 / 43,070 |
| Exact ID, group, input, raw-state and normalized-state matches | 0 in every category |
| Bounded near-state matches | 0 |
| Pinned model tokenizer maximum | 252 of 1,024 tokens |
| Native Score option conversion | 240 of 240 preserved ordered levels 0, 1, 2 |
| Reviewer packet shape | 240 gold-free rows, 80 opaque group IDs |

The protected inventory includes frozen parent TRAIN/SELECT/CAL prompt fields,
the three serialized attempted Score curricula, earlier gold-free decision and
human-transfer rosters, the full Score v3 gold-free projection, multilingual
pilots, and the newest authored editorial packets. The Score v4 and v5 attempts
stopped before a candidate corpus; Score v6 is design only. The overlap check
looks for exact and bounded near matches. It cannot prove semantic independence.

The tokenizer is the pinned Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded with Transformers
4.57.1. Source records are generated from a private random seed under four
new rule mechanisms. This is a small structured diagnostic; repeated rule
wrappers, simple arithmetic, and artificial source prose may limit transfer.
No Chinese-transfer claim is supported before qualified bilingual review.

The private freeze completed at `2026-09-26T22:00:20.944590+00:00`. Publicly
shareable digest commitments are:

| Artifact | SHA-256 |
| --- | --- |
| Freeze manifest | `cbd8175786230bbc9574a9414bcf5b3b5d7294b1fa083ccb9be1790c78eb5023` |
| Gold-free reviewer packet | `d38702cee5ef1b50458a4ee11d4370a7fda44013321b2dac43b40d01600ea88a` |
| Reviewer manifest | `a9868c0e8562c4450a2be57556327bdaba84719713cfbc77d8227a5ecff6ef59` |
| Separate freeze receipt | `447615d762327f29babecef9c7847ecc463fac81e19963bdb4c480b236b84309` |

An independent reviewer must solve all 240 rows and inspect each complete
triplet **without access to the source casebook, oracle, key or join**. The
reviewer must seal row judgments and shortcut findings before post-key
comparison. One material ambiguity, mistranslation, answer mismatch or group
shortcut blocks r1; a repair gets a new version. Until then no model inference,
checkpoint selection, training, or publication may use this diagnostic.
