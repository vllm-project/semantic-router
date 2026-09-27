# 0.6B ContractNLI long-input triage: population gate stopped inference

**Verdict: STOP before GPU inference.** The prospective protocol is
[`small06-contractnli-long-transfer-prereg-2026-09-27.md`](small06-contractnli-long-transfer-prereg-2026-09-27.md).
Its minimum of 20 independent documents whose complete requests exceed 2,048
native tokens for **both** untouched source models was missed by one document.
There was no optimizer step, model prediction, label score, prompt selection on
model outputs, or protected JevArena/JevBench evaluation. Measured GPU use for
this screen was **0 GPU-hours**. It establishes no model-quality comparison and
does not advance or release a 0.6B checkpoint.

The source is the authors' [ContractNLI development
split](https://stanfordnlp.github.io/contract-nli/) under CC BY 4.0. The
original archive SHA-256 is
`e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b`;
its `dev.json` SHA-256 is
`310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e`.
The 61 original development documents each have up to 17 fixed hypotheses.
No train or test document was used. The private archive, extracted source and
audit receipts remain out of the public repository.

Before opening the development annotations for class-balanced selection, a
source-overlap scanner verified exact rights-clean v2 TRAIN/SELECT/CAL and
source-manifest hashes, then checked source IDs, URLs, normalized whole text and
200-character fragments across 745 existing candidate-data and experiment
JSONL files, or 772,994 records. It found zero suspected collisions and zero
parse errors. This is a finite text-overlap check, not a guarantee about
unobserved pretraining or semantic paraphrases. The private overlap receipt
SHA-256 is
`d2d5ebb2bf5a3ff69da14bc198edd2dd5435a9565729f7643a611d1105a69a20`.

Both source tokenizers measured every one of the 1,037 document-hypothesis
requests without truncation. The untouched multilingual GLiNER2.5 model is
`fastino/GLiNER2.5-multi-Decide@6bc1d43d201b0691e733626389af8c57eea3ea68`
(287,355,159 loaded parameters); its native length receipt SHA-256 is
`9c5e48e809839a11142a57193e91a55910b0fe969f659e80caf67ed47f3ba7b8`.
The untouched reranker is
`Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473`
(595,776,512 loaded parameters); its native length receipt SHA-256 is
`d01fc9292c169ae985e3a070a66c9fd31e3c36c25e616e5ce81a3e82e5e4b25e`.
The source weight identities and adapter versions are captured in these private
receipts and checked by the versioned preparation code. The later model-identity
receipts have SHA-256 values
`08f10b816b6b6fa20e147b4c7f1027f8ca602b1b3ca5d0084e6c1cfcda6e224c`
and
`dd0a625d99e9c812250d41a50fdd848aae508621d0e4106bada5a02463a20e5b`.
The CPU tokenizer images were pinned by IDs
`sha256:6a0c206a5ddf443d0982d8561ad0c6f0dd2ae42f41e5c02b26216bd98c23d37c`
and
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

| Preregistered admission view | Count |
| --- | ---: |
| Official development documents | 61 |
| Complete document-hypothesis requests measured in both adapters | 1,037 |
| Requests strictly above 1,024 and at most 4,096 native positions in both | 714 |
| Independent documents represented after frozen class-capped selection | 42 |
| Selected questions, at most one per class per document | 122 |
| Entailed / contradicted / not mentioned selected questions | 42 / 38 / 42 |
| Independent documents above 2,048 in both adapters | **19 / required 20** |

Checking **all** 714 length-eligible hypotheses still yields only 19 such long
documents, so choosing a different hypothesis under the same protocol would
not clear the gate. The frozen private population receipt SHA-256 is
`9933fe29ff66cd2952b3b3997fa28aade7e035907941a64a24f5050c32b8b874`.
It records `population_pass=false`; the preparation program wrote neither
model prompt nor gold files. Development labels were accessed solely to count
the preregistered classes. No model saw those labels or ran on this panel.

The preregistered source-quality, recall and paired macro-F1 gates remain
**unmeasured**, so a GLiNER long-structured training pilot is not authorized by
this experiment. A future test requires a separate prospective protocol and a
source-disjoint collection with enough independently labeled full documents;
the existing length floor and 4,096-token ceiling were not relaxed after this
near miss. The executed preparation source had final SHA-256
`7287366857d1de121998c9ea8ab635353fabe21124936af3f04debcd836578a8` and is
[`contractnli_long_transfer.py`](contractnli_long_transfer.py); the two focused
admission tests and repository checks passed locally.
