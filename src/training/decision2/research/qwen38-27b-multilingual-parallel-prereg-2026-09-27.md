# Qwen3.8 27B multilingual parallel DEV preregistration

Status: preregistered before this candidate's predictions were generated. This
is a public human-translated validation diagnostic, not a sealed release test.
The same frozen 600 prompt panel already used for the 4B and GLiNER source
comparisons will be reused without edits. It contains 100 independent source
IDs: XNLI Choice 60 IDs × en/ar/de/es/fr/zh and PAWS-X Noul 40 IDs ×
en/de/es/fr/ja/zh. There is no Score item. Translations of a source ID are
paired, not 600 independent samples.

| Frozen component | Identity |
| --- | --- |
| Prompts | SHA-256 `d4f83ba17030ddb1591b1ec7da22ae851859fd2d0e2ce6020997a8ba96ffad39` |
| Targets | SHA-256 `43eee8ac2add99bf19fe8bf09692334ab88f8d5a3bb793aa7b795b8599740b38` |
| Panel manifest | SHA-256 `c9f094c3190b5efe91d29552399982a31595be29f06d5f1e2af28016ec4b754d` |
| Model | Existing Qwen3.8-27B clean-v2 BEST checkpoint `checkpoint-0000368`; source revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; frozen package fingerprint `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2` |
| Selection receipt | `BEST.json` SHA-256 `38b19be1914ef9efd7ee3de7b4d461d9a43bf2a671b151fbb8fa014d9f5832f2`; `COMPLETE.json` SHA-256 `88bad39ec664ebe307f1a0438024ad1e4a65d8969617bbc4e73087d861cf1f1a` |
| Calibration | Existing CAL700 temperatures, file SHA-256 `e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78` |

Run `training.model.infer` with the selected LoRA checkpoint and its pinned
source path, the existing calibration, `--max-length 4096`, temperature 1.0,
one item at a time, native typed answers, no truncation. Use the gold-blind
prompts only during inference. The standalone `multilingual.parallel_score`
verifies target/prompt hashes, model identity, all 600 IDs, and counts invalid
or over-budget answers as wrong. Record prediction and sidecar hashes.

Report correct/60 for XNLI and correct/40 for PAWS-X separately by language,
invalid counts, paired changes versus English on the same source IDs and
versus the existing 4B Decision 1.0 Nox and Eikos clean-v2 results. Do not
average translated rows as independent samples or infer Score transfer. Do
not select a new checkpoint based on this panel; the 27B weights and
calibration remain fixed. A performance failure or runtime error is recorded
rather than hidden or retried with altered rules.

## Frozen run result

The fixed adapter answered **600/600** items valid, with zero over-budget or
truncated items. Inference did not read the targets. The returned model
fingerprint was the preregistered
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`;
the prediction, prediction manifest and standalone score SHA-256 values are
respectively
`84699c8cbd34f84afb601e45051c8f5e17e70d3edfd83253d9a6c6c650eb80e0`,
`3777d83efad4861c28cc459375d4ab028c4a9c36be259fe58369b84574c0b144`
and `99ef32b78a3d06151a235eb4f94a095354b42e4b3688aeffdeca61afc0472af2`.
The scorer files `parallel_score.py`, `score.py` and `parallel.py` had SHA-256
`be9d498c7be699b860a45756c039357bc3ce17fbe7ebb87f9bc83a37c15ae7bc`,
`60da17dc97560c1967cd6c7644dcbe1ed442752637846b7179cef70f6b16fa20`
and `acc4858ee07911cca1517910550e2b4936d1d48da350677f7b7d498de7d70ba5`.

| Human-translated validation | en | ar | de | es | fr | ja | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XNLI Choice, correct/60 | 57 | 48 | 47 | 50 | 50 | N/E | 48 |
| PAWS-X Noul, correct/40 | 33 | N/E | 33 | 30 | 31 | 26 | 28 |

The 27B candidate has 481 correct *translated rows* out of 600; that count is
descriptive only because there are **100 independent source IDs**, not 600.
English-relative losses are nine XNLI items each in Arabic and Chinese and
seven PAWS-X items in Japanese. All models were evaluated on identical IDs:
Decision 1.0 Nox 4B scored XNLI `51/42/48/50/45/47` in en/ar/de/es/fr/zh
and PAWS-X `28/27/27/29/24/29` in en/de/es/fr/ja/zh; Eikos clean-v2 4B
scored XNLI `49/44/47/47/46/48` and PAWS-X `31/32/29/30/25/26` in those
same language orders. Thus 27B exceeds the 4B clean-v2 English XNLI count by
eight, while it ties 4B in Chinese XNLI and gains only one in Japanese
PAWS-X. Exact source-ID paired Nox-vs-27B and Eikos-vs-27B comparison report
SHA-256 values are
`7ccb74da2025ca7e83058d3042e59cb8e17f8997fcd2531715ace41a35f2b4b3`
and `8b5168c9c1d5df525f93f7b8cb6a962f584c7fd374a862c959bafd361acad283`.
The small per-language cells and multiple comparisons do not support a broad
language ranking claim. In particular, this run cannot establish Score
transfer or multilingual JevArena performance.
