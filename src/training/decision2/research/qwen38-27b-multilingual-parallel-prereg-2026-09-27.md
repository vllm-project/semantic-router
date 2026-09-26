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
