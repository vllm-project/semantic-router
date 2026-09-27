# Own-Sol 2B unmerged package: development readout

Status: **the preregistered development advancement gate passed; release is
not established**. This is a continuation of the prospective [package
preflight](sol2b-own-source-adapter-package-preflight-2026-09-27.md), not a new
training run or a selection among checkpoints. No protected typed FINAL or
15-task transfer labels were accessed in this experiment. The formal JevArena
v3 comparison and a same-panel near-size open-model peer remain outstanding.

## Fixed model and runtime

| Item | Fixed identity |
| --- | --- |
| Direct weight source | Our `llm-semantic-router/Decision-1.0-Sol-2B`, HF commit `0665a41108e8f0b33a9515c98311c45947b99399` |
| Adapter | Existing `targeted3024` BEST160 checkpoint, model fingerprint `49ce326b5b116ed81397cda874f6e287e8c37126ec2f1f86e7140a4ca39ff629` |
| Calibration | Original hard900 CAL, SHA-256 `0ed1805105febba3c1639a8b230781085913d0c6e80e456a18f036f354c61227` |
| Package | Unmerged PEFT adapter plus native loader, `MODEL_MANIFEST.json` SHA-256 `a917ffe0cc64aa6e551825f0d4458d16deffe56ab0bcb451bd8f91687c953d8a` |
| Loaded parameters | 1,900,750,144, including the externally pinned own-1.0 text backbone, adapter and decision head |
| Code source | `6dfb660707be5e076b0c3194315d607e6935bac2` plus this documentation commit; mirrored inference-source SHA-256 aggregate `7dd35d91ac6bd97ca55c5d92c722433b0ae3d055d8c197d6b69e3dddbc9a8902` |
| Runtime | Python 3.12.13, Torch 2.12.0+git6bbd260, Transformers 5.17.0, PEFT 0.21.0, safetensors 0.8.0, Hugging Face Hub 1.31.0; one BF16 AMD GPU, one pinned container image |

The own-1.0 control used its original native loader in that same image and
physical GPU. Its pinned source required FLA 0.5.2. The first control smoke
failed before inference because FLA was absent from `PYTHONPATH`; using the
already installed FLA 0.5.2 restored the validated own-1.0 runtime. The
runtime comparison then reported no dependency differences. No candidate
weight, prompt, calibration, gate or score was changed to resolve the smoke.

The old BEST160 scored manifest was produced by a different inference-source
revision and could not satisfy the current package builder. Two **fresh**
current-runtime, gold-free 32-question source processes agreed on all answers
and probabilities. The resulting new manifest bound the original checkpoint,
CAL and current inference hashes. Historical scores were not transferred to
the new package.

## Package identity and development evidence

The fixed 32-question gold-free roster had 12 Choice, 10 Noul and 10 Score
questions (SHA-256
`3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55`).
The new package loader and original unmerged source were 32/32 valid with
zero categorical changes and maximum probability/Score drift of zero; the
package inventory remained unchanged after inference. The private package
parity receipt has SHA-256
`02d535619319ecf4311e7af0d2edb6d2eb1dd4dcb6a2327eeee7d35b848a6af8`.

All scored rows below used the same frozen gold-free prompts, package-native
inference for 2.0, own-1.0 native inference for the matched control, and the
same panel-specific scorer. Missing and invalid answers remained in each
denominator; all returned answers here were valid.

| Development panel | Own Sol 1.0 | BEST160 unmerged package | Difference |
| --- | ---: | ---: | ---: |
| Typed DEV, total correct / 1,600 | 945 | 938 | -7 |
| Typed DEV, four-family macro accuracy `T` | 0.590625 | 0.586250 | -0.004375 |
| Choice correct / 800 | 416 | 427 | +11 |
| Noul correct / 400 | 218 | 218 | 0 |
| Score correct / 400 | 311 | 293 | -18 |
| CSS pilot, task-median macro-F1 `H`, 3 tasks / 1,430 | 0.315173 | 0.350592 | +0.035419 |
| CSS pilot, micro accuracy | 0.387413 | 0.423776 | +0.036363 |
| Public JevBench subset, correct / 231 | 161 | 162 | +1 |

On the public 231-question subset, both models answered 48/48 easy and 47/111
hard; the package answered 67/72 standard versus 66/72 for own Sol 1.0.
This is an independent public-subset reproduction, not the upstream closed
benchmark rank. The single-question difference is too small to support a
strong public-benchmark claim.

The frozen **development proxy** `100 × sqrt(T_DEV × H_pilot)` is 45.33594 for
the package versus 43.14498 for own Sol 1.0: **+2.19096 points**, clearing
the preregistered +2 advancement threshold. Public231 is also non-inferior.
Typed DEV regressed slightly overall, especially Score, while Choice and
transfer improved. This tradeoff must accompany any later aggregate result.
These are repeatedly visible development/public panels, so this proxy cannot
be called JevArena v3 or independent release evidence.

Panel prompt SHA-256: typed DEV
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`,
CSS pilot
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`,
public231
`642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`.
The corresponding private score-report SHA-256 values are:

| Panel | Own Sol 1.0 | BEST160 package |
| --- | --- | --- |
| Typed DEV | `a1902f0f22f9947358a6a5d2e7cb5c9d269aa176bfe92033b54ad1f346a4febd` | `d2ba23f70fb42783fc65585e692cdc0217659426c18f574b534ef61edba34856` |
| CSS pilot | `b14f53b85f3ed9a66e41f489b4557da70db6652817fa2567526a790b63a5bd45` | `6c0135ca246c85a63abc064bf8843831fcc3918809893b1ad3c1bc43dad97aaf` |
| Public231 | `76f1faeee309b8f766ed8d9b1fa885b684acb38dce7e3b43a1e0034e2e85df8a` | `a5e2d5e4bf09408478043e996e7e3dbe144bc2588cab3148aed0cab141212c28` |

The source BEST160 checkpoint was rerun against the copied package on the
**full** typed DEV and CSS pilot prompts. All 3,030 answers were valid, with
zero category changes and maximum probability drift zero. Private parity
receipt SHA-256: typed DEV
`ac0b8485756741579147a4044a3126b7cbb51f228186509fff5e32fba8ec0554`,
CSS pilot
`5c4a8b2ea2f31a4bccec6c097117e6d082f3a30c785223a599ab1c35876e1eea`.
The first generic parity comparator expected all three question types and
rejected the Choice-only CSS panel without comparing predictions. A
panel-aware comparator retained the same fixed numerical and category gates
and produced the passing receipt; no model, prompt or answer changed.

Estimated GPU consumption for the preflight and separately authorized full
development/control readout is approximately **0.17 GPU-hour** on one GPU.
The initial gold-free package preflight consumed about 0.027 GPU-hour,
within its registered 0.15 GPU-hour cap. Private raw predictions, logs,
runtime lock and original data remain in the authorized experiment store.

## Decision and next gate

Advance this **exact** unmerged package to a prospective, gold-free formal
candidate freeze; do not substitute the failed full merge or a new checkpoint.
Before a release claim, complete the registered same-panel JevArena v3
comparison, dataset-overlap and publication audits, matched near-size peer,
and exact downloaded-package/native-output verification. Any formal v3 use is
post-key prospective comparison because the protected v3 key had already been
used on an unrelated arm; it must not be described as a virgin blind test.
JevBench ranks may include only models evaluated on the identical public231
panel and scoring path. No model was uploaded or published by this experiment.
