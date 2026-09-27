# Gemma 4 ~26B Decision arm: one-cell development result

**Decision: HOLD.** The single authorized full development cell completed its
frozen 456 updates without a numeric, budget, or Score-collapse stop. Its
best SELECT family macro accuracy was **0.72491**, below the prospective
**0.794** threshold for an independent diagnosis. These are development
results, not JevArena, JevBench, release, or model-quality claims.

## Locked run and resource accounting

The [prospective lock](gemma4-full-development-arm-lock-2026-09-27.md) used
official `google/gemma-4-26B-A4B-it` revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, a text-only native
state/typed-question Decision head plus q/o LoRA, and rights-clean v2 TRAIN
and SELECT. The **25,233,141,760 loaded text parameters** and
**25,805,933,872 total loaded parameters** are separate from the nominal
**3,822,530,560 active text parameters per token**. Exactly 6,540,800
adapter/head parameters were trainable. The source is an official general
model, not a third-party Decision checkpoint.

The roster was fixed at 7,287 TRAIN rows across Choice/Noul/Score, a seeded
one-epoch schedule of 456 optimizer updates, and 3,620,578 observed TRAIN
tokens. SELECT contained 700 rows. The runner mounted only TRAIN and SELECT;
it did not open CAL, DEV, formal, JevBench, Decision Index or public
benchmark labels. Its exit status was 0 after **4,230 wall seconds = 1.175
conservative GPU-hours** on one accelerator. Peak allocated HBM was
88,049,071,616 bytes. The accelerator was released after the run. There
was no automatic retry, source change, threshold change, or extra optimizer
arm.

## Frozen SELECT readouts

| Update | TRAIN tokens seen | Family macro accuracy | Family macro normalized Brier | Choice accuracy | Noul accuracy | Score accuracy | Score predicted levels | Largest Score-level share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 510,551 | 0.54291 | 0.30172 | 0.55938 | 0.68621 | 0.45556 | 5 | 0.61111 |
| 128 | 1,037,523 | 0.61063 | 0.24863 | 0.61250 | 0.78621 | 0.42222 | 4 | 0.51111 |
| 256 | 2,081,612 | 0.70102 | 0.19677 | 0.70312 | 0.89655 | 0.41111 | 3 | 0.90000 |
| 456 | 3,620,578 | **0.72491** | **0.17881** | 0.69688 | 0.89655 | 0.64444 | 5 | 0.46667 |

The step-16 checkpoint was a numeric and GPU-hour projection gate only;
its 131,431 tokens took 101.08 TRAIN seconds and it did not read SELECT.
At step 256 the best family macro score had just crossed the frozen 0.70
futility threshold. Score retained three predicted levels with 90% in its
largest level, passing the separately frozen collapse rule. At step 456,
Score recovered to 5 predicted levels and 0.64444 accuracy. The frozen
choice rule selected checkpoint **456** by highest SELECT family macro
accuracy.

Individual-family results expose a substantial unresolved weakness: the
40-row SELECT string-composition family changed from 0.15 at step 64 to
0.25 at step 128, 0.275 at step 256, then **0.175** at step 456. In the
full control TRAIN, that family has only 60 of 7,455 rows; this count is
descriptive and does not establish causality. At step 456, the 90-row
targeted quantized-median family reached 0.64444. Near-perfect SELECT
accuracy on the narrative-reading and open-world-abstention pilot families
may reflect family overlap and cannot establish cross-source transfer.

## Integrity receipt and next decision

The private lock SHA-256 is
`20b7524b0d6dceeb57237e312fc8d341b31b962bca5b177688ffb7be3b1798c3`.
The completed run recorded this same lock digest and 456 updates. Every
saved checkpoint at updates 16, 64, 128, 256 and 456 was independently
rehash-checked against its milestone manifest. The selected 456 package is:

| File | SHA-256 |
| --- | --- |
| `adapter/adapter_config.json` | `f015ed2c628826d0cc7059f20abacddd83f25e9eef9e9d5f671eaa2801cffb86` |
| `adapter/adapter_model.safetensors` | `27a044d2015d6056fc170fb7baf937aa38a9d263568a52a93ac1bb03a5647939` |
| `decision_head.safetensors` | `af273b58f07b3bf4b53e49a008a6130a41a28487bf06f850b236c57399a859b8` |
| `trainer_state.pt` | `4c721d3688740df4e52e70d04b83273fc7c42420fd5aeb397d2efc6b004ce6ae` |

The private log, timing, completion and selected-milestone SHA-256 digests
are respectively `7a78058c7aee5825c8d9c407ad3b645240dafa7d570cc89b312304bd538a271a`,
`52b51f938e04786c19b14c23acd7137cc525e8df7a6aee5658ac8ce63a3e2477`,
`510ab983c0f5c995c5f1274498577040b55a5932542d33aa8a9e7d605099df60`
and `60657fa34d6558b8587e1f0c5950e20a5cf6fb11d45b656529e33eb7bb547b7a`.
Raw SELECT predictions and the checkpoint remain private.

**Next discriminative work is CPU-only first:** audit source/group/template
coverage of the 60 TRAIN and 40 SELECT string-composition cases and their
relationship to the rest of the roster. If a distinct mechanism gap is
confirmed, freeze one data-only, equal-token intervention from the same
official source and adapter, with program-verifiable composition examples
and a generator-disjoint diagnostic. Preserve a no-intervention comparator,
Score non-collapse checks, and the existing SELECT rule. This is a proposed
experiment, not approval to train or access formal labels. The present
checkpoint stays HOLD and is not an initial 27B release candidate.
