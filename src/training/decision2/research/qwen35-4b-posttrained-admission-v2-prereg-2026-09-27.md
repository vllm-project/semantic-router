# Official Qwen3.5 4B Posttrained: corrected runtime admission v2

**Status: bounded v2 admission PASS; the full 466-update arm remains unstarted
and unapproved.** The [v1 attempt](qwen35-4b-posttrained-admission-lock-2026-09-27.md)
remains a separate HOLD, with no model forward pass. This record changes only
the interpreter used inside the same immutable image. It does not revise the
previous result, reuse its lock, select a model, or authorize the 466-update
arm.

The archived successful Base zero-step, one-update, and full-run containers
used the exact image ID
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
and invoked `/usr/bin/python`, which contains Transformers `5.17.0` and
PyTorch `2.12.0+git6bbd260`. Their source loader is the same
`decision_model.py` SHA-256
`ee3db820db73011d60e99b98c3067e33d85c8f4cbae08b53ea0604869ffba0ca`.
The v1 admission script instead invoked the image's separate Vela Python with
Transformers `4.57.6`; `AutoConfig` rejected the shared `qwen3_5` model type
before any model forward pass. Both official 4B source configurations have
identical SHA-256
`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.

## Corrected CPU gate and immutable inputs

The v2 runner is schema `decision2-qwen35-4b-posttrained-admission/2`. It
refuses any interpreter other than `/usr/bin/python` or package version other
than the archived Base runtime, in both CPU and GPU phases. The frozen image,
official Posttrained source revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, source shards, rights-clean
v2 TRAIN 7,455 / SELECT 700 / CAL 700, objective, native renderer, rank-16
LoRA, head, seed, and maximum stage durations remain those of the
[prospective source ablation](qwen35-4b-posttrained-vs-base-prereg-2026-09-27.md).
The completed Base control is never repeated. No formal or public benchmark
label is available to this gate.

Before writing a **new** private lock, `prepare` must run inside the pinned
image without any GPU device or network access. It must verify every frozen
source/data/code hash, then fully load the official Posttrained weights into
CPU memory via the actual `DecisionModel.from_base` source path. The resulting
model must report `qwen3_5`, the pinned revision and Posttrained stage, and
all parameters must remain on CPU. The source tokenizer must natively encode
one SELECT item from each of Choice, Noul, and Score within 8,192 tokens;
prompt/token hashes are sealed in the private preflight receipt. No score or
prediction is computed. This gate tests the precise loader that v1 missed,
not the model's accuracy or inference repeatability.

The private output directory must be new, empty, mode `0700`. The runner
writes its CPU preflight receipt and then a mode-`0600` lock binding that
receipt, the same immutable image and source, all data/code hashes, interpreter
and package versions, and the fixed bounded GPU protocol. A second fresh
no-device process must reopen and verify that lock. Any CPU load failure,
hash mismatch, unsupported type, missing item, overlength input, accidental
GPU exposure, or runtime mismatch is **HOLD**. The complete preflight log and
receipt remain private. No lock is treated as permission to start GPU work.

Only after independent review of the new lock may an operator consider the
original bounded order: two fresh zero-step starts on all SELECT700 rows,
CPU pair comparison requiring zero category changes and probability drift
p99 ≤`.005` / max ≤`.02`, one fresh-source TRAIN optimizer update with finite
loss/gradient, and a separate 32-row checkpoint reload under the same parity
limits. The exact phase caps remain 540, 540, 360 and 180 seconds, combined
0.45 GPU-hour maximum, with **no retries**. A failed phase stops this lock.
Passing all admission gates would justify separately reviewing the prospective
466-update development arm, not launching it automatically or making a release
claim. The known Base v3 HOLD and 2B Posttrained Score collapse remain live
risks; the source hypothesis is untested until valid development readout.

## CPU-only execution and review handle

The signed local runner revision is `f1a280acfd467b6ab4dc20312d2fd3732ae95060`;
its SHA-256 is
`a9d67d6d2412cd5352aad55fb1b423c04687a69e828ab0ea06e426c0a571267a`.
The local `make test-training-contracts` gate and eight directly relevant
admission/source unit tests passed before mirroring. The exact code mirror ran
the pinned image with no GPU device and no network. The CPU source loader
completed on all 723 weight tensors, yielded **4,205,751,296 text parameters**
on CPU, and encoded a representative SELECT item from each native type.
The private CPU preflight receipt has SHA-256
`49f85678b61cc80353d95a25453f811d9b5395cccc0b08712508490d80abd726`.
Its `visible_gpu_count` is zero. A separate fresh no-device process reopened
the lock and returned `CPU_LOCK_VERIFIED` with the same SHA-256:
`cb2690675dc1d6d39370acbf4445ea667b0d08ee54c369077ee8c9f55fcfb86a`.
The two private CPU console logs have SHA-256
`f4d39e3266c3fb26a8ef91a77fc6769135f0069361d9dee0a82a4288f5e96ab9`
and `9748bd0cc1fc19d70e000d2c8d775d11bc75e4f7943b5dde9e5bc8bb0c5e6b1a`.
The lock and preflight receipt are mode `0600` under a mode `0700` experiment
directory. No prediction, training update, formal label access, or GPU use
occurred during this CPU stage. Independent review approved only the bounded
GPU admission stages below.

## Bounded GPU admission result

The exact v2 lock was used in four fresh, sequential, single-GPU processes;
the original Base control was not rerun. Each process exited successfully and
the source/data/code/image bindings were checked again. The two zero-step
processes covered all **700 SELECT rows**: Choice 320, Noul 290, Score 90.
The CPU pair gate found **zero categorical changes**, with probability p99
and maximum drift both **0**. The one-update smoke consumed **5,131 tokens**;
its loss `1.3252565` and gradient norm `14.9877062` were finite, and the
step-one checkpoint and 700-row predictions were durable. This is a numerical
smoke, not an accuracy or causal learning result. Separate reload of that
checkpoint on the frozen first 32 rows covered Choice 13, Noul 14, Score 5;
category changes and probability drift were all **0**.

| GPU phase | Runner walltime | Private receipt SHA-256 |
| --- | ---: | --- |
| `zero-a` | 53.447 s | `cb00b08ecf0b5c42f9e4fdb62f4f9e0184763d0d049eeb246ca22aeb5b3d5f38` |
| `zero-b` | 53.798 s | `23d6db469290cd1b92983abcaf1ad2d4c5a68357ca84b65a9a83ca3c7abbf07c` |
| `one` | 91.848 s | `6a25f6b6c01ce4d7d699b4606eaaf67e41c8269b1ee835e1fc9e8e38cf2fd3a9` |
| `reload` | 31.385 s | `b8a996b0c6354aa72989ab0efc6d6179ccc5f130374ab60ba83aec534f9296cf` |

The four frozen runner phases total **230.478 seconds / 0.064022 one-GPU
hours**, excluding container start/teardown and CPU-only pair comparison, well
below the 0.45-hour cap. The zero gate receipt SHA-256 is
`3f4fe285b5be539fe2da96ca796db7dbaefbe390d8cb00b783ad5f69d340cf83`.
The private signed final admission decision has SHA-256
`4b8a21348a8e0d957cf7f5f520d1b2ee7a81f8816453b25d8f50ee4a5aa21054`.
No formal, CAL, or public benchmark label was accessed, and no model was
uploaded. The result admits review of the separately preregistered 466-update
Posttrained development arm; it is **not** a release claim or authority to
launch that arm automatically.
