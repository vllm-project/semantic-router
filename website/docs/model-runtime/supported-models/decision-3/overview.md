---
title: Decision 3.0
sidebar_label: Decision 3.0
description: The d3 family answers choice, noul and score questions about text, images and video, zero-shot, from 0.6B to 27B.
---

# Decision 3.0

Decision 3.0 (d3) answers your own `choice`, `noul` and `score` questions about
text, images and video, zero-shot. Every question in a request sees the same
images and videos. The family runs on a GPU and returns no spans.

```bash title="Start the smallest size on an AMD Instinct MI325X"
vllm-sr serve vllm-sr/d3-lite --engine --platform rocm
```

## Sizes

<!-- BEGIN GENERATED MODEL TABLE -->

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [d3-edge](./d3-edge) | 0.595B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-lite](./d3-lite) | 0.853B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-nano](./d3-nano) | 2.21B | choice, noul, score | — | text, image, video | 8 GiB |
| [d3-mini](./d3-mini) | 4.54B | choice, noul, score | — | text, image, video | 16 GiB |
| [d3-flash](./d3-flash) | 8.39B | choice, noul, score | — | text, image, video | 32 GiB |
| [d3](./d3) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-turbo](./d3-turbo) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-ultra](./d3-ultra) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |

<!-- END GENERATED MODEL TABLE -->

- **d3-edge to d3** are the plain sizes. Their answers match the packages' own
  runtime bit for bit
  ([parity record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-parity.md)).
- **d3-turbo and d3-ultra** are 27B state-once models, below.
- d3-edge is a pruned d3-lite. d3-lite, d3-mini, d3-flash and d3-turbo are also
  the bases of the larger [Vela 3.0](../vela-3/overview.md) sizes.

:::warning No spans
d3 answers no span questions. When a d3 size is the decision model, bind `pii`
and `hallucination` to a span-capable deployment, such as vela3-fast; each
model page shows the configuration.
:::

## State-once models: d3-turbo and d3-ultra

`vllm-sr/d3-turbo` reads a request's input once and answers every question from
that reading, so asking many questions about one input costs little more than
asking one
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/state-once.md)):

- Besides `choice`, `noul` and `score`, it answers `box` questions (where
  something is in an image, or in a video at the second given by `at`) and
  `track` questions (a box for each moment of a whole video, 1 or 2 per
  second) on the [System One API](../../../system-one/use/api.md).
- A question or an option may hold an image of its own (`assets`).
- Each question reads up to 32,768 tokens by default and up to 262,144 with
  `options.long_context`, on one AMD Instinct MI325X.

`vllm-sr/d3-ultra` also reads a request's input once and answers the same
question types as d3-turbo
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/d3-ultra.md)):

- A text-only request reads its list of questions in front of its state, so
  every question sees what else is asked; a request with images or videos reads
  as d3-turbo's does.
- A one-question request answers at the temperature the package sets for it.
- Each question reads up to 131,072 tokens by default, on one AMD Instinct
  MI325X.

## Quantized packages

A d3 package with FP8, int4 (GPTQ or AWQ) or NVFP4 weights in the
compressed-tensors format, and a `MODEL_MANIFEST.json`, is served the same way,
from a local path or from its repository with the commit to load. The published
packages of `vllm-sr/d3` are `vllm-sr/d3-FP8`, `vllm-sr/d3-W4A16-AWQ`,
`vllm-sr/d3-W4A16-GPTQ`, `vllm-sr/d3-NVFP4` and `vllm-sr/d3-NVFP4-AWQ`.

| Package | Keeps of the BF16 answers | Device memory (BF16: 52.2 GB) |
| --- | ---: | ---: |
| FP8 | 99.08% | 31.9 GB |
| GPTQ int4 | 97.98% | 16.4 GB |
| AWQ int4 | 97.76% | 16.5 GB |
| NVFP4 AWQ | 96.81% | 17.2 GB |
| NVFP4 | 96.64% | 17.2 GB |

[Quantization](../../quantization/overview.md) explains the formats, the
hardware each needs and how to serve one.

## Precision and hardware

- d3 runs in BF16 on every device, its published precision.
- It is validated on AMD Instinct MI300X and MI325X; the model pages list the
  devices whose reference answers are recorded for each revision.
- The ROCm image's PyTorch has no CPU LAPACK, which the Qwen3.5 backbone needs
  on a CPU: serve d3 on the GPU, or on a CPU from the CPU image.
- On a GPU, d3 batches text rows and image or video rows in separate lanes, and
  sizes a merged forward to about 0.3 s of device time
  ([Engine arguments](../../deploy/engine-args.md#batching)).
