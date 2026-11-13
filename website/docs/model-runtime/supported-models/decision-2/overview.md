---
title: Decision 2.0
sidebar_label: Decision 2.0
description: Six Decision 2.0 sizes, from Kai-0.6B on a CPU to Vega-27B on one large GPU, answer choice, noul and score questions about text, zero-shot.
---

# Decision 2.0

Decision 2.0 answers your own `choice`, `noul` and `score` questions about
text, zero-shot. Pick the smallest size that is accurate enough for your
questions: Kai-0.6B runs on a CPU, and the larger sizes on a GPU.

```bash title="Start the smallest size on a CPU"
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
```

## Sizes

<!-- BEGIN GENERATED MODEL TABLE -->

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Decision-2.0-Kai-0.6B](./kai-0-6b) | 0.597B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Eos-0.8B](./eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Sol-2B](./sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-2.0-Nox-4B](./nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-2.0-Lux-9B](./lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |
| [Decision-2.0-Vega-27B](./vega-27b) | 29.4B | choice, noul, score | — | text | 72 GiB |

<!-- END GENERATED MODEL TABLE -->

| Model | Runs well on | Good for |
| --- | --- | --- |
| Kai-0.6B | CPU (about 0.2 s for two questions on 16 cores) or any GPU | Fast, simple routing questions; the default choice to start with |
| Eos-0.8B | CPU or any GPU | Slightly harder questions at similar cost |
| Sol-2B | GPU; CPU for low traffic | Questions that need more judgment |
| Nox-4B | GPU | Nuanced questions and many options |
| Lux-9B | GPU (24 GB or more) | The most accurate at moderate cost |
| Vega-27B | One GPU with 64 GB or more | The most accurate overall |

:::warning No spans
Decision 2.0 answers no span questions. When a Decision 2.0 size is the decision
model, bind `pii` and `hallucination` to a span-capable deployment; each model
page shows the configuration.
:::

## How it runs

- **Precision:** BF16 on a GPU, FP32 on a CPU.
- **Shared context is on.** A request with many questions about one input runs
  the input once, up to 3.4× faster for 128 questions on Vega-27B; its answers
  equal the released runtime's
  ([Engine arguments](../../deploy/engine-args.md#shared-context)).
- **Batching** merges concurrent requests. On ROCm, merged rows changed as many
  decisions as asking the same questions alone instead of together does
  ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)).
- **Faster kernels on NVIDIA:** on GPUs from Ampere on, with Triton, it runs
  fused element-wise kernels, 11% to 48% faster with no decision changed
  ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-fused-approximate.md)).
- The Qwen3.5 sizes (all but Kai-0.6B) need CPU LAPACK, which the ROCm image's
  PyTorch lacks: on a CPU, serve them from the CPU image.
