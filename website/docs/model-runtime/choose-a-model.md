---
title: Choose a model, size and hardware
sidebar_label: Choose a model
description: Which model to use for each task, how big a decision model you need, and what hardware runs it.
---

# Choose a model, size and hardware

Start from the task. Every built-in model below is pinned to an exact Hugging
Face revision, so the same name always loads the same files.

## By task

| You want to | Model | Size | Notes |
| --- | --- | --- | --- |
| Route by subject (math, law, code, ...) | `vllm-sr/Vela-1.0-Encoder-307M-Domain` | 307M | 14 domains |
| Spot requests that need fact checking | `vllm-sr/Vela-1.0-Encoder-307M-FactCheck` | 307M | It flags the need; it does not check facts |
| Read how a user reacts to the last answer | `vllm-sr/Vela-1.0-Encoder-307M-Feedback` | 307M | Satisfied, needs clarification, wrong answer, wants something different, no feedback |
| Tell text requests from image requests | `vllm-sr/Vela-1.0-Encoder-307M-Modality` | 307M | Reads the written request only |
| Find personal information | `vllm-sr/Vela-1.0-Encoder-307M-PII` | 307M | 17 entity types, with exact character spans |
| Stop prompt injection and jailbreaks | `vllm-sr/Vela-1.0-Encoder-307M-Guard` | 307M | |
| Flag unsafe content | `vllm-sr/Vela-1.0-Encoder-307M-Safety` or `-Shield` | 307M | Shield is an alternative safety model |
| Name the kind of risk | `vllm-sr/Vela-1.0-Encoder-307M-Hazard` | 307M | 12 independent hazard categories with published thresholds |
| Check an answer against its sources | `vllm-sr/Vela-1.0-Encoder-307M-Halu` | 307M | Marks unsupported spans of the answer |
| Embeddings for cache, memory, RAG and tools | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | 307M | Smaller sizes and fewer layers trade quality for speed |
| Larger or instructed text embeddings | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1,024 dimensions |
| Rerank retrieved documents | `vllm-sr/Vela-1.0-Encoder-307M-Reranker` | 307M | |
| Embed text, images and audio together | `vllm-sr/Vela-1.0-Omni-Nano` or `-Mini` | 164M / 1.36B | Mini is more accurate and accepts longer text |
| Ask your own questions in plain language | A decision model (next section) | 0.6B to 27B | |

The task models all run well on a CPU: on 16 cores the median Vela Domain
request takes about 12 ms, three times faster than the native bindings that
earlier releases used
([measurements](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela1-performance.md)).
Most of them read up to 32,768 tokens; longer or shorter limits are listed on
each model card and in `GET /v1/models`.

## Decision models

Decision models answer questions you write yourself, such as "does this need
step-by-step reasoning?" or "which of these models should answer?". Pick the
smallest one that is accurate enough for your questions.

| Model | Size | Runs well on | Good for |
| --- | --- | --- | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 0.6B | CPU (about 0.2 s for two questions on 16 cores) or any GPU | Fast, simple routing questions; the default choice to start with |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 0.8B | CPU or any GPU | Slightly harder questions at similar cost |
| `vllm-sr/Decision-2.0-Sol-2B` | 2B | GPU; CPU for low traffic | Questions that need more judgment |
| `vllm-sr/Decision-2.0-Nox-4B` | 4B | GPU | Nuanced questions and many options |
| `vllm-sr/Decision-2.0-Lux-9B` | 9B | GPU (24 GB or more) | The most accurate at moderate cost |
| `vllm-sr/Decision-2.0-Vega-27B` | 27B | One GPU with 64 GB or more | The most accurate overall |

Decision 1.0 models (`vllm-sr/Decision-1.0-Kai-0.6B`, `-Lex-0.6B`,
`-Route-0.6B`, `-Eos-0.8B`, `-Sol-2B`, `-Nox-4B`, `-Lux-9B`) are also built in
and answer the same kinds of questions. Vela 2.0 (`vllm-sr/Vela-2.0-0.3B`,
`-4B`, `-9B`) adds questions that pick several labels or mark spans of text; it
is a private preview and needs a Hugging Face token with access.

`vllm-sr-runtime models` prints every built-in model with its pinned revision.

## Hardware

| Hardware | Status | Use |
| --- | --- | --- |
| CPU | Validated | Every router image runs models on CPU out of the box. |
| AMD Instinct MI300X, MI325X | Validated | Set `device: rocm:0`. `vllm-sr serve --platform amd` and the `extproc-rocm` image ship PyTorch for ROCm. |
| NVIDIA GPUs | Works, not yet validated | Set `device: cuda:0`. `vllm-sr serve --platform nvidia` ships PyTorch for CUDA. |
| Intel GPUs | Available, not yet validated | `device: xpu:0`, with the runtime installed next to an XPU build of PyTorch. |
| Apple silicon | Available, not yet validated | `device: mps`, with the runtime installed on macOS. |

On AMD GPUs the router images ship the stack the runtime is validated on:
PyTorch 2.12 for ROCm 7.2, FLA 0.5.2, and `causal-conv1d` 1.7.0 built for
ROCm. The built-in models' GPU reference answers are checked on that stack,
and every model compares itself with them when it loads. With another
PyTorch, ROCm or kernel build, a model can fail that check or report
[`unverified`](./troubleshooting.md#a-ready-models-self-check-says-unverified).
If a model's reference answers had to be recorded again on this stack, its
family's [record](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)
says so and gives how often it agrees with the released answers.

`device: auto` (the default) picks the first validated GPU with enough free
memory and otherwise the CPU. A GPU you name explicitly must exist, or the
model fails to load with a clear reason instead of quietly running on the CPU.

### How much memory

Plan for about 4 bytes per parameter on a CPU and 2 bytes per parameter on a
GPU, plus room for the requests: a 307M task model needs about 1.3 GB on a
CPU, and Decision 2.0 Lux-9B about 18 GB on a GPU. The runtime refuses to load
a model that does not fit its device and says why. To keep large models apart,
give them their own process (see
[Run it with the router](./deploy.md#group-models-into-processes)).

## Your own models

Hugging Face ModernBERT and mmBERT classifiers, token classifiers and
embedding models load the same way as the built-in Vela models: give a Hub
repository with `revision`, or an absolute path to a local copy. Models of
another architecture need a family plugin; see
[Add your own model family](./plugins.md).
