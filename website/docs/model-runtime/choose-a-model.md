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
| Find personal information | `vllm-sr/Vela-1.0-Encoder-307M-PII` | 307M | 17 entity types, with exact character spans; Vela 2.0 finds them too, in the same call as its other questions |
| Stop prompt injection and jailbreaks | `vllm-sr/Vela-1.0-Encoder-307M-Guard` | 307M | |
| Flag unsafe content | `vllm-sr/Vela-1.0-Encoder-307M-Safety` or `-Shield` | 307M | Shield is an alternative safety model |
| Name the kind of risk | `vllm-sr/Vela-1.0-Encoder-307M-Hazard` | 307M | 12 independent hazard categories with published thresholds |
| Check an answer against its sources | `vllm-sr/Vela-1.0-Encoder-307M-Halu` | 307M | Marks unsupported spans of the answer; Vela 2.0 marks them too |
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
each model card and in `GET /v1/models`. These are the defaults: with no model
configured, each built-in signal runs on its Vela 1.0 model.
[Vela 2.0](#vela-20) can answer all of them in one call, except Hazard,
embeddings, reranking and Omni.

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
`-0.8B`, `-4B`, `-9B`) adds questions that pick several labels (`set`) or mark
spans of text (`span`), and the router routes on both. Its router span head
also answers the [`pii`](tutorials/signal/learned/pii.md#vela-20) and
[`hallucination`](tutorials/signal/learned/hallucination.md#vela-20) signals,
so one deployment can replace the separate PII and Halu models. On a CPU,
run the 0.3B. On a GPU, the larger sizes read inputs of up to 16,384 tokens
(the 0.3B reads 8,192): the 0.8B costs the least of them, and the 4B and 9B
are the most accurate.

`vllm-srun models` prints every built-in model with its pinned revision.

## Run the built-in signals on Vela 2.0 {#vela-20}

Vela 2.0 is public on Hugging Face
([collection](https://huggingface.co/collections/vllm-sr/vela-20)). One
deployment of it can answer the router's domain, jailbreak, safety, fact
check, user feedback, modality, PII and hallucination signals. Each signal
asks the question the model was trained on for it, with the labels of its
Vela 1.0 model, so rules, thresholds and policies read the answer as before,
and a request asks all of them in one call. Bind the signals to the
deployment:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      domain_classifier: {deployment: vela2, contract: label_distribution.v1}
      prompt_guard: {deployment: vela2, contract: label_distribution.v1}
      fact_check_classifier: {deployment: vela2, contract: label_distribution.v1}
      feedback_detector: {deployment: vela2, contract: label_distribution.v1}
      modality_detector: {deployment: vela2, contract: label_distribution.v1}
      pii_classifier: {deployment: vela2, contract: token_spans.v1}
      hallucination_detector: {deployment: vela2, contract: token_spans.v1}
```

A safety rule binds as `safety.<rule name>`. The model reads the whole text,
so the deployment takes no `input` and the signals no `window`; remove the
`prompt_guard` and PII windows if your configuration sets them. Hazard
categories, embeddings, multimodal embeddings and reranking keep their Vela
1.0 models. To go back to Vela 1.0 for a signal, remove its binding.

What you get is one model and one call for every signal, and spans for PII
and unsupported claims. What it costs is CPU time: the defaults stay on Vela
1.0 because on a CPU the 0.3B is much slower than the separate Vela 1.0
models. Every request carries the questions, their options and the 17 PII
labels (at least 560 tokens) through one 307M-parameter forward, where each
Vela 1.0 model reads only the request. Through the Router on 12 CPU cores,
for the five request signals of the
[latency record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md):

| Router on 12 CPU cores | p50 | p95 | Requests per second |
| --- | ---: | ---: | ---: |
| Vela 1.0 (the defaults) | 16 ms | 59 ms | 38 |
| Vela 2.0 0.3B | 128 ms | 154 ms | 7.5 |

On the [router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)'s
held-out rows, through the Router, the 0.3B is ACCURACY_SUMMARY. The
[A/B record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)
has every signal and file. [#4668](https://github.com/vllm-project/semantic-router/issues/4668)
evaluates Vela 2.0 as the default on GPUs: on one AMD Instinct MI325X the 0.3B
answers the same router questions in about 7 ms at the median
([measurements](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)).

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
ROCm. Its `causal-conv1d` also carries code for MI200 and MI350 GPUs, so the
models that use it run there too, though only MI300X and MI325X are validated.
The built-in models' GPU reference answers are checked on that stack,
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
