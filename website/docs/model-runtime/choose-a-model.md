---
title: Choose a model, size and hardware
sidebar_label: Choose a model
description: Which model to use for each task, how big a decision model you need, and what hardware runs it.
---

# Choose a model, size and hardware

Start from the task. Every built-in model below is pinned to an exact Hugging
Face revision, so the same name always loads the same files.

## By task

| You want to | Default model | Vela 1.0 specialist | Notes |
| --- | --- | --- | --- |
| Route by subject (math, law, code, ...) | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Domain` | 14 domains |
| Spot requests that need fact checking | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-FactCheck` | It flags the need; it does not check facts |
| Read how a user reacts to the last answer | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Feedback` | Satisfied, needs clarification, wrong answer, wants something different, no feedback |
| Tell text requests from image requests | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Modality` | Reads the written request only |
| Find personal information | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-PII` | 17 entity types, with exact character spans |
| Stop prompt injection and jailbreaks | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Guard` | |
| Flag unsafe content | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Safety` or `-Shield` | Shield is an alternative safety model |
| Check an answer against its sources | Vela 2.0 0.3B | `vllm-sr/Vela-1.0-Encoder-307M-Halu` | Marks unsupported spans of the answer |
| Name the kind of risk | `vllm-sr/Vela-1.0-Encoder-307M-Hazard` | | 12 independent hazard categories with published thresholds |
| Embeddings for cache, memory, RAG and tools | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | | Smaller sizes and fewer layers trade quality for speed |
| Larger or instructed text embeddings | `Qwen/Qwen3-Embedding-0.6B` | | 0.6B, 1,024 dimensions |
| Rerank retrieved documents | `vllm-sr/Vela-1.0-Encoder-307M-Reranker` | | |
| Embed text, images and audio together | `vllm-sr/Vela-1.0-Omni-Nano` or `-Mini` | | 164M / 1.36B; Mini is more accurate and accepts longer text |
| Ask your own questions in plain language | A decision model (next section) | | 0.6B to 27B |

With no model configured, the built-in signals the table gives Vela 2.0 0.3B
run on one deployment of it, in one call per request
([below](#vela-20)). Hazard, embeddings, reranking and Omni run their own
models. The Vela 1.0 specialists remain built in, and naming them restores
them. Each is a 307M encoder that runs well on a CPU: on 16 cores the median
Vela Domain request takes about 12 ms
([measurements](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela1-performance.md)).
Most read up to 32,768 tokens; the 0.3B reads 8,192. Each model card and
`GET /v1/models` list the limits.

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
which is how the default 0.3B deployment replaces the separate PII and Halu
models. On a CPU, run the 0.3B. On a GPU, the larger sizes read inputs of up to 16,384 tokens
(the 0.3B reads 8,192): the 0.8B costs the least of them, and the 4B and 9B
are the most accurate.

`vllm-srun models` prints every built-in model with its pinned revision.

## The built-in signals run on Vela 2.0 0.3B {#vela-20}

The domain, prompt guard, safety, fact check, user feedback, modality, PII and
hallucination signals default to `vllm-sr/Vela-2.0-0.3B`
([collection](https://huggingface.co/collections/vllm-sr/vela-20)). All of them
share one deployment, `@Vela-2.0-0.3B`, and a request asks every one of its
questions in one call.

- **Questions:** each signal asks the question the model was trained on for it,
  with the labels of its Vela 1.0 model, so rules and policies read the answer
  as before. PII and hallucination use the model's span head, so their spans
  keep exact character offsets.
- **CPU profile:** on a CPU the deployment runs `max_speed`, a packed copy of
  the model's weights. It gives the same answers to within about 0.00001 and
  is about 1.6 times faster than `exact`.
- **Input:** the model reads up to 8,192 tokens of a request and truncates the
  rest. The Vela 1.0 Guard and PII specialists scan up to 32K in windows.
- **Thresholds:** the module defaults are calibrated to the 0.3B's scores
  (below).

The maintainers chose this default although it misses two goals they had set
for it ([#4639](https://github.com/vllm-project/semantic-router/issues/4639)):
level or better accuracy on every signal, and level or better latency on a
CPU. Measured through the Router on the
[router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
([A/B record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)):

- **Ahead:** prompt guard (held-out AUC +0.026; on the E2E attack fixtures it
  blocks all six attacks, Vela 1.0 Guard five) and safety (+0.052 held-out, and
  ahead on every set). One model and one call serve every signal.
- **Level:** PII and hallucination on held-out and fresh files.
- **Behind, most:** modality (held-out AUC −0.180; the 0.3B misses most
  requests that ask for a new image) and user feedback (accuracy −0.038
  held-out, −0.178 fresh).
- **Behind:** domain (accuracy −0.037 held-out, −0.088 fresh) and fact check
  (held-out AUC −0.101).
- **CPU time:** every request carries the questions, their options and the 17
  PII labels (at least 560 tokens) through one 307M-parameter forward, where
  each Vela 1.0 model reads only the request. On 12 CPU cores, for the five
  request signals of the
  [latency record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md),
  the median request takes about 4.9 times as long:

| Router on 12 CPU cores | p50 | p95 | Requests per second | At concurrency 16 |
| --- | ---: | ---: | ---: | ---: |
| Vela 1.0 specialists (restored) | 16 ms | 58 ms | 38.9 | 51.8 |
| Vela 2.0 0.3B (the default) | 79 ms | 100 ms | 11.9 | 12.8 |

[#4668](https://github.com/vllm-project/semantic-router/issues/4668) works on the
CPU latency. On a GPU (`use_cpu: false`), the 0.3B answers the same questions
in about 7 ms at the median on one AMD Instinct MI325X
([measurements](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)).

### Thresholds

Each default threshold keeps the Vela 1.0 specialist's operating point on the
suite's dev split: its false-positive rate, or for a confidence floor its share
of requests below the floor. The defaults are prompt guard 0.75, domain 0.28,
PII 0.01, fact check 0.93 and user feedback 0.37.

- **PII:** the 0.3B's span head applies its own per-label thresholds before it
  returns a span, so 0.01 accepts every span it returns.
- **Other models:** a module that runs any other model and sets no threshold
  keeps its earlier default.
- **Your own rule thresholds** (`routing.signals.jailbreak[].threshold` and the
  like) are yours, and they were likely chosen for Vela 1.0. The record maps
  each Vela 1.0 value to the 0.3B: prompt guard 0.3–0.9 → 0.74–0.77, PII → 0.01,
  safety 0.5 → 0.46, fact check 0.95 → 0.93, modality `confidence_threshold`
  0.7 → 0.51.

### Restore the Vela 1.0 specialists

One block brings them back. Module thresholds you do not set return to the
specialists' defaults with them:

```yaml
global:
  model_catalog:
    system:
      safety: models/Vela-1.0-Encoder-307M-Safety
      prompt_guard: models/Vela-1.0-Encoder-307M-Guard
      domain_classifier: models/Vela-1.0-Encoder-307M-Domain
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
      fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck
      hallucination_detector: models/Vela-1.0-Encoder-307M-Halu
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

A modality classifier names `models/Vela-1.0-Encoder-307M-Modality` as its
`classifier.model_path`.

To bring back one signal only, set its line alone. User feedback, for example:

```yaml
global:
  model_catalog:
    system:
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

| Signal | Line under `global.model_catalog` |
| --- | --- |
| Domain | `system.domain_classifier: models/Vela-1.0-Encoder-307M-Domain` |
| Prompt guard | `system.prompt_guard: models/Vela-1.0-Encoder-307M-Guard` |
| Safety | `system.safety: models/Vela-1.0-Encoder-307M-Safety` |
| Fact check | `system.fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck` |
| User feedback | `system.feedback_detector: models/Vela-1.0-Encoder-307M-Feedback` |
| PII | `system.pii_classifier: models/Vela-1.0-Encoder-307M-PII` |
| Hallucination | `system.hallucination_detector: models/Vela-1.0-Encoder-307M-Halu` |
| Modality | `modules.modality_detector.classifier.model_path: models/Vela-1.0-Encoder-307M-Modality` |

Rule thresholds that a configuration sets itself stay where they are. The
built-in recipes' rules are calibrated to the 0.3B, so a signal moved back
takes its Vela 1.0 rule thresholds with it. In `mom-v1` those are prompt guard
0.5, safety 0.5 and PII 0.7; the record lists every recipe's.

Specialist overrides remain explicit task bindings; selecting a default
decision model does not remove them.

## Choose a size {#choose-a-size}

The default decision binding names a deployment that answers the Router's
judgment tasks and [`decision` questions](tutorials/signal/learned/decision.md)
without an override. Declare the resource once, then select its exact key.
Omission selects the built-in `primary` deployment, Vela 2.0 0.3B on CPU:

```bash
vllm-sr serve --decision-model primary --platform amd
```

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        device: rocm
    system:
      decision_model:
        deployment: primary
```

`serve` writes the line into the active configuration as a new version, which
`vllm-sr config versions` lists and `vllm-sr config rollback` undoes; later
starts keep it, and `vllm-sr status` shows it. The Helm chart's
`decisionModel` value and the operator's `spec.config.decision_model` set the
same binding. Deployment keys are exact and case-sensitive. Model identity,
device and profile belong to the deployment, not the binding.

Measured through the Router on the router signal suite, against the Vela 1.0
specialists, and for the latency record's five request signals
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)):

| Decision model | Hardware | Held-out accuracy against Vela 1.0 | p50 on a GPU | p50 on 12 CPU cores |
| --- | --- | --- | ---: | ---: |
| `Vela-2.0-0.3B` (default) | CPU or GPU | Ahead on prompt guard and safety, behind on domain, modality and feedback | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | CPU or GPU | Ahead on domain, prompt guard, safety, modality and hallucination; behind on PII | 40.1 ms | about 3 s |
| `Vela-2.0-4B` | GPU, about 17 GB | Ahead on every signal but fact check | 55.2 ms | GPU only |
| `Vela-2.0-9B` | GPU, about 32 GB | Ahead on every signal | 76.5 ms | GPU only |
| `Vela-1.0` | CPU or GPU | The specialists themselves | n/a | 16 ms |

- **GPU:** one AMD Instinct MI325X, sequential requests. At concurrency 16 a
  GPU serves about 154 (0.3B), 25 (0.8B), 18 (4B) and 13 (9B) requests per
  second.
- **The 4B and 9B need a GPU.** `vllm-sr serve` refuses them with
  `--platform cpu` or on a host without the platform's GPU, and the Router
  refuses them where the model runtime finds no GPU. On a GPU they run whatever
  a module's `use_cpu` says.
- **The 0.8B on a CPU** is a decoder: a request takes seconds, as the table
  shows. Serve it on a GPU, or keep the 0.3B on a CPU.
- **Every size** is behind Vela 1.0 on user feedback's fresh file (CrossWOZ)
  and on PII in distribution. Per-signal numbers with intervals are in the
  record.
- **Decision 1.0 and Decision 2.0** may be the default judgment deployment.
  Available tasks follow the model's native capabilities; an unsupported task
  is unavailable regardless of the model family.
- **Specialists** remain explicit task overrides and can run alongside the
  default decision deployment.

Each size has its own module thresholds, which a module that sets none takes
when you switch:

| Decision model | Prompt guard | Domain | PII | Fact check | User feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| `Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| `Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| `Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |
| `Vela-1.0` | 0.5 | 0.5 | 0.9 | 0.95 | 0.7 |

Rule thresholds a configuration sets, such as the built-in recipes', stay;
the record maps each to every size (for example `mom-v1`'s `prompt_attack`
0.75 is 0.71 on the 0.8B, 0.63 on the 4B and 0.42 on the 9B). A
`system.<module>` line or a binding keeps that one signal on its own model.

## Hardware

| Hardware | Status | Use |
| --- | --- | --- |
| CPU | Validated | Every router image runs models on CPU out of the box. |
| AMD Instinct MI300X, MI325X | Validated | Set `device: rocm:0`. `vllm-sr serve --platform amd` and the `vllm-sr-rocm` image ship PyTorch for ROCm. |
| NVIDIA GPUs | Works, not yet validated | Set `device: cuda:0`. `vllm-sr serve --platform nvidia` ships PyTorch for CUDA. |
| Intel GPUs | Available, not yet validated | `device: xpu:0`, with the runtime installed next to an XPU build of PyTorch. |
| Apple silicon | CPU only in this release | On macOS the docker target runs the CPU image, because Docker's Linux VM gets no GPU. Host GPU support is tracked in [#4636](https://github.com/vllm-project/semantic-router/issues/4636). |

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
[Run it with the router](./deploy.md#place-and-scale-replicas)).

## Your own models

Hugging Face ModernBERT and mmBERT classifiers, token classifiers and
embedding models load the same way as the built-in Vela models: give a Hub
repository with `revision`, or an absolute path to a local copy. Models of
another architecture need a family plugin; see
[Add your own model family](./plugins.md).
