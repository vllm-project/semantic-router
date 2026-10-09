---
title: Model runtime
sidebar_label: Overview
description: The models that classify, protect, embed and route your requests run in the built-in model runtime. Start here.
---

# Model runtime

The **built-in model runtime** serves the decision, classifier, embedding,
reranking, and hallucination models used by routing features. It can run them
as managed workers or attach to independently operated workers. Features with
an explicit [external service](../installation/runtime/external) use that
service instead.

The runtime is not where your chat models run. The models that answer your
users stay behind your providers (vLLM, Ollama, a hosted API); the runtime
serves the models the router consults about each request. Managed workers run
as `vllm-srun` processes inside the Router container. Starting
`vllm-sr serve ARTIFACT --engine` keeps the same instance frontend and model
management, with Chat routing disabled. Router mode can serve native System
One requests at the same time as routed Chat requests.

![Frontend, optional decision engine, and on-demand model runtime](/img/architecture/system-one/01-component-composition.svg)

See [Component Architecture](../overview/component-architecture) for the
request paths and the distinction between model selection and replica dispatch.

Built-in features resolve their model defaults automatically. During
configuration preparation, the Router starts managed workers only for actual
model consumers and explicitly published native models. An unused deployment
does not load weights. Startup waits for required managed models; attached
model readiness follows the [deployment rules](./deploy.md#when-a-model-is-not-ready). If a runtime is slow or crashes later, the signal deadline bounds
how long a request waits. Unfinished signals follow their configured error or
unscanned policy.

## Three ways to use it

| You want to | Do this | Read |
| --- | --- | --- |
| Use the router's built-in features | Nothing extra. The router starts and supervises the runtime for you; `--platform rocm` or `--platform cuda` selects a GPU-capable image; deployment placement controls each worker. | [Run it with the router](./deploy.md) |
| Share the models between routers, or run them on another machine | Start a runtime yourself and point the router at it with `endpoint`. | [Run it with the router](./deploy.md#attach-to-a-runtime-you-run) |
| Call the models from your own code | Run `vllm-sr serve ARTIFACT --engine` and send HTTP requests. | [Quickstart](model-runtime/quickstart.md) |

## What it can serve

| Task | Built-in models | Guide |
| --- | --- | --- |
| Pick a domain, detect a need for fact checking, read user feedback, detect the requested output modality | Vela 1.0 Domain, FactCheck, Feedback, Modality | [Classify requests](./guides/classify.md) |
| Find personal information | Vela 1.0 PII | [Detect PII](./guides/pii.md) |
| Stop prompt attacks and unsafe content | Vela 1.0 Guard, Safety, Shield, Hazard | [Prompt attacks and unsafe content](./guides/safety.md) |
| Check an answer against its sources | Vela 1.0 Halu | [Hallucination checks](./guides/hallucination.md) |
| Semantic cache, memory, RAG, tool selection, embedding signals | Vela 1.0 Embedding, Qwen3-Embedding-0.6B | [Embeddings](./guides/embeddings.md) |
| Rerank retrieved documents | Vela 1.0 Reranker | [Rerank documents](./guides/rerank.md) |
| Route on images and audio | Vela 1.0 Omni Nano and Mini | [Images and audio](./guides/multimodal.md) |
| Ask your own routing questions in plain language | Decision 2.0, Decision 1.0, Vela 2.0 | [Decision models](./guides/decisions.md) |

[Choose a model](model-runtime/choose-a-model.md) helps you pick a size and hardware.

## What you can rely on

- **Pinned and verified.** Built-in models are pinned to an exact Hugging Face
  revision. Every file is checked against its recorded SHA-256 before it is
  loaded, and code shipped inside a model repository is never run.
- **Same answers as the released models.** The default `exact` profile gives
  the answers the model publishers measured. Faster settings are opt-in and say
  that they may change results. See [Profiles](./profiles.md).
- **Bounded waits.** Slow or unavailable models resolve through the signals'
  deadline and error policies. A model forward already running may continue
  after the caller times out and delay queued work. The router restarts a
  crashed runtime.
- **Batched calls.** Compatible model work from the same routing stage can
  travel together in one API call. That call may require several model forward
  passes, and later stages can make additional calls. Independent workers can
  answer in parallel when hardware capacity allows.
- **Pluggable.** New model families, engines and hardware back ends are
  ordinary Python packages. See [Add your own model family](./plugins.md).

## Hardware

CPU and AMD GPUs (MI300X, MI325X) are validated. NVIDIA GPUs work but are not
yet validated; Intel GPUs (`xpu`) and Apple GPUs (`mps`) are available and not
yet validated. Every router image runs models on the CPU. The AMD and NVIDIA
images (`vllm-sr serve --platform rocm` or `--platform cuda`) also run them on
the GPU.

## Coming from an older release?

The candle, ONNX Runtime and OpenVINO back ends are gone. Run
`vllm-sr config migrate` to update your configuration; see
[Migrate from the native bindings](model-runtime/migrate.md).
