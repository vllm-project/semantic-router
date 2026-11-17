---
title: Model runtime
sidebar_label: Overview
sidebar_position: 1
slug: /model-runtime/overview
description: The decision model and the embedding model the Router asks run in the built-in model runtime. Start here.
---

# Model runtime

The **built-in model runtime** serves the models the Router consults about each
request: the [decision model](../../system-one/get-started/overview.md), which
answers every System One question, and the embedding model that recall
features use. You don't install it separately: it ships inside the Router
images, and the Router starts and supervises it for you.

```bash title="Serve a decision model and ask it directly"
vllm-sr serve vllm-sr/vela3-fast --engine --platform cpu
curl -s localhost:8899/v1/systemone/models
```

:::note
The runtime is not where your chat models run. The models that answer your
users stay behind your providers (vLLM, Ollama, a hosted API); the runtime
serves the small models the Router asks about each request.
:::

![Frontend, optional decision engine, and on-demand model runtime](/img/architecture/system-one/01-component-composition.svg)

## Three ways to use it

| You want to | Do this | Read |
| --- | --- | --- |
| Use the Router's built-in features | Nothing extra. The Router starts the runtime for you; `--platform rocm` or `--platform cuda` selects a GPU-capable image | [Connect the Router](../deploy/connect-the-router.md) |
| Share the models between Routers, or run them on another machine | Start a runtime yourself and point the Router at it with `endpoint` | [Attach a runtime you run](../deploy/connect-the-router.md#attach-to-a-runtime-you-run) |
| Call the models from your own code | Run `vllm-sr serve ARTIFACT --engine` and send HTTP requests | [Quickstart](./quickstart.md) |

Managed workers run as `vllm-srun` processes inside the Router container.
Starting `vllm-sr serve ARTIFACT --engine` keeps the same instance frontend and
model management, with Chat routing disabled. Router mode can serve native
System One requests at the same time as routed Chat requests.
[Component Architecture](../../overview/component-architecture.md) covers the
request paths.

## What it can serve

| Task | Built-in models | Guide |
| --- | --- | --- |
| Answer every routing judgment: the application signals, your own questions, the System One selector, guards and judges | Vela 3.0 (vela3-fast by default), Decision 3.0, Decision 2.0 and 1.0, Vela 2.0 | [Supported models](../supported-models/overview.md) |
| Recall for the response cache, memory and RAG | Vela 1.0 Embedding, Qwen3-Embedding-0.6B | [Embeddings](../guides/embeddings.md) |

[Choose a model](./choose-a-model.md) helps you pick a size and hardware.

## How it starts and fails

- **Only what you use loads.** Built-in features resolve their model defaults
  automatically
  ([Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md)).
  The Router starts managed workers only for deployments that something
  resolves to, and for explicitly published native models. An unused
  deployment loads no weights.
- **Startup waits for required models.** Attached model readiness follows the
  [readiness rules](../deploy/readiness-and-monitoring.md#when-a-model-is-not-ready).
- **A slow model never stalls a request.** The signal deadline bounds how long
  a request waits; an unfinished signal gives no answer and resolves through
  its `on_error` or `on_unknown` policy.

## What you can rely on

- **Pinned and verified.** Built-in models are pinned to an exact Hugging Face
  revision. Every file is checked against its recorded SHA-256 before it is
  loaded, and code shipped inside a model repository is never run.
- **Throughput first, at the published precision.** Like vLLM, the runtime
  batches concurrent requests by default and runs each model in the precision
  it was published in. A batched answer can move by rounding, which the
  records measure per model; a faster setting whose records don't show
  unchanged accuracy is an explicit switch. See
  [Engine arguments](../deploy/engine-args.md).
- **Bounded waits.** Slow or unavailable models resolve through the signals'
  deadline and error policies. A model forward already running may continue
  after the caller times out and delay queued work. The Router restarts a
  crashed runtime.
- **Batched calls.** Compatible model work from the same routing stage can
  travel together in one API call. That call may require several model forward
  passes, and later stages can make additional calls. Independent workers can
  answer in parallel when hardware capacity allows.
- **Pluggable.** New model families, engines and hardware back ends are
  ordinary Python packages. See [Add your own model family](../extend/plugins.md).

## Hardware

| Hardware | Status |
| --- | --- |
| CPU | Validated. Every Router image runs models on the CPU |
| AMD GPUs (MI300X, MI325X) | Validated. `vllm-sr serve --platform rocm` |
| NVIDIA GPUs | Validated. `vllm-sr serve --platform cuda` |
| Intel GPUs (`xpu`), Ascend NPUs (`npu`), Apple GPUs (`mps`) | Available, not yet validated |

[Choose a model](./choose-a-model.md#hardware) has the details, and
[Quantization](../quantization/overview.md) the reduced-precision packages
each GPU runs.

:::tip Coming from an older release?
The candle, ONNX Runtime and OpenVINO back ends are gone. Run
`vllm-sr config migrate` to update your configuration; see
[Migrate from the native bindings](../deploy/migrate.md).
:::
