# Built-in Model Runtime

## Overview

The model runtime runs every model the router uses: the classifiers behind
signals such as domain, PII and jailbreak, the embedding models behind the
semantic cache, memory and RAG, the reranker, the hallucination detector and
decision models. It is a separate process with one HTTP API. The router starts
and supervises it for you, or attaches to one you run yourself.

## What Problem Does It Solve?

Every feature that needs a model gets it the same way: one place downloads,
verifies, loads and serves models on CPU or GPU, and calls with compatible model inputs can share a native batch. Independent
logical deployments have separate managed workers. A model that is slow or not ready makes
its feature unknown instead of holding up the request.

## When to Use

You use it whenever a configured feature needs a model; there is nothing to
turn on. Configure it yourself to put a model on a GPU, to pin a different
model, to place or scale its replicas, or to attach a shared external worker.
Use `vllm-sr serve ARTIFACT --engine` to expose native System One
requests through the same persistent frontend. Enable saved recipe routing
by starting without `--engine`; the frontend and Dashboard remain available
in both startup modes.

## Configuration

A model you choose yourself is a `model_runtime` deployment:

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
      decision-shared:
        provider: model_runtime
        endpoint: http://decision-runtime:8100
```

Without `endpoint`, the router starts the runtime for the deployment and
restarts it if it exits. With `endpoint`, it attaches to a runtime you run.

Start here:

- [Quickstart](model-runtime/quickstart.md): serve a model and use it from the router.
- [Choose a model, size and hardware](model-runtime/choose-a-model.md).
- [Run it with the router](../../model-runtime/deploy.md): devices, processes, attaching, Kubernetes.
- [Profiles](../../model-runtime/profiles.md): exact answers or faster, approximate settings.
- [Migrate from the native bindings](model-runtime/migrate.md).
- [Troubleshooting and FAQ](model-runtime/troubleshooting.md).
