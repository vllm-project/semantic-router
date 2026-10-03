---
title: Add your own model family
sidebar_label: Add your own model family
description: Serve a new kind of model by installing a small Python plugin, with a complete, tested example.
---

# Add your own model family

The runtime is built from plugins. A **model family** knows a model format:
how to read the package, how to turn a request into model input and how to
turn the model output into an answer. An **engine** runs the network itself,
an **accelerator** drives a kind of hardware, and a **profile** decides
numerics and batching. The built-in families, engines, accelerators and
profiles register exactly the way yours will, so a plugin is a first-class
part of the runtime.

You need a plugin when you want to serve a model the built-in families do not
understand. You do not need one for Hugging Face ModernBERT or mmBERT
classifiers and embedding models; those load as they are.

## The example plugin

The repository contains a complete plugin small enough to read in one
sitting:
[`src/model-runtime/examples/third_party_plugin`](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/examples/third_party_plugin).
It adds a keyword "model" family and the engine that runs it. A package is one
JSON file that maps labels to keywords; the family answers `/v1/classify`,
`/v1/embeddings` and `/v1/rerank` from keyword counts. The runtime's test
suite installs it through its entry points and serves it next to a decision
model in one process, on every endpoint.

Try it:

```bash
pip install ./src/model-runtime ./src/model-runtime/examples/third_party_plugin
mkdir -p /tmp/keywords && cat > /tmp/keywords/example_model.json <<'JSON'
{"format": "vllm-sr-example/1", "labels": ["billing", "shipping", "other"],
 "keywords": {"billing": ["refund", "invoice", "charge"], "shipping": ["parcel", "delivery"]}}
JSON
vllm-sr-runtime serve /tmp/keywords --engine example_counts --device cpu --port 8100
```

```bash
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Please refund the invoice", "Where is my parcel?"]}'
vllm-sr-runtime plugins
```

The first request returns `billing` for the first text and `shipping` for the
second. `vllm-sr-runtime plugins` and `GET /v1/models` list
`example_keywords` and `example_counts` with their distribution and version.

## Write your own

A plugin is an ordinary Python distribution.

**1. The family.** Subclass `ModelFamily` and `LoadedModel` from
`vllm_sr_runtime.plugins.base`:

| Method | What it does |
| --- | --- |
| `detect(package)` | Cheap check: is this directory or repository yours? Read one small file, never the weights. |
| `verify(package)` | Read the package's files, check them and return their identity (`model_sha256`), limits and licence. |
| `describe(package)` | Say what the engine must run: the backbone, the weight files and the numerics. |
| `load(package, spec, engine_model)` | Return the loaded model with its `ModelInfo`: the surfaces it serves, its heads and labels, its embedding and rerank views. |
| `plan_surface(surface, request)` | Validate one request and turn it into work items. |
| `run(items)` | One pass of the engine over a batch of items. |
| `finish_surface(plan, results)` | Turn the results into the response body of that endpoint. |

Declare the endpoints you serve in `surfaces` and describe the plugin in
`descriptor()`; `/v1/models` shows it to clients.

**2. The engine, if you need one.** Most families reuse the built-in `native`
(PyTorch) or `onnxruntime` engine by describing their network in the
`ModelSpec`. Write an engine (`Engine` and `EngineModel`: `supports`, `load`,
`forward` or `encode`) only for a new kind of network or a new execution
library.

**3. Register it** in your `pyproject.toml`:

```toml
[project.entry-points."vllm_sr_runtime.families"]
example_keywords = "vllm_sr_example.family:KeywordFamily"

[project.entry-points."vllm_sr_runtime.engines"]
example_counts = "vllm_sr_example.engine:CountsEngine"
```

The groups are `vllm_sr_runtime.families`, `vllm_sr_runtime.engines`,
`vllm_sr_runtime.accelerators` and `vllm_sr_runtime.profiles`. A name that is
already taken is refused at startup.

**4. Make it fast.** Two flags turn on the runtime's shared optimizations:

- set `cache_key` on work items whose result depends only on their content,
  so repeated inputs are answered from the result cache;
- set `fuse_bundled_jobs = True` on the loaded model when one pass can serve
  several requests that arrive together in a bundle.

**5. Test it** the way the example is tested: install the distribution, start
a runtime on a small package and check every endpoint against the OpenAPI
contract
([`tests/test_third_party_plugin.py`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/tests/test_third_party_plugin.py)).

## Use it from the router

Install your plugin next to the runtime the router uses, then bind a feature
to a deployment of your model. For a runtime you run yourself, attach to it:

```yaml
global:
  model_catalog:
    deployments:
      ticket-topics:
        provider: model_runtime
        endpoint: http://runtime.internal:8100
        served_name: keywords
```

A custom classifier signal can then read its labels; see
[Classify requests](./guides/classify.md#use-your-own-classifier). The router
checks the binding against the labels your model reports in `/v1/models`.

## Rules for plugins

- Plugins run inside the runtime process. Install only plugins you trust.
- The runtime never runs code shipped inside a model package. If a format
  needs code, that code belongs in your plugin.
- Keep the layers apart: a family never imports an engine, and an engine never
  reads a package. That is what lets your family run on a built-in engine, or
  your engine serve a built-in family.
