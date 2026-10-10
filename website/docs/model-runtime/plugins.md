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
It adds a keyword "model" family and the engine that runs it, plus an
accelerator and a profile, so it shows all four kinds of plugin. A package is
one JSON file that maps labels to keywords; the family answers `/v1/classify`,
`/v1/embeddings` and `/v1/rerank` from keyword counts. The runtime's test
suite builds and installs its wheel, discovers it through its entry points and
serves it next to a decision model in one process, on every endpoint, and on
the example's own accelerator and profile.

Try it:

```bash
pip install ./src/model-runtime ./src/model-runtime/examples/third_party_plugin
mkdir -p /tmp/keywords && cat > /tmp/keywords/example_model.json <<'JSON'
{"format": "vllm-sr-example/1", "labels": ["billing", "shipping", "other"],
 "keywords": {"billing": ["refund", "invoice", "charge"], "shipping": ["parcel", "delivery"]}}
JSON
vllm-srun serve /tmp/keywords --engine example_counts --device cpu --port 8100
```

```bash
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Please refund the invoice", "Where is my parcel?"]}'
vllm-srun plugins
```

The first request returns `billing` for the first text and `shipping` for the
second. `vllm-srun plugins` lists `example_keywords` among the families,
`example_counts` among the engines, `example_host` among the accelerators and
`example_one_by_one` among the profiles; `GET /v1/models` also shows the
distribution and version each came from.

The example's accelerator offers the host CPU as a device of its own, and its
profile runs every request alone, in arrival order. Name them like the
built-in ones:

```bash
vllm-srun serve /tmp/keywords --engine example_counts --device example_host --profile example_one_by_one --port 8100
```

For this classify-only plugin, run an explicit worker container from an image
that has the plugin installed. The instance's System One frontend requires a
decision-capable model. The example ships a Dockerfile
that adds it to the router image. From the repository root:

```bash
docker build -t vllm-sr-example src/model-runtime/examples/third_party_plugin
docker run --rm -p 127.0.0.1:8100:8100 -v /tmp/keywords:/app/keywords:ro --entrypoint vllm-srun vllm-sr-example serve /app/keywords --device example_host --profile example_one_by_one --host 0.0.0.0 --port 8100
```

The container reads `/tmp/keywords` through a read-only mount. The runtime
selects the example's engine and rejects uninstalled devices or profiles.
A configured instance can attach to this worker using `endpoint` and its model
name; the classify endpoint remains an independently operated worker API.

## Write your own

A plugin is an ordinary Python distribution.

**1. The family.** Subclass `ModelFamily` and `LoadedModel` from
`vllm_srun.plugins.base`:

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

A family that answers questions on `/v1/decisions` subclasses `DecisionModel`
from `vllm_srun.plugins.decisions` instead of `LoadedModel`. It writes
`plan` (turn a request's questions into work items) and `answer` (one
question's answer from its result); `DecisionModel` serves the endpoint, its
startup self-check and its per-question metrics.

A family can also ship models of its own, pinned to a revision: name a module
whose `MODELS` lists them (`BuiltinModel` entries with the repository,
revision, identity and recorded golden answers) in `builtin_table`. The
runtime then serves them by repository ID, or by bare model name while no
other organisation's built-in model has that name, and checks their answers
at startup. A table that fails to import pins nothing, and the runtime logs
why. A table that pins a repository another family's table pins, or lists
another family's model, stops every model of the process from loading,
built-in ones included, until you remove the conflict. Name the module that
writes tiny test packages in
`fixture_writer`, and `vllm-srun fixture --family <name>` writes one.
The built-in families declare both the same way.

**2. The engine, if you need one.** Most families reuse the built-in `native`
(PyTorch) or `onnxruntime` engine by describing their network in the
`ModelSpec`. Write an engine (`Engine` and `EngineModel`: `supports`, `load`,
`forward` or `encode`) only for a new kind of network or a new execution
library. Set `auto_priority` if `engine: auto` should try your engine before
others (lower first; the built-in `native` engine is 0); without it, `auto`
tries it after the engines that set one, by name. Build your `descriptor()`
on `super().descriptor()`, which lists `auto_priority` on the model cards. If
your `load` reads weights from disk, override `read` too: it does that host
work before the runtime takes the device and returns the device work that
finishes the load, so the device's other models keep answering while it
reads. Without it, all of `load` is device work.

**3. An accelerator or a profile, if you need one.** For new hardware,
subclass `Accelerator` (`available`, `devices`, `torch_device`, `kernels`), and
set `auto_priority` if `device: auto` may pick it; without it, only a request
that names the device uses it. For a new way to run requests together,
subclass `Profile` (`plan`, and `bind` for what it reads from the model).

**4. Register it** in your `pyproject.toml`:

```toml
[project.entry-points."vllm_srun.families"]
example_keywords = "vllm_sr_example.family:KeywordFamily"

[project.entry-points."vllm_srun.engines"]
example_counts = "vllm_sr_example.engine:CountsEngine"

[project.entry-points."vllm_srun.accelerators"]
example_host = "vllm_sr_example.accelerator:HostAccelerator"

[project.entry-points."vllm_srun.profiles"]
example_one_by_one = "vllm_sr_example.profile:OneByOneProfile"
```

The groups are `vllm_srun.families`, `vllm_srun.engines`,
`vllm_srun.accelerators` and `vllm_srun.profiles`. A name that is
already taken is refused at startup.

**5. Make it fast.** Two flags turn on the runtime's shared optimizations:

- set `cache_key` on work items whose result depends only on their content,
  so repeated inputs are answered from the result cache;
- set `fuse_bundled_jobs = True` on the loaded model when one pass can serve
  several requests that arrive together in a bundle.

**6. Test it** the way the example is tested: install the distribution, start
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
checks the binding against the labels your model reports in `/v1/models`. A
deployment names your accelerator in `device` and your profile in `profile`
the same way; the router checks only their form, and the runtime refuses a
name it has no plugin for and lists the names it has.

## Rules for plugins

- Plugins run inside the runtime process. Install only plugins you trust.
- The runtime never runs code shipped inside a model package. If a format
  needs code, that code belongs in your plugin.
- Keep the layers apart: a family never imports an engine, and an engine never
  reads a package. That is what lets your family run on a built-in engine, or
  your engine serve a built-in family.
