# Built-in Virtual Model Catalog

## Overview

The built-in catalog contains virtual models distributed with `vllm-sr`. A
virtual model is a request-facing model ID backed by a routing recipe and a set
of physical provider models. It lets clients choose a stable behavior without
knowing which backend will handle each request.

Catalog models configure routing; they do not download or start inference
engines. Start or bind the required provider services before serving a model.

## Available model families

| Family | Description | Model Card |
| --- | --- | --- |
| MoM V1 | Five balanced, cost, speed, accuracy, and privacy profiles over a shared local model pool. | [MoM V1](latest/mom-v1/README.md) |

## Build a Mixture-of-Model

Run `vllm-sr serve` and open **Build → Mixture-of-Models → Recipes** in
Dashboard. Select a built-in Recipe, assign connected Models to each decision,
and choose the public names that clients will send through the OpenAI-compatible
API. Dashboard keeps connection credentials on Models and routing policy in the
Recipe.

## Customize a model

For a reviewed YAML workflow, copy a built-in bundle before changing provider
bindings, replicas, routing rules, or entrypoints:

```bash
vllm-sr validate --config mom-custom.yaml
vllm-sr serve --config mom-custom.yaml
```

Recommended backends describe capabilities and pool shape, not mandatory
vendors. A modified fork is reported as custom so users can distinguish it
from the maintainer-reviewed catalog asset.

## Catalog versions

- `latest` is the catalog bundled with the installed distribution.
- `vX.Y` identifies an installed release snapshot for reproducible inspection,
  validation, and forking.
- A future family major, such as `mom-v2-blend`, uses a new public model ID and may
  coexist with MoM V1.

Compatibility is declared by catalog metadata rather than inferred from a
model name. Release snapshots remain immutable for reproducible deployments.

## Limitations

- Serving a catalog model does not prove that its provider backends can
  generate; verify those endpoints separately.
- Virtual model selection does not provide an implicit physical fallback.
- A Model Card describes the reference pool. Fork the config when deployment
  capabilities or data-handling requirements differ.
- Dashboard builds user-owned Mixture-of-Models; production topology remains an
  operator responsibility.

## For contributors

Catalog source lives here under `config/recipes/built-in/`; installable package
resources are generated from it. See [Recipe authoring and
conformance](../CONFORMANCE.md) for the bundle contract.
