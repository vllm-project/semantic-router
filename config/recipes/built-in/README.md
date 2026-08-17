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

## Discover a model

```bash
vllm-sr model list
vllm-sr model show vllm-sr/mom-v1-blend
```

`model list` shows compatible models from the installed `latest` catalog. Use
`--all-versions` to inspect installed release catalogs, or `--all` to include
models that are incompatible with the current CLI or Router feature set.

## Serve a model

After the physical backends described by its Model Card are reachable:

```bash
# One routing profile
vllm-sr serve vllm-sr/mom-v1-blend

# Several profiles sharing the same provider pool
vllm-sr serve \
  vllm-sr/mom-v1-lite \
  vllm-sr/mom-v1-flash
```

Catalog IDs must be passed explicitly. Bare `vllm-sr serve` keeps the normal
setup or config-file flow and does not select the catalog default implicitly.

The local shorthand starts the routing stack only. Kubernetes users should
fork or materialize a config and deploy it with the Helm chart or operator.

## Customize a model

Fork a built-in model before changing provider bindings, replicas, routing
rules, or enabled entrypoints:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-custom.yaml \
  --enable vllm-sr/mom-v1-vault \
  --default vllm-sr/mom-v1-blend

vllm-sr model validate mom-custom.yaml
vllm-sr serve --config mom-custom.yaml
```

Recommended backends describe capabilities and pool shape, not mandatory
vendors. A modified fork is reported as custom so users can distinguish it
from the maintainer-reviewed catalog asset.

## Catalog versions

- `latest` is the catalog bundled with the installed CLI build.
- `vX.Y` identifies an installed release snapshot for reproducible inspection,
  validation, and forking.
- A future family major, such as `mom-v2-blend`, uses a new public model ID and may
  coexist with MoM V1.

Compatibility is declared by catalog metadata rather than inferred from a
model name. Use `--catalog-version vX.Y` when a specific installed release is
required.

## Limitations

- Serving a catalog model does not prove that its provider backends can
  generate; verify those endpoints separately.
- Virtual model selection does not provide an implicit physical fallback.
- A Model Card describes the reference pool. Fork the config when deployment
  capabilities or data-handling requirements differ.
- Catalog shorthand is a local Docker workflow; production topology remains an
  operator responsibility.

## For contributors

Catalog source lives here under `config/recipes/built-in/`; installable package
resources are generated from it. See [Recipe authoring and
conformance](../CONFORMANCE.md) for the bundle contract.
