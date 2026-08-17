---
title: Models and Recipes
description: Discover built-in virtual models, bind provider backends, fork routing policies, and move custom recipes between environments.
---

# Models and Recipes

Semantic Router distinguishes the model name a client requests from the model
endpoint that ultimately generates a response.

| Object | Meaning |
| --- | --- |
| Provider model | A logical name bound to one or more physical inference endpoints. |
| Virtual model | A public model ID that selects a routing objective. |
| Entrypoint | The mapping from public model aliases to a recipe. |
| Recipe | The isolated routing policy and runtime state for an entrypoint. |
| Model Card | The intended use, requirements, data handling, evaluation, and limitations of a maintained recipe or virtual model. |

## Discover built-in models

The `vllm-sr` package includes a versioned catalog of maintained virtual models:

```bash
vllm-sr model list
vllm-sr model show vllm-sr/mom-v1-blend
```

`model show` reports the public entrypoints, intended use, backend roles,
minimum pool requirements, and recommended candidates. Recommendations are
starting points, not mandatory vendor IDs.

By default, list commands use the installed `latest` catalog. Use
`--all-versions` to inspect release snapshots, `--catalog-version vX.Y` to
select one, and `--all` to include entries incompatible with the current CLI or
Router feature set.

## Serve a built-in objective

The local Docker target accepts one or more compatible virtual model IDs:

```bash
vllm-sr serve vllm-sr/mom-v1-blend
vllm-sr serve vllm-sr/mom-v1-lite vllm-sr/mom-v1-flash
```

This starts Router, Envoy, Dashboard, and supporting services. It does not
download or start the provider models referenced by the asset. Start or bind
those inference endpoints first.

Bare `vllm-sr serve` retains the normal setup or `config.yaml` workflow; it does
not silently choose the catalog default.

## Fork and customize

Create a user-owned canonical config before changing a built-in policy or
provider pool:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
vllm-sr model validate mom-v1.yaml
vllm-sr serve --config mom-v1.yaml
```

An untouched deterministic fork retains its verified catalog provenance.
Editing or rebinding it produces a custom, unverified config. Validation checks
the document; it does not certify backend quality or availability.

## Dashboard workflow

In **Built-in Models**, read the Model Card before adding a virtual model. Then:

1. bind the required provider roles under **Models & Routing**;
2. use **Verify** to send a real generation request to each backend;
3. inspect representative scenarios under **Probes**;
4. use **Validate** for routing-only evaluation or **Run** for generation; and
5. preview configuration and topology changes before activation.

Routing validation and backend verification answer different questions; review
both before production use.

## Package a custom Recipe

`vllm-sr recipe pack` creates a transport archive from a custom recipe
directory:

```bash
vllm-sr recipe pack path/to/custom-recipe
```

The archive does not become a built-in catalog entry and does not install
provider models or other runtime dependencies. Review its checksum, Model Card,
configuration, required environment bindings, and provider topology before use.

Packaging preserves environment references instead of expanding them. Literal
credentials, sensitive headers, credential-bearing URLs, unsafe YAML forms, and
unsupported local-process transports are rejected. Export each required value
on the target host and authorize its name when serving:

```bash
export PROVIDER_API_KEY=...
vllm-sr serve --config path/to/recipe/config.yaml \
  --recipe-env PROVIDER_API_KEY
```

## Import and migrate

Migrate an older flat or mixed configuration into the canonical structure:

```bash
vllm-sr config migrate --config old-config.yaml
```

Import supported OpenClaw provider endpoints and point OpenClaw at the first
Router listener:

```bash
vllm-sr config import --from openclaw \
  --source openclaw.json \
  --target config.yaml
```

The setup Dashboard can also import a complete canonical YAML document from an
HTTPS URL. Treat remote configuration as code: review the source, provider
endpoints, credentials, plugins, and data stores before activation.

## Next

- [Mixture of Models](../overview/mom-model-family) for the virtual-model model.
- [Recipe Model Cards](https://github.com/vllm-project/semantic-router/tree/main/config/recipes)
  for maintained examples.
- [Recipe authoring and conformance](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md)
  for custom recipe requirements.
- [Configuration Workflows](configuration-workflows) for deployment ownership.
