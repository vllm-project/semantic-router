---
title: Models, Entrypoints, and Serving
description: Discover virtual models, expose stable entrypoints, connect inference backends, and operate Semantic Router from the CLI.
---

# Models, Entrypoints, and Serving

## Overview

Semantic Router gives applications stable model names while operators retain
control over the models and routing policy behind them. This guide covers the
complete CLI path from discovering a built-in virtual model to serving requests.

## Understand the four objects

| Object | What it represents |
| --- | --- |
| Provider model | A logical model connected to one or more physical inference endpoints. |
| Virtual model | A public model ID such as `vllm-sr/mom-v1-flash` that promises a routing objective. |
| Entrypoint | The mapping from one or more public model names to a recipe. |
| Recipe | The isolated signals, decisions, algorithms, plugins, and runtime state for that objective. |

```text
client model -> entrypoint -> recipe -> selected provider model -> inference endpoint
```

The virtual model name is the client contract. It does not identify a model
checkpoint, and it never reaches the selected provider backend.

## What Problem Does It Solve?

Without an entrypoint, clients must know which physical model to call and carry
deployment-specific model names in application code. A stable virtual model
lets operators change backend pools, routing logic, or rollout strategy without
changing the client contract.

## When to Use

Use this workflow when you want to serve a built-in routing objective, publish
several objectives from one Router, or fork a maintained model into a
user-owned configuration. Use a direct provider endpoint when one physical
model is the intentional and durable client contract.

## Start with a built-in virtual model

### 1. Discover the catalog

The installed CLI contains a versioned catalog of maintained virtual models:

```bash
vllm-sr model list
vllm-sr model show vllm-sr/mom-v1-blend
```

`model show` reports the entrypoint, intended use, required backend roles,
recommended candidates, compatibility, and verified asset digest. Recommended
models are starting points, not mandatory vendor IDs.

### 2. Serve one or more entrypoints

Pass catalog model IDs as positional arguments to the local Docker target:

```bash
vllm-sr serve vllm-sr/mom-v1-blend

vllm-sr serve \
  vllm-sr/mom-v1-lite \
  vllm-sr/mom-v1-flash \
  vllm-sr/mom-v1-ultra \
  vllm-sr/mom-v1-vault
```

This starts Router, Envoy, Dashboard, and supporting services. Each request
should name the entrypoint it needs; serving several entrypoints does not turn
their command-line order into a routing fallback chain.

Bare `vllm-sr serve` is intentionally different: it uses `config.yaml` or opens
the Dashboard-first setup flow. It does not silently enable the catalog default.

### 3. Connect physical inference backends

Catalog models are routing products, not checkpoint installers. `serve` does
not download or start the physical vLLM, Ollama, or hosted API models referenced
by a recipe.

Choose either workflow:

- In the Dashboard, open **Built-in Models**, review the Model Card, bind each
  required role under **Models & Routing**, and use **Verify** to send a real
  generation request to every backend.
- Fork the built-in model, edit `providers.models[].backend_refs[]`, validate the
  resulting YAML, and serve that user-owned configuration.

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
vllm-sr model validate mom-v1.yaml
vllm-sr serve --config mom-v1.yaml
```

An untouched deterministic fork retains verified catalog provenance. Changing
the provider pool or routing policy makes it a custom, unverified configuration.
Validation checks structure and references; it does not certify backend quality
or availability.

### 4. Discover and call entrypoints

List the model names exposed by the running Router:

```bash
curl -sS http://localhost:8899/v1/models
```

Then use a virtual model through the standard OpenAI-compatible `model` field:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/mom-v1-flash",
    "messages": [
      {"role": "user", "content": "Summarize the release notes."}
    ]
  }'
```

The Router resolves the entrypoint, evaluates only its recipe, selects an
eligible provider model, and rewrites the backend request to that model's
provider-facing ID.

### 5. Operate the stack

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

Use `status` before assuming ports or workspace paths. If the Router is ready
but generation fails, verify the selected backend's network address, model ID,
credentials, and `/v1/models` response.

## `vllm-sr model` command reference

### List models

```bash
vllm-sr model list
vllm-sr model list --output json
vllm-sr model list --all-versions
vllm-sr model list --catalog-version latest
vllm-sr model list --all
vllm-sr model list --config my-config.yaml
```

- The default view shows compatible models from the installed `latest` catalog.
- `--all-versions` includes every immutable release snapshot installed with the
  CLI.
- `--all` includes incompatible entries and their compatibility reason.
- `--config` inspects provider and virtual models in one explicit configuration
  instead of reading the installed catalog.

### Inspect one built-in model

```bash
vllm-sr model show vllm-sr/mom-v1-blend
vllm-sr model show --output json vllm-sr/mom-v1-blend
```

Read the Model Card and backend-role requirements before deploying a model.

### Fork one or more models

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml

vllm-sr model fork vllm-sr/mom-v1-lite mom-v1.yaml \
  --enable vllm-sr/mom-v1-flash \
  --default vllm-sr/mom-v1-flash
```

Compatible assets merge fail-closed: conflicting shared settings, providers,
entrypoints, or recipes produce an error instead of an implicit override.

### Validate a fork or user-owned config

```bash
vllm-sr model validate mom-v1.yaml
```

This runs canonical configuration validation and reports whether the document
still matches a verified catalog projection.

## Configuration

A custom entrypoint maps public aliases to a named recipe:

```yaml
entrypoints:
  - model_names:
      - company/assistant-fast
      - company/assistant-interactive
    recipe: fast

recipes:
  - name: fast
    description: Prefer the lowest-latency eligible backend.
    routing:
      strategy: priority
      decisions: []
```

Both names select the same isolated policy. Use a concrete provider model name
only when the caller deliberately needs to bypass signals, decisions,
algorithms, and recipe-local plugins.

See [Entrypoints and Recipes](entrypoints-and-recipes) for request resolution
and [Recipes](recipes) for policy isolation and lifecycle behavior.

## Docker and Kubernetes

Positional catalog models currently target local Docker serving. For
Kubernetes, fork the selected model into a user-owned config and deploy that
config through Helm or the Operator:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
vllm-sr serve --target k8s --config mom-v1.yaml --namespace semantic-router
```

Kubernetes credentials and sensitive environment values belong in Secrets,
not in catalog assets, ConfigMaps, shell history, or Helm values committed to
source control.

## Package and move a custom recipe

Package a reviewed custom recipe directory for transport:

```bash
vllm-sr recipe pack path/to/custom-recipe
```

The archive does not become a built-in model and does not install provider
models or runtime dependencies. On the target host, authorize each required
environment variable by name:

```bash
export PROVIDER_API_KEY=...
vllm-sr serve --config path/to/recipe/config.yaml \
  --recipe-env PROVIDER_API_KEY
```

For an older configuration, migrate it explicitly before serving:

```bash
vllm-sr config migrate --config old-config.yaml
```

## Next

- [Entrypoints and Recipes](entrypoints-and-recipes) for routing semantics.
- [Entrypoints](entrypoints) for naming, discovery, and validation rules.
- [Recipes](recipes) for isolation, lifecycle APIs, and limitations.
- [Mixture of Models](../../overview/mom-model-family) for the serving-system
  architecture behind MoM V1.
- [Configuration Workflows](../../installation/configuration-workflows) for
  ownership across YAML, Dashboard, Helm, the Operator, and DSL.
- [Container Connectivity](../../troubleshooting/container-connectivity) when
  Router or backend networking fails.
