---
title: Models, Entrypoints, and Serving
description: Connect inference backends, compose Recipes, and publish stable Mixture-of-Model entrypoints from one control plane.
---

# Models, Entrypoints, and Serving

## Overview

Semantic Router gives applications stable model names while operators retain
control over the configured Models and routing policy behind them. The CLI starts
the deployment. The Dashboard and Router Management API own the serving graph.

## What Problem Does It Solve?

Applications need a stable model contract even as operators change providers,
backends, routing policy, and fallback order behind it.

## Understand the three resources

| Object | What it represents |
| --- | --- |
| Model | A logical model connected to one or more physical inference endpoints. |
| Recipe | An isolated set of signals, projections, decisions, algorithms, and plugins. |
| Entrypoint | The stable API model name that selects a Recipe and assigns configured Models to its decisions. |

```text
client model -> entrypoint -> recipe -> decision assignment -> inference endpoint
```

The entrypoint is the client contract. The Dashboard presents it as a
**Mixture-of-Models**; that is a product view, not a fourth resource or API. An
Entrypoint does not identify a checkpoint, and its public name never reaches the
selected provider backend.

## When to Use

Use this workflow to publish a new model product, change its Recipe, or move a
decision to a different backend without changing the client-facing API name.

## Configuration

Start the control plane, connect Models, choose a Recipe, assign each decision,
then publish the resulting entrypoint.

## Start the control plane

```bash
vllm-sr serve
# Or select another immutable bootstrap manifest.
vllm-sr serve --config /path/to/config.yaml
```

If `config.yaml` is absent, local Docker startup creates a secure managed
bootstrap, starts PostgreSQL and Valkey, and brings up Router and Dashboard in
the same run. `--config` selects another immutable v0.4 deployment bootstrap;
it does not activate a Model or Recipe.

`serve` does not select a model product or change a Recipe. Those are durable
control-plane operations and remain consistent across Router replicas.

## 1. Connect Models

Open **Build → Models**, select a provider, and enter the endpoint credentials.
For providers with discovery support, test the connection and import one or
more models. Advanced settings contain retry, timeout, stream-timeout, context,
capability, and token-cost metadata.

A Model is reusable. Configure it once, then assign it to decisions in any
compatible Recipe. Physical inference services must already be reachable;
Semantic Router does not download or launch them.

## 2. Choose or create a Recipe

Open **Build → Recipes**. A built-in Recipe supplies a complete routing shape;
a custom Recipe can reuse Signals, Projections, Decisions, Algorithms, and
Plugins created for that Recipe.

Managed Router replicas install the release's built-in Recipes automatically.
They appear in the ordinary Recipe list and are read-only. Copy the document to
a custom Recipe when you need to change its policy; a Router upgrade installs
a new version beside the old one and never silently moves an Entrypoint.

A draft remains control-plane state until it has a complete path from an
entrypoint rule through a decision to at least one Model assignment. Publishing
an incomplete graph is rejected instead of producing a partial Router config.

## 3. Create a Mixture of Models

Open **Build → Mixture of Models** and choose a Recipe. For each decision,
assign one or more Models from the configured inventory. When a decision accepts
several candidates, set their priority and fallback behavior in the assignment
rather than duplicating the Recipe.

The same Recipe revision can back several entrypoints with different model
assignments. Recipe logic stays reusable while each Mixture of Models remains a
complete, independently publishable product.

## 4. Publish and verify

Give the Mixture of Models a stable entrypoint name, review its topology, and
publish it. The topology view is available to read-only users; mutation controls
remain permission-gated.

List the model names visible to an authorized API key:

```bash
curl -sS http://localhost:8899/v1/models \
  -H "Authorization: Bearer $VLLM_SR_API_KEY"
```

Then call the entrypoint through the OpenAI-compatible API:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_SR_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "model": "acme/assistant",
    "messages": [
      {"role": "user", "content": "Summarize the release notes."}
    ]
  }'
```

The Router resolves the entrypoint, evaluates only its Recipe, selects an
eligible assigned Model, and rewrites the provider request to the backend model
ID. Model visibility, quota, usage, and request logs follow the API key's
effective policy.

## Automate the same workflow

The Dashboard is an optional Router Management API client. A custom console can
use the same versioned APIs to create Models and Recipes, assign decisions,
publish Entrypoints, and inspect operations without depending on Dashboard
storage or private endpoints.

Use Management API operation resources for asynchronous publish and probe
workflows. Keep provider secrets in the configured credential store; API
responses expose references and status, never stored secret material.

## Operate the stack

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

For Kubernetes, place the complete deployment bootstrap at `./config.yaml` and
choose only infrastructure options on the command line:

```bash
vllm-sr serve \
  --target k8s \
  --namespace semantic-router
```

Kubernetes credentials and sensitive environment values belong in Secrets,
not ConfigMaps, shell history, or Helm values committed to source control.

## Package a custom Recipe

Package a reviewed Recipe directory for transport:

```bash
vllm-sr recipe pack path/to/custom-recipe
```

The archive carries Recipe assets, not provider models or credentials. Import
it through the Dashboard or Management API, bind the destination Models, and
publish a new entrypoint only after its assignments are complete.

## Next

- [Virtual Models](entrypoints-and-recipes) for entrypoint resolution and
  Recipe isolation.
- [Entrypoints](entrypoints) for naming, discovery, and validation rules.
- [Recipes](recipes) for policy composition and lifecycle behavior.
- [Mixture of Models](../../overview/mom-model-family) for the serving-system
  architecture.
- [Configuration Workflows](../../installation/configuration-workflows) for
  ownership across Dashboard, Management API, Helm, the Operator, and DSL.
- [API Keys, Access, and Usage](access-and-usage) to grant those Entrypoints,
  enforce exact quota, and attribute requests and cost.
