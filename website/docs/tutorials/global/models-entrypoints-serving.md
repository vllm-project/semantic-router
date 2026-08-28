---
title: Models, Entrypoints, and Serving
description: Connect inference backends, compose a Mixture-of-Model, and expose stable OpenAI-compatible model names.
---

# Models, Entrypoints, and Serving

Semantic Router gives applications stable model names while operators can
change the physical models and routing policy behind them. The Dashboard is the
fastest path to a working topology; YAML remains available for reviewed,
version-controlled deployments.

## The four objects

| Object | What it represents |
| --- | --- |
| Model | One logical model connected to one or more inference endpoints. |
| Recipe | Reusable signals, projections, decisions, algorithms, and plugins. |
| Mixture-of-Model | A Recipe whose decisions have been assigned connected Models. |
| Entrypoint | One or more public model names that resolve to that Mixture-of-Model. |

```text
client model -> entrypoint -> recipe decision -> selected model -> inference endpoint
```

The public model name is the client contract. It does not identify a checkpoint
and is never forwarded as the selected backend model ID.

## Build a model path

### 1. Connect Models

Start the stack and open the Dashboard:

```bash
vllm-sr serve
vllm-sr dashboard
```

Open **Build → Models**, choose a provider, and enter the endpoint credentials.
For compatible providers the Dashboard discovers available model IDs, so you
can import several in one step. Use **Advanced settings** only when you need to
override metadata, pricing, or connection behavior. Verify each connection
before assigning it to a Recipe.

### 2. Choose a Recipe

Open **Build → Mixture-of-Models → Recipes**. A Recipe describes the routing
logic without embedding provider URLs or credentials. Review its decisions and
probes, or create a custom Recipe from the Signals, Projections, and Decisions
you already maintain.

### 3. Publish a Mixture-of-Model

In **Models**, create a Mixture-of-Model, choose the Recipe, and assign eligible
connected Models to each decision. A decision can use one Model or an ordered
set when the algorithm supports multiple candidates. Add concise public aliases
and publish only after the topology and probes are complete.

### 4. Test in Playground

Select the new public model in **Playground** and send a representative request.
The response metadata shows the decision, algorithm, selected Model, latency,
TTFT, and TPOT without interrupting the conversation. Use **Insights** for a
deeper routing trace and cost comparison.

### 5. Call the OpenAI-compatible API

List the public model names exposed by the running stack:

```bash
curl -sS http://localhost:8899/v1/models
```

Then use an entrypoint through the standard `model` field:

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

The Router resolves the entrypoint, evaluates only its Recipe, selects an
eligible Model, and rewrites the upstream request to the provider-facing model
ID.

## Operate the stack

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

`vllm-sr serve --config my-models.yaml` remains the explicit path for a
reviewed user-owned configuration. Kubernetes deployments use the same config
through Helm or the Operator:

```bash
vllm-sr serve --target k8s --config my-models.yaml --namespace semantic-router
```

Keep provider credentials in environment bindings or Kubernetes Secrets, not
in Recipe assets, ConfigMaps, shell history, or committed Helm values.

## Move a custom Recipe

Package a reviewed Recipe directory for transport:

```bash
vllm-sr recipe pack path/to/custom-recipe
```

The archive contains routing policy, not physical model credentials or runtime
dependencies. On the target host, authorize every required environment variable
by name and serve the complete configuration:

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

- [Virtual Models](entrypoints-and-recipes) for request resolution and isolation.
- [Entrypoints](entrypoints) for naming and validation rules.
- [Recipes](recipes) for lifecycle behavior and limitations.
- [Mixture of Models](../../overview/mom-model-family) for the MoM architecture.
- [Configuration Workflows](../../installation/configuration-workflows) for YAML, Dashboard, Helm, Operator, and DSL ownership.
