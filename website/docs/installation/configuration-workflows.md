---
title: Configuration Workflows
description: Choose how the CLI, Dashboard, Helm, Operator, and DSL author and apply one canonical Router configuration.
---

# Configuration Workflows

All supported workflows produce or consume the same canonical configuration.
Choose one primary source of truth for a deployment and use other interfaces to
inspect or validate it, not to overwrite it independently.

## CLI and YAML

Use YAML when configuration belongs in source control or an existing deployment
pipeline:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

The local runtime derives stack-specific service addresses in runtime-owned
state without rewriting the source file. Concurrent `serve` and `stop`
operations for the same runtime and stack are serialized; retry after the
active lifecycle operation finishes.

For contributor-oriented deployment, recipe scaffolding, validation, evaluation,
and reviewed activation, see the
[vLLM-SR contributor journey](../tutorials/agent/vllm-sr-journey.md).

## Dashboard

An empty local workspace starts the Dashboard in setup mode. Use it to bind
model endpoints, choose a baseline policy, preview the result, and activate a
complete config.

After activation, the **Mixture-of-Models** workspace separates three tasks:

- **Built-in Models** discovers installed virtual models and their Model Cards;
- **Models & Routing** edits physical models, entrypoints, recipes, and routing;
- **Probes** inspects recipe scenarios and supports generation or routing-only
  validation.

Verify provider generation separately from routing evaluation. A probe can
select the expected route even when the selected backend cannot generate.

The visual DSL editor owns routing semantics. It preserves listeners, providers,
global settings, and setup state when replacing its routing surface. Multi-recipe
lifecycle changes are managed from Models & Routing or the management API so a
visual edit cannot silently discard another recipe.

## Helm

Direct Helm deployments place a complete canonical document under
`configOverride`. This replaces the chart's example configuration as one
document before the chart applies explicit Kubernetes integration rewrites; it
does not merge sample models or decisions into your policy. Author and validate
the canonical document as `config.yaml`, then place that document under
`configOverride` in the Helm values file.

```yaml
configOverride:
  version: v0.3
  listeners:
    - name: grpc-50051
      address: 0.0.0.0
      port: 50051
      timeout: 300s
    - name: http-8080
      address: 0.0.0.0
      port: 8080
      timeout: 300s
  providers:
    defaults:
      default_model: local/general
    models:
      - name: local/general
        provider_model_id: my-served-model
        backend_refs:
          - name: primary
            endpoint: model-server.default.svc.cluster.local:8000
            protocol: http
            weight: 100
  routing:
    strategy: priority
    modelCards:
      - name: local/general
        modality: text
        capabilities: [chat]
    decisions:
      - name: default-route
        description: Route requests to the configured model.
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: local/general
            use_reasoning: false
  global:
    services:
      response_api:
        enabled: false
        store_backend: memory
```

```bash
vllm-sr validate --config config.yaml

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml
```

`vllm-sr serve --target k8s --config config.yaml` passes the selected document
as an atomic override, so chart example routes cannot merge into it. The command
rejects an empty or setup-only document and does not inject local-Docker service
addresses or knowledge-base paths. Run `vllm-sr validate` first so schema and
reference errors fail before deployment.

Choose Kubernetes GPU images, resources, and device plugins through Helm or the
Operator. The local `--platform amd` and `--platform nvidia` shortcuts do not
configure Kubernetes scheduling.

## Operator

The Operator renders a canonical config from two Kubernetes-native inputs:

- `spec.vllmEndpoints` discovers model services and creates provider bindings
  and model cards; and
- `spec.config.routing` accepts the canonical routing object.

Other `spec.config` fields are typed Operator adapters for response cache,
classifiers, tools, observability, reasoning families, and related shared
settings. The Operator translates them into canonical provider and `global`
sections.

```yaml
spec:
  vllmEndpoints:
    - name: local-backend
      model: local/model
      backend:
        type: service
        service:
          name: model-server
          port: 8000
  config:
    routing:
      strategy: priority
```

Do not copy arbitrary `providers` or `global` keys directly under `spec.config`;
they are not CRD fields. See [Kubernetes Operator](k8s/operator) and the
[SemanticRouter CRD reference](../api/semantic-router-crd).

## Routing DSL

The DSL is a focused authoring surface for model cards, signals, projections,
decisions, algorithms, plugins, entrypoints, and recipes. Providers, listeners,
credentials, and global services remain YAML-owned.

Use DSL when routing policy benefits from a compact, reviewable representation.
Use canonical YAML as the complete deployment artifact.

## Avoid split ownership

- Do not edit a generated runtime config as if it were the source document.
- Do not let both GitOps and an interactive Dashboard session write the same
  deployment without an explicit handoff.
- Do not put secrets in DSL, ConfigMaps, or committed YAML.
- Do not assume an evaluated route proves backend readiness; verify generation.
- Preview and validate a complete change before applying it to a live stack.

For management endpoints and concurrency contracts, see the
[management API reference](../api/apiserver).
