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
vllm-sr config init --output config.yaml
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

- **Start from a template.** `config init` writes the packaged minimal
  canonical template, and refuses to replace an existing file unless you pass
  `--force`.
- **Or from the running Router.** When a Router is already running, start from
  `vllm-sr config get`, so unrelated active settings are preserved.
- **Models and decisions.** Declare physical models in `providers.models`, and
  reference their names in each decision's `modelRefs` under
  `recipes[].decisions`. Built-in recipes get their model assignments when an
  entrypoint is published.
- **Model facts** stay on the shared `providers.models` entries. Add what the
  selected algorithm or capability needs, such as context limits or LoRA
  adapters.

The local runtime keeps stack-specific service addresses in its own state,
without rewriting the source file. Concurrent `serve` and `stop` operations on
the same stack run one at a time; retry after the active one finishes.

## Dashboard

An empty local workspace starts the Dashboard in setup mode. Use it to bind
model endpoints, choose a baseline policy, preview the result, and activate a
complete config; the waiting `vllm-sr serve` then starts the Router from it.

After activation, open **Build → Routing → Models** to connect physical Chat
backends. **Mixture-of-Models** has three tabs:

- **Models** publishes and manages public model entrypoints;
- **Recipes** manages the policies those entrypoints select and lists the
  built-in Recipes installed with vLLM-SR; **Create model** on a built-in
  Recipe assigns your Models to its decisions and publishes it;
- **Probes** inspects recipe scenarios and supports generation or routing-only
  validation.

**Build → System One** separately provides **Decision Models**, **Decision
Playground**, and **Decision Monitoring** for judgment deployments, typed
questions, and their runtime observations.

Verify provider generation separately from routing evaluation. A probe can
select the expected route even when the selected backend cannot generate.

The visual DSL editor owns routing semantics. It preserves listeners, providers,
global settings, and setup state when replacing its routing surface. Multi-recipe
lifecycle changes are managed from Mixture-of-Models or the management API so a
visual edit cannot silently discard another recipe.

## Helm

Direct Helm deployments place a complete canonical document under
`configOverride`. This replaces the chart's example configuration as one
document before the chart applies explicit Kubernetes integration rewrites; it
does not merge sample models or decisions into your policy. Author and validate
the canonical document as `config.yaml`, then place that document under
`configOverride` in the Helm values file.

```yaml title="values.yaml"
configOverride:
  version: v0.5
  listeners:
    - name: http-8899
      address: 0.0.0.0
      port: 8899
      timeout: 300s
  providers:
    defaults:
      model: local/general
    models:
      - name: local/general
        provider_model_id: my-served-model
        capabilities: [chat]
        backend_refs:
          - name: primary
            endpoint: model-server.default.svc.cluster.local:8000
            protocol: http
            provider: vllm
            weight: 100
  recipes:
    - name: default
      strategy: priority
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
vllm-sr config validate --config config.yaml

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml
```

Helm values seed the Router ConfigMap at install:

- **Live edits survive upgrades.** A config edit saved later by the Dashboard
  or Router API is kept by an ordinary Helm upgrade.
- **Apply a revised `configOverride` on purpose** by changing
  `configMap.applyValuesRevision` for that upgrade. Reusing the same revision
  later keeps subsequent live edits.
- **Rollout.** A saved ConfigMap edit takes effect after the Router deployment
  rolls out.

`vllm-sr serve --target kubernetes --config config.yaml` passes the selected document
as an atomic override, so chart example routes cannot merge into it. The command
rejects an empty or setup-only document and does not inject local-Docker service
addresses or knowledge-base paths. Run `vllm-sr config validate` first so schema and
reference errors fail before deployment.

On Kubernetes, `--platform rocm` and `--platform cuda` select the ROCm or CUDA
Router image, unless an image is set explicitly. They request `amd.com/gpu` or
`nvidia.com/gpu` resources from canonical device ordinals: replicas sharing one
device need one GPU, and distinct ordinals need one each.

The pod schedules onto a node whose device plugin offers those GPUs. Node
selectors, tolerations, other resources and the device plugin itself come from
Helm values or the Operator.

The chart runs the Dashboard as its own Deployment and Service, and that
Deployment is disabled by default. The Router Service carries the gRPC and HTTP
API ports only, so port 8700 appears in the cluster only after the Dashboard is
enabled.

```bash
helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml --set dashboard.enabled=true

kubectl --namespace vllm-semantic-router-system port-forward \
  svc/semantic-router-dashboard 8700:8700
```

## Operator

The Operator renders a canonical config from two Kubernetes-native inputs:

- `spec.vllmEndpoints` discovers model services and creates provider bindings;
  and
- `spec.config.recipes`, `spec.config.entrypoints`, `spec.config.plugins` and
  `spec.config.models` accept canonical recipes, entrypoints, default plugins
  and model facts.

- **The model catalog.** `spec.config.model_deployments`, `model_bindings` and
  `decision_model` carry it; see
  [Model deployments and the decision model](../api/semantic-router-crd#model-deployments-and-the-decision-model).
- **Typed adapters.** The other `spec.config` fields, such as the response
  cache, reasoning effort, API and observability settings, are typed Operator
  adapters. The Operator translates them into canonical `providers` and
  `global` sections.
- **Refused fields.** `spec.config.routing` and the fields of removed features,
  such as `classifier` and `prompt_guard`, are refused.

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
    recipes:
      - name: default
        strategy: priority
```

Do not copy arbitrary `providers` or `global` keys directly under `spec.config`;
they are not CRD fields. See [Kubernetes Operator](k8s/operator) and the
[SemanticRouter CRD reference](../api/semantic-router-crd).

## Routing DSL

The DSL is a focused authoring surface for model facts, signals, projections,
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
