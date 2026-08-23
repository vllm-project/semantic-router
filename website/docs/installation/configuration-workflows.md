---
title: Configuration Workflows
description: Keep deployment bootstrap, standalone manifests, and managed Router resources under one clear owner.
---

# Configuration Workflows

Semantic Router has two explicit authorities. A standalone deployment reads one
immutable routing manifest. A managed deployment stores Models, Recipes,
Entrypoints, identities, keys, and policies behind the Router Management API.
Bootstrap YAML configures the process and its infrastructure; it never becomes a
second managed-resource store.

## CLI and bootstrap YAML

Use YAML for deployment bootstrap and infrastructure settings. Start the local
control plane with one command:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve
# Or select another immutable bootstrap manifest.
vllm-sr serve --config /path/to/config.yaml
```

The local runtime derives stack-specific service addresses in runtime-owned
state without rewriting the source file. Concurrent `serve` and `stop`
operations for the same runtime and stack are serialized; retry after the
active lifecycle operation finishes.

`--config` chooses one immutable v0.4 bootstrap manifest. Dynamic Models,
Recipes, decision assignments, Entrypoints, identities, keys,
and policies live in Router-owned stores and are managed through the versioned
Management API. The CLI has no Model/Recipe operand and does not author routing
at launch time.

## Dashboard

An empty local workspace creates and starts a managed Router in one `serve`
run. Connect physical backends in **Models**; fixed-origin integrations ask only for a credential,
while private endpoints also ask for their base URL. Discovery can import one
or many Provider model IDs into Router-owned Model resources.

The **Mixture-of-Models** workspace then separates the two concepts users need:

- **Recipes** defines reusable signals, projections, decisions, and algorithms;
- **Models** publishes an Entrypoint by choosing a Recipe and assigning configured
  physical Models to each stable decision.

Provider probes verify generation separately from Recipe evaluation. An
Entrypoint can route correctly even when a selected backend is unhealthy, so
publication validation and live probes report those states independently. The
Dashboard performs this lifecycle through the same Management API available to
automation and independent control planes; it has no setup-only config writer.

## Helm

For **standalone mode**, direct Helm deployments place a complete canonical
routing manifest under `configOverride`. This replaces the chart's example as one
document before the chart applies explicit Kubernetes integration rewrites; it
does not merge sample Models or decisions into your policy. Author and validate
the manifest as `config.yaml`, then place it under `configOverride` in the Helm
values file.

```yaml
configOverride:
  version: v0.4
  listeners:
    - name: grpc-50051
      address: 0.0.0.0
      port: 50051
      timeout: 300s
    - name: http-8080
      address: 0.0.0.0
      port: 8080
      timeout: 300s
  models:
    - name: local/general
      card:
        modality: text
        capabilities: [chat]
      connections:
        - provider: vllm
          endpoint: http://model-server.default.svc.cluster.local:8000/v1
          model: my-served-model
  recipes:
    - name: default
      document:
        strategy: priority
        decisions:
          - name: default-route
            description: Route requests to the configured model.
            priority: 1
            rules: {operator: AND, conditions: []}
  entrypoints:
    - name: vllm-sr/default
      recipe: default
      assignments:
        default-route:
          models: [{model: local/general}]
  global:
    services:
      backend_dispatch:
        bind_address: 0.0.0.0
        port: 8180
        audience: vllm-sr.backend-dispatch
        capability_ttl: 30s
        max_request_body_bytes: 67108864
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

`vllm-sr serve --target k8s` passes the workspace `config.yaml` document as an
atomic override, so chart example routes cannot merge into it. The command
rejects an empty or setup-only document and does not inject local-Docker service
addresses or knowledge-base paths. Run `vllm-sr validate` first so schema and
reference errors fail before deployment.

For **managed mode**, `configOverride` contains only Router bootstrap and service
configuration. Bind PostgreSQL and Valkey references to Kubernetes Secrets with
`extraEnv` entries. The chart runs one pre-install or pre-upgrade migration Job
from the Router image, then rolls out Router only after migration succeeds. It
also creates a dedicated `ClusterIP` Service for the backend-dispatch listener.
Point the gateway's one stable internal upstream at that Service; do not create a
route or cluster for each Model or API key.

```yaml
extraEnv:
  - name: ACCESS_DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: router-stores
        key: postgres-dsn
  - name: ACCESS_RUNTIME_URL
    valueFrom:
      secretKeyRef:
        name: router-stores
        key: valkey-url
```

The ConfigMap contains immutable Router bootstrap only. Dynamic routing and
access resources remain in Router-owned stores and never enter Helm values,
ConfigMaps, gateway resources, or CRDs. Dashboard and observability remain
independent opt-ins.

Choose Kubernetes GPU images, resources, and device plugins through Helm or the
Operator. The local `--platform amd` and `--platform nvidia` shortcuts do not
configure Kubernetes scheduling.

## Operator

The Operator may provide a Kubernetes-native authoring adapter for routing
resources:

- `spec.vllmEndpoints` discovers model services and creates Model backends; and
- `spec.config.routing` is an Operator authoring adapter compiled into Recipes
  and Entrypoints.

Other `spec.config` fields are typed Operator adapters for response cache,
classifiers, tools, observability, and related shared settings. In standalone
mode the Operator compiles one immutable manifest. In managed mode it reconciles
Models, Recipes, and Entrypoints through the ordinary versioned Management API;
it does not write a second mounted routing document.

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

Do not copy arbitrary `models` or `global` keys directly under `spec.config`;
they are not CRD fields. See [Kubernetes Operator](k8s/operator) and the
[SemanticRouter CRD reference](../api/semantic-router-crd).

## Routing DSL

The DSL Builder is a focused authoring surface for one model-free Recipe:
signals, projections, decisions, algorithms, and Recipe-scoped plugins. Models,
decision assignments, and Entrypoints remain Management API resources. Providers,
listeners, credentials, and global services are outside the Recipe DSL.

Use DSL when Recipe policy benefits from a compact, reviewable representation.
Publish the Recipe and connect it to an Entrypoint through the same Management API
used by the Dashboard.

## Avoid split ownership

- Do not edit a generated runtime config as if it were the source document.
- Do not let both GitOps and an interactive Dashboard session write the same
  deployment without an explicit handoff.
- Do not put secrets in DSL, ConfigMaps, or committed YAML.
- Do not assume an evaluated route proves backend readiness; verify generation.
- Preview and validate a complete change before applying it to a live stack.

For management endpoints and concurrency contracts, see the
[management API reference](../api/apiserver).
