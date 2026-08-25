---
title: Configuration Workflows
description: Keep the v0.3 bootstrap and dynamic Router resources under one clear authority.
---

# Configuration Workflows

Semantic Router always starts from one readable `version: v0.3` manifest. The
components configured in that manifest determine which capabilities are active;
there is no separate deployment-mode switch.

- Without `global.stores.management`, the file is the immutable routing authority.
- With a Management store, Router initializes an empty store atomically from the
  file. PostgreSQL is the sole desired-state authority after that transaction.
- Enabling the Management API exposes versioned CRUD, import, and query operations.
- Enabling Router-native access also requires the runtime store for global API-key
  authentication, authorization, quota, settlement, usage, and audit state.

A later file edit never merges into initialized Management state. Apply a reviewed
manifest deliberately through `POST /management/v1/routing/imports`, with its
idempotency key and expected revision.

## CLI and bootstrap YAML

Use YAML for physical connections, an initial routing graph, listeners, and
infrastructure settings. Start the local stack with one command:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve
# Or select another immutable bootstrap manifest.
vllm-sr serve --config /path/to/config.yaml
```

`--config` chooses one v0.3 bootstrap manifest. It is not a Model or Recipe
operand. The CLI has no second launch path that authors a Mixture-of-Models.
Runtime-owned addresses and generated identities stay outside the source file.

## Dashboard and other control planes

The Dashboard is an optional Management API client. Connect physical backends in
**Models**. Fixed-origin Providers ask for a credential and provider model; private
Providers also ask for a base URL. Discovery can import one or many provider models.

The **Mixture-of-Models** workspace keeps two concepts separate:

- **Recipes** owns reusable signals, projections, decisions, algorithms, and plugins.
- **Models** publishes a request-facing Entrypoint by choosing a Recipe and assigning
  connected Models to each Decision.

The Dashboard never owns inference authentication, policy evaluation, rate limiting,
usage settlement, or routing publication. An independent console can implement the
same workflows through `/management/v1` and send inference directly to `/v1`.

## Helm

Place one complete v0.3 document under `configOverride`. It replaces the chart
example atomically; Helm does not merge sample Models or Decisions into it.

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
    models:
      - name: local/general
        provider_model_id: my-served-model
        backend_refs:
          - provider: vllm
            base_url: http://model-server.default.svc.cluster.local:8000/v1
  routing:
    modelCards:
      - name: local/general
        modality: text
        capabilities: [chat]
  recipes:
    - name: default
      routing:
        strategy: priority
        decisions:
          - name: default-route
            description: Route requests to the configured model.
            priority: 1
            rules: {operator: AND, conditions: []}
  entrypoints:
    - model_names: [vllm-sr/default, default]
      recipe: default
      assignments:
        default-route:
          models: [{model: local/general}]
```

Add dynamic capabilities without changing this routing vocabulary:

```yaml
configOverride:
  version: v0.3
  global:
    stores:
      management:
        postgres:
          dsn_env: VLLM_SR_POSTGRES_DSN
      runtime:
        redis:
          url_env: VLLM_SR_REDIS_URL
    services:
      agent:
        public_inference_endpoint: https://inference.example.com/v1/chat/completions
      backend_credentials:
        provider_kek_keyring_env: VLLM_SR_PROVIDER_CREDENTIAL_KEKS
      backend_egress:
        policy_file: /etc/vllm-sr/backend-egress-policy.yaml
      routing_security:
        hmac_keyring_env: VLLM_SR_ROUTING_HMAC_KEYS
      management_api:
        enabled: true
        tls:
          certificate_env: VLLM_SR_MANAGEMENT_TLS_CERTIFICATE
          private_key_env: VLLM_SR_MANAGEMENT_TLS_PRIVATE_KEY
        auth:
          mode: router
          token_signing_keyring_env: VLLM_SR_MANAGEMENT_SIGNING_KEYS
          service_account_hmac_keyring_env: VLLM_SR_MANAGEMENT_SERVICE_ACCOUNT_KEYS
          invitation_hmac_keyring_env: VLLM_SR_INVITATION_KEYS
          response_kek_keyring_env: VLLM_SR_MANAGEMENT_RESPONSE_KEKS
      access:
        enabled: true
        credentials:
          api_key_hmac_keyring_env: VLLM_SR_API_KEY_HMAC_KEYS
          delegation_hmac_keyring_env: VLLM_SR_DELEGATION_HMAC_KEYS
        tenant_context:
          signing_key_env: VLLM_SR_TENANT_CONTEXT_KEYS
```

Bind DSNs and keyrings through Kubernetes Secrets and environment references.
The chart runs the release's migration Job before new Router replicas become ready.
Dynamic resources never enter Helm values, ConfigMaps, gateway routes, or per-key
Kubernetes objects.

```bash
vllm-sr validate --config config.yaml

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  -f values.yaml
```

`vllm-sr serve --target k8s` passes the workspace document as the same atomic
override. Run validation first so schema and reference errors fail before rollout.

## Operator

The Operator owns Kubernetes objects and rollout state. A file-authoritative
deployment rolls when its immutable ConfigMap reference changes. When a Management
store is configured, the Operator still reconciles only deployment and bootstrap
concerns. Models, Recipes, and Entrypoints are changed through the ordinary
Management API; the Operator does not maintain a second routing authority.

Typed Operator adapters such as `spec.vllmEndpoints` may discover model Services and
render the same public Model contract. They must not place generated resource IDs,
credentials, users, API keys, policies, or counters in the CRD.

See [Kubernetes Operator](k8s/operator) and the
[SemanticRouter CRD reference](../api/semantic-router-crd).

## Routing DSL

The DSL Builder authors one model-free Recipe: signals, projections, decisions,
algorithms, and Recipe-scoped plugins. Models, Decision assignments, and Entrypoints
remain Management resources. Providers, listeners, credentials, stores, and global
services are outside the Recipe DSL.

The Playground Builder uses the Router's schema and catalog tools, probes candidate
routes, and asks for confirmation before publishing through the same Management API.

## Avoid split ownership

- Do not edit generated runtime state as if it were the source manifest.
- Do not change an initialized Management deployment by replacing its bootstrap file;
  use the explicit import or resource APIs.
- Do not let GitOps and an interactive client write the same resource revision without
  optimistic concurrency and an explicit handoff.
- Do not put secrets in DSL, ConfigMaps, committed YAML, or Dashboard local storage.
- Do not assume a valid Recipe proves backend readiness; run generation probes.
- Preview and validate a complete change before applying it to live traffic.

For endpoint and concurrency contracts, see the
[Management API reference](../api/apiserver).
