# API and Observability

## Overview

This page covers the shared runtime blocks that expose interfaces and telemetry.

These settings are router-wide and belong in `global:`, not in route-local plugin fragments.

## Key Advantages

- Keeps observability and interface controls consistent across routes.
- Avoids duplicating metrics or API settings inside route-local config.
- Makes replay and response APIs explicit shared services.
- Keeps operational controls in one router-wide layer.

## What Problem Does It Solve?

If API and telemetry behavior is configured per route, the operational surface becomes fragmented and hard to reason about.

This part of `global:` solves that by collecting shared interfaces and monitoring settings in one place.

## When to Use

Use these blocks when:

- the router should expose shared APIs
- the response API should be enabled for the whole router
- metrics and tracing should be configured once
- replay capture should be retained as a shared operational service

## Configuration

### API

```yaml
global:
  services:
    api:
      enabled: true
```

### Response API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis        # default; use "memory" only for local development
      redis:
        address: "redis:6379"
```

The `store_backend` field controls where response and conversation history is persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `redis` | Survives router restart, shared across replicas | Production (default) |
| `memory` | Lost on router restart | Local development only |

### Observability

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
```

### Skip Processing Header

`global.router.skip_processing.enabled` is the deployment-level gate that
opts the router into honoring the `x-vsr-skip-processing` request header.
When the gate is on and an upstream filter sets that header to `true`, the
router becomes a no-op for that single request — every Envoy ext_proc
callback returns CONTINUE without classifying, routing, mutating, caching,
or inspecting the request or upstream response. When the gate is off (the
default) the header is ignored entirely.

```yaml
global:
  router:
    skip_processing:
      enabled: false        # default; flip to true to honor the header
```

The Helm chart exposes the same gate as a top-level value
(`router.skipProcessing.enabled`) so it can be enabled at install time
without editing the embedded canonical config:

```bash
helm install vsr ./deploy/helm/semantic-router \
  --set router.skipProcessing.enabled=true
```

Enable this gate only when an authenticated upstream filter (Envoy AI
Gateway, ext_authz, route-level filters, etc.) is responsible for setting
or stripping the header on trust grounds. Background on the AI Gateway
interop pattern that motivates this gate lives in
[issue #1808](https://github.com/vllm-project/semantic-router/issues/1808).

### Router Replay

```yaml
global:
  services:
    router_replay:
      store_backend: postgres     # default; SQL-queryable audit storage
      enabled: true
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: router-secret
```

`global.services.router_replay.enabled` is the router-wide default. When it is on, a decision captures replay unless that decision adds a route-local `router_replay` plugin with `enabled: false`.

The `store_backend` field controls where routing-decision replay records are persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `postgres` | Full SQL queryability, long-term audit retention | Production (default) |
| `redis` | Survives router restart, shared across replicas | Lightweight deployments already running Redis |
| `milvus` | Vector-searchable replay records | Semantic replay search |
| `memory` | Lost on router restart | Local development only |
