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

### Router Replay

```yaml
global:
  services:
    router_replay:
      store_backend: postgres     # default; SQL-queryable audit storage
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: router-secret
```

The `store_backend` field controls where routing-decision replay records are persisted. Available backends:

| Backend | Durability | Use case |
|---------|-----------|----------|
| `postgres` | Full SQL queryability, long-term audit retention | Production (default) |
| `redis` | Survives router restart, shared across replicas | Lightweight deployments already running Redis |
| `milvus` | Vector-searchable replay records | Semantic replay search |
| `memory` | Lost on router restart | Local development only |
