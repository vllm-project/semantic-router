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
      store_backend: memory
```

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
      async_writes: true
```
