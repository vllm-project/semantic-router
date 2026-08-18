---
title: Dashboard Integration
---

# Dashboard integration

The dashboard exposes Fleet Sim through its authenticated backend. Browser
requests go to the dashboard proxy, not directly to the simulator service.

## Local stack

From the repository root:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

Unless disabled, the CLI starts a Fleet Sim sidecar and supplies its service URL
to the dashboard backend. No simulator port needs to be published to the
browser.

## Proxy contract

The dashboard proxies Fleet Sim under:

```text
/api/fleet-sim/*
```

For example, the browser requests built-in workloads from
`/api/fleet-sim/api/workloads`. The backend removes the proxy prefix before
forwarding and supplies `X-Forwarded-Prefix` so Swagger and OpenAPI links work
through the proxy.

If no simulator URL is configured, the proxy returns a structured service
unavailable response. Check the Fleet Sim container or
`TARGET_FLEET_SIM_URL`; changing the router's inference listener does not affect
this connection.

## External service

Point the dashboard stack to an existing Fleet Sim service:

```bash
export TARGET_FLEET_SIM_URL=http://fleet-sim.internal:8000
vllm-sr serve --image-pull-policy never
```

The URL must be reachable from the dashboard backend, which may be a container
rather than the user's browser. Do not use `localhost` unless Fleet Sim runs in
that same network namespace.

## Dashboard pages

| Page | Purpose |
| --- | --- |
| **Overview** | Check service availability and review the workflow |
| **Workloads** | Inspect built-in CDFs or upload and preview a trace |
| **Fleets** | Save pool, GPU, context, and routing definitions |
| **Runs** | Submit optimize, simulate, or what-if jobs and inspect results |

The dashboard saves simulator objects through the service API. Treat those
objects as planning inputs, not as live router configuration; Fleet Sim does
not deploy or resize inference workers.

## Security boundary

Dashboard authentication applies to the proxy routes. The standalone Fleet Sim
service does not implement bearer authentication, so keep it on a trusted
network or place an authenticated proxy in front of it. Avoid uploading traces
that contain prompt text or identifiers unless the storage and access boundary
is appropriate for that data.
