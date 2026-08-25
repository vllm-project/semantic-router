---
title: Docker
description: Run Semantic Router as a local or single-host container stack and connect model backends that you operate separately.
---

# Deploy with Docker

Docker is the shortest path from a Semantic Router configuration to a running
stack. It is a good fit for evaluation, development, CI, edge hosts, and
single-host deployments that do not need Kubernetes scheduling or failover.

The CLI manages Router, Envoy, and only the services selected by the
configuration. Model servers remain separate: a healthy Router stack does
not mean that its provider endpoints are installed, running, or able to
generate.

## Start the stack

Complete the [Quickstart](/docs/installation) to install the CLI, then start the
stack:

```bash
vllm-sr serve
```

`vllm-sr serve` uses `config.yaml` in the current directory as its static
bootstrap. When the Management API is configured, the Dashboard can create and
publish Models, Recipes, assignments, and Entrypoints without rewriting that
file.

Select another v0.3 bootstrap without copying it into the workspace:

```bash
vllm-sr serve --config /path/to/config.yaml
```

The flag selects deployment bootstrap only; it does not select a Model or Recipe.

| Endpoint | Default | Purpose |
| --- | --- | --- |
| Dashboard | `http://localhost:8700` | Configure and inspect the stack. |
| Routed listener | `http://localhost:8899` | Send OpenAI-compatible model requests. |
| Management API | Private stack network | Manage routing, identity, keys, policy, usage, and operations when configured. |

Ports can change with the active configuration or a stack port offset. Use
`vllm-sr status` when you are unsure which endpoints are active.

## Add only the capabilities you need

There is no deployment-mode switch. The same v0.3 file works in Docker and
Kubernetes, and typed service and store blocks determine what starts:

| Configuration | Local services | Routing and access |
| --- | --- | --- |
| No Management store | Router and Envoy | The validated file is the routing authority. |
| `global.stores.management.postgres` | Router, Envoy, and PostgreSQL | An empty database is seeded from the file; PostgreSQL then owns durable Models, Recipes, Entrypoints, and publication state. |
| Management store plus `global.services.management_api.enabled` | Durable stack plus the private Management listener | Versioned routing and identity operations are available to Dashboard or another authorized client. |
| Management plus `global.stores.runtime.redis` and `global.services.access.enabled` | Router, Envoy, PostgreSQL, and Valkey | Router-native API keys, model grants, global quotas, usage, and audit are active on every inference path. |

Store connection values are environment or file references, never literal URLs
in YAML. When a reference is not populated, the single-host profile starts the
corresponding PostgreSQL or Valkey container with a named volume and supplies a
private connection value. A populated environment reference selects an
external store. A file reference must be an absolute local secret file; the CLI
mounts it read-only at the same path.

Before Router starts, the CLI runs `management-migrate` once from the same Router
image. Migration failure prevents Router startup. Router replicas never run
migrations as a startup side effect.

Users, Teams, API keys, policies, usage records, and audit events never belong
in YAML. A routing bootstrap may contain Models, Recipes, and Entrypoints. On an
empty Management store they are imported atomically once; after that,
PostgreSQL is the only mutable routing authority. When the Management API is
enabled, import a later file through its versioned endpoint so validation,
audit, and publication stay atomic.

See [API Keys, Access, and Usage](../tutorials/global/access-and-usage) for the
Team, policy, Budget, key, and live-quota workflow.

## Connect model backends

Choose the connection that matches where the model server runs:

| Model location | Configure the backend with |
| --- | --- |
| On the Docker host | `host.docker.internal:<port>`; the CLI adds the host-gateway mapping. |
| In a container on the same network | The model container's DNS name and service port. |
| On another host or managed service | Its reachable HTTPS base URL and environment-backed credentials. |

For a small local model, follow [Configure models with Ollama](ollama). For a
GPU-backed vLLM server, choose a guide under **Hardware**. In every case,
verify the model endpoint directly before debugging routing.

## Operate a local stack

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

Use `--minimal` to run only Router and Envoy, without Dashboard or the optional
observability stack. Prometheus, Grafana, and Jaeger are not ordinary `serve`
dependencies; add `--with-observability` to start them. Use `--readonly` to keep the Dashboard available without
allowing configuration changes. Neither Dashboard nor observability is needed
for request routing or access enforcement. Pin images and review
[Security Hardening](security-hardening) before exposing a listener beyond a
trusted host.

## When to move to Kubernetes

Docker does not provide multi-node scheduling, rolling deployment control, or
cluster-level recovery. Move to a Kubernetes path when you need replicas,
declarative rollout, gateway integration, or platform-managed model discovery.
The same canonical configuration can be deployed with the CLI and Helm or
managed through the Semantic Router Operator.
