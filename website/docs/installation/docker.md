---
title: Docker
description: Run Semantic Router as a local or single-host container stack and connect model backends that you operate separately.
---

# Deploy with Docker

Docker is the shortest path from a Semantic Router configuration to a running
stack. It is a good fit for evaluation, development, CI, edge hosts, and
single-host deployments that do not need Kubernetes scheduling or failover.

The CLI manages Router, Envoy, and only the services required by the selected
control-plane mode. Model servers remain separate: a healthy Router stack does
not mean that its provider endpoints are installed, running, or able to
generate.

## Start the stack

Complete the [Quickstart](/docs/installation) to install the CLI, then start the
stack:

```bash
vllm-sr serve
```

`vllm-sr serve` uses `config.yaml` in the current directory as deployment
bootstrap or opens first-run setup in the Dashboard. Models, Recipes, decision
assignments, and Entrypoints are managed after startup. The default local
endpoints are:

Select another immutable v0.4 bootstrap without copying it into the workspace:

```bash
vllm-sr serve --config /path/to/config.yaml
```

The flag selects deployment bootstrap only; it does not select a Model or Recipe.

| Endpoint | Default | Purpose |
| --- | --- | --- |
| Dashboard | `http://localhost:8700` | Configure and inspect the stack. |
| Routed listener | `http://localhost:8899` | Send OpenAI-compatible model requests. |
| Management API | Private stack network | Manage routing, identity, keys, policy, usage, and operations. Managed mode terminates TLS and does not publish this listener as a public endpoint. |

Ports can change with the active configuration or a stack port offset. Use
`vllm-sr status` when you are unsure which endpoints are active.

## Choose the control-plane mode

The same `global.control_plane.mode` contract is used by Docker and
Kubernetes:

| Mode | Local services | Routing state |
| --- | --- | --- |
| `standalone` | Router and Envoy; no PostgreSQL or Valkey | One immutable routing manifest; no managed access control |
| `managed` routing-only | Router, Envoy, PostgreSQL, and Valkey | Models, Recipes, and Entrypoints are managed through the API |
| `managed` access | Router, Envoy, PostgreSQL, and Valkey | Managed routing plus identities, keys, grants, quotas, usage, and audit |

Standalone uses in-memory service defaults and does not start a state store.
Managed mode requires `global.stores.access.type: postgres` and
`global.stores.access_runtime.type: redis`. Their connection values are always
environment or file references, never literal URLs in YAML.

When a managed environment reference is not populated, the single-host Docker
profile starts PostgreSQL and Valkey with named volumes and binds the reference
inside the private stack network. A populated reference selects an external
store instead. A file reference must be an absolute local secret file; the CLI
mounts it read-only at the same path.

Before Router starts, the CLI runs `access-migrate` once from the same Router
image. Migration failure prevents Router startup. Router replicas never run
migrations as a startup side effect.

Managed bootstrap YAML contains no Models, Recipes, Entrypoints, identities,
API keys, policies, usage records, or audit events. Create those resources
through the Management API or Dashboard so their changes do not rewrite YAML,
Envoy routes, or containers.

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
