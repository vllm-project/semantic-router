---
title: Docker
description: Run Semantic Router as a local or single-host container stack and connect model backends that you operate separately.
---

# Deploy with Docker

Docker is the shortest path from a Semantic Router configuration to a running
stack. It is a good fit for evaluation, development, CI, edge hosts, and
single-host deployments that do not need Kubernetes scheduling or failover.

The CLI manages Router, Envoy, Dashboard, and the supporting services required
by the selected configuration. Model servers remain separate: a healthy Router
stack does not mean that its provider endpoints are installed, running, or
able to generate.

## Start the stack

Complete the [Quickstart](/docs/installation) to install the CLI and create a
configuration, or start from an existing canonical YAML file:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

With no `--config`, `vllm-sr serve` uses `config.yaml` in the current directory
or opens first-run setup in the Dashboard. The default local endpoints are:

| Endpoint | Default | Purpose |
| --- | --- | --- |
| Dashboard | `http://localhost:8700` | Configure and inspect the stack. |
| Routed listener | `http://localhost:8899` | Send OpenAI-compatible model requests. |
| Management API | `http://localhost:8080` | Validate config and use evaluation, replay, or vector-store APIs. |

Ports can change with the active configuration or a stack port offset. Use
`vllm-sr status` when you are unsure which endpoints are active.

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

Use `--minimal` to run only Router and Envoy. Use `--readonly` to keep the
Dashboard available without allowing configuration changes. Pin images and
review [Security Hardening](security-hardening) before exposing a listener
beyond a trusted host.

## When to move to Kubernetes

Docker does not provide multi-node scheduling, rolling deployment control, or
cluster-level recovery. Move to a Kubernetes path when you need replicas,
declarative rollout, gateway integration, or platform-managed model discovery.
The same canonical configuration can be deployed with the CLI and Helm or
managed through the Semantic Router Operator.
