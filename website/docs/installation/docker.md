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
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

A stable CLI pulls Router and Dashboard images tagged with its own release
version. Development CLI builds use `:latest`; `--image` and the documented
image environment overrides select a different build when needed.

With no `--config`, `vllm-sr serve` uses `config.yaml` in the current directory
or opens first-run setup in the Dashboard. The default local endpoints are:

| Endpoint | Default | Purpose |
| --- | --- | --- |
| Dashboard | `http://localhost:8700` | Configure and inspect the stack. |
| Routed listener | `http://localhost:8899` | Send OpenAI-compatible model requests. |
| Management API | `http://localhost:8080` | Validate config and use evaluation, replay, or vector-store APIs. |

The Dashboard port binds to `127.0.0.1` by default. First-run admin
registration remains available from the local host. For a remote machine,
forward the port with `ssh -L 8700:127.0.0.1:8700 <host>` and open the local
Dashboard URL. To publish the Dashboard on all host interfaces, set
`VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0` and provision both
`DASHBOARD_ADMIN_EMAIL` and `DASHBOARD_ADMIN_PASSWORD`. An explicit
`DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true` also permits a wildcard bind when
first-admin registration on that network is intended. The CLI rejects a
wildcard bind with neither choice configured.

Ports can change with the active configuration or a stack port offset. Use
`vllm-sr status` when you are unsure which endpoints are active.

After starting containers, the CLI waits up to 1800 seconds for Router
readiness, or Dashboard readiness during first-run setup. Set
`--startup-timeout SECONDS` to a positive integer when model loading or GPU
compilation needs a different budget:

```bash
vllm-sr serve --config config.yaml --startup-timeout 7200
```

Choose the budget using startup measurements for your models and hardware.
This option applies to the local Docker target and does not change inference
request deadlines. On timeout, the CLI exits with an error and leaves the
containers available for `vllm-sr status` and `vllm-sr logs router`.
Use `vllm-sr stop` when you want to stop them.

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

Envoy uses the safe `info` log level by default. To troubleshoot temporarily,
set `VLLM_SR_ENVOY_LOG_LEVEL=debug` before starting the stack, then unset it
when finished: debug logging can expose forwarded request headers, including
provider `Authorization` headers, in logs.

## Expose the Dashboard through a reverse proxy

Grafana Live validates the `Origin` header used to establish its WebSocket.
When an HTTPS reverse proxy exposes the Dashboard on a public hostname, allow
that exact origin before starting the stack:

```bash
export GF_LIVE_ALLOWED_ORIGINS='https://dashboard.example.com'
vllm-sr serve --config config.yaml
```

Separate multiple origins with commas. Use origins in
`scheme://host[:port]` form without paths, queries, fragments, or trailing
slashes. Grafana supports wildcard patterns, but exact origins are safer for
public deployments. `vllm-sr serve` writes the value into the generated
`.vllm-sr/grafana/grafana.serve.ini`; restart the stack after changing it.

## When to move to Kubernetes

Docker does not provide multi-node scheduling, rolling deployment control, or
cluster-level recovery. Move to a Kubernetes path when you need replicas,
declarative rollout, gateway integration, or platform-managed model discovery.
The same canonical configuration can be deployed with the CLI and Helm or
managed through the Semantic Router Operator.
