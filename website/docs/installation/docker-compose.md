---
sidebar_position: 3
---

# Install with Docker Compose

:::warning Deprecated
The bundled `deploy/docker-compose/docker-compose.yml` has been removed, and Docker
Compose is no longer the supported way to run vLLM Semantic Router locally.

For local development with Docker, use the `vllm-sr` CLI instead — `vllm-sr serve`
provisions the router, Envoy, and dashboard for you. See the
**[Quickstart](./installation.md)** for the current workflow.
:::

## Why this changed

Earlier releases shipped a `docker-compose.yml` under `deploy/docker-compose/`.
Local Docker orchestration is now handled by the `vllm-sr` CLI, which keeps the
container wiring, configuration bootstrap, and dashboard aligned with each release.

## Run locally with the CLI

```bash
# Install the CLI (see the Quickstart for prerequisites)
pip install --pre vllm-sr

# Start the router, Envoy, and dashboard in Docker
vllm-sr serve
```

The **[Quickstart](./installation.md)** covers prerequisites, model downloads, and the
full set of `vllm-sr` commands (`vllm-sr status`, `vllm-sr logs`, `vllm-sr stop`).

## Kubernetes and production

For cluster deployments, use the **[Operator](k8s/operator)** or the Helm charts
under `deploy/helm/`.
