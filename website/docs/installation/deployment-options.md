---
title: Choose a Deployment
description: Pick the simplest vLLM Semantic Router deployment path for local development, GPUs, Kubernetes, or an existing gateway.
---

# Choose a Deployment

Semantic Router and the inference backends are separate services. The Router
can run on CPU while the models run locally, on GPUs, in Kubernetes, or behind
a remote provider API.

Start with the smallest deployment that represents your production request
path. Add a gateway, operator, or external store only when you need what it
provides.

## At a glance

| Goal | Recommended path | Continue with |
| --- | --- | --- |
| Try the Router and Dashboard locally | CLI-managed Docker stack | [Quickstart](/docs/installation) |
| Use a local model without a GPU stack | Ollama plus the local Router | [Ollama](ollama) |
| Serve models on AMD Instinct GPUs | vLLM ROCm backend plus the Router | [AMD ROCm](amd-rocm) |
| Deploy from a complete config into Kubernetes | `vllm-sr serve --target k8s` and Helm | [Configuration Workflows](configuration-workflows#helm) |
| Manage Router resources as Kubernetes objects | Semantic Router Operator | [Kubernetes Operator](k8s/operator) |
| Attach routing to an existing gateway | Envoy AI Gateway, agentgateway, Istio, or Gateway API Inference Extension | [Kubernetes Gateways](k8s/gateways) |
| Integrate an inference platform | vLLM Production Stack, AIBrix, llm-d, or Dynamo | [Inference Platforms](k8s/inference-platforms) |

## Local Docker stack

`vllm-sr serve` starts the local Router, Envoy, Dashboard, and supporting
services. It is the fastest path for configuration work, evaluation, and a
single-host deployment.

The command does not normally provision the provider models referenced by a
custom config or built-in virtual model. Start those endpoints first, or bind
the config to endpoints that already exist.

Use the local stack when you want:

- interactive Dashboard setup;
- a repeatable development environment;
- local validation and recipe evaluation; or
- one host without Kubernetes lifecycle requirements.

## Kubernetes

There are two main Kubernetes paths:

- **CLI and Helm** translate a complete canonical config into a Helm release.
  This suits teams that already manage configuration and releases through
  command-line or GitOps workflows.
- **Operator** manages `SemanticRouter` resources and related lifecycle inside
  the cluster. This suits Kubernetes-native control planes and backend
  discovery.

Gateway and inference-platform guides are integrations, not alternative Router
policy models. They show where Semantic Router fits into an existing data plane
or model-serving stack.

## Model placement

Choose model placement independently from the Router deployment:

- **local or edge** for privacy, offline operation, or small pools;
- **GPU datacenter** for shared, high-throughput model services;
- **hybrid** when some requests must stay local and others may use a remote
  provider; or
- **hosted providers** when the deployment does not own model servers.

Whichever path you choose, verify that each backend supports the context,
modality, tool, and protocol requirements declared by its routes.

## Before production

Review these areas before exposing a deployment:

- [Configuration](configuration) for canonical YAML and environment bindings;
- [Security Hardening](security-hardening) for trust boundaries and credentials;
- [Data and Storage](storage-overview) for persistence and retention; and
- [Upgrade and Rollback](upgrade-rollback) for version pinning and recovery.
