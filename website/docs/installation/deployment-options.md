---
title: Choose a Deployment
description: Choose a Docker, Kubernetes, or hardware-specific path for deploying vLLM Semantic Router.
---

# Choose a Deployment

Choose two things independently:

1. where Semantic Router runs; and
2. where the model backends run.

The Router does not load or provision the model weights referenced by a custom
configuration. It sends requests to reachable model endpoints, which can run on
the same host, in a cluster, or behind a hosted API.

## Choose the Router topology

| Need | Recommended path | Start here |
| --- | --- | --- |
| Evaluate locally or run on one host | Docker stack managed by the CLI | [Deploy with Docker](docker) |
| Deploy a complete canonical config through GitOps | Helm | [CLI and Helm workflow](configuration-workflows#helm) |
| Let Kubernetes reconcile Router resources and discovery | Kubernetes Operator | [Kubernetes Operator](k8s/operator) |
| Attach routing policy to an existing gateway | Gateway integration | [Gateways](k8s/gateways) |
| Let another platform own model replicas and scheduling | Inference-platform integration | [Inference Platforms](k8s/inference-platforms) |

Gateway and inference-platform integrations do not replace Router policy. They
connect semantic model selection to infrastructure that owns traffic or model
lifecycle.

Before committing to a path, check its project-maintained status, recurring
test evidence, and external ownership boundary in the
[Deployment Support](support-matrix).

## Choose the model backend

| Backend situation | Start here |
| --- | --- |
| You already have a reachable model or provider endpoint | [Protocol Compatibility](protocol-compatibility), then [Backend Target Compatibility](backend-target-compatibility) |
| You want a small local model for evaluation | [Local model with Ollama](ollama) |
| You want to serve models on AMD Instinct | [AMD ROCm](amd-rocm) |
| You want to serve models or accelerate Router-side models on NVIDIA | [NVIDIA CUDA](nvidia-cuda) |
| A Kubernetes platform owns model deployment and replicas | [Inference Platforms](k8s/inference-platforms) |

Hardware is an overlay, not a separate Router topology. A GPU-backed model
server can connect to either a Docker or Kubernetes Router deployment. Keep the
Router on CPU unless measurements show that its local embeddings or classifiers
benefit from GPU acceleration.

Test a model endpoint directly before sending traffic through the Router. A
configured URL is not proof that the backend implements the selected wire
protocol or supports the recipe's context, modality, and tool requirements.

## Before production

Before exposing a deployment:

1. pin the Router, model server, model, and integration versions together;
2. validate buffered, streaming, failure, and rollback behavior through the
   actual data plane;
3. move credentials into a secret manager; and
4. review [Configuration](configuration),
   [Security Hardening](security-hardening),
   [Data and Storage](storage-overview), and
   [Upgrade and Rollback](upgrade-rollback).
