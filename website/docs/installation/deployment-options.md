---
title: Choose a Deployment
description: Choose a Docker, Kubernetes, or hardware-specific path for deploying vLLM Semantic Router.
---

# Choose a Deployment

Semantic Router and its inference backends are separate services. Choose how to
run the Router first, then connect model endpoints that it can reach.

The deployment guides are organized into three paths: **Docker** for one host,
**Kubernetes** for cluster-managed deployments, and **Hardware** for preparing
GPU-backed model servers or accelerating Router-side models.

## At a glance

| Deployment | Choose it when | Start here |
| --- | --- | --- |
| Docker | You are evaluating the Router, developing locally, or deploying on one host. | [Deploy with Docker](docker) |
| Kubernetes | You need replicas, declarative rollouts, an operator, or an existing gateway or inference platform. | [Kubernetes](#kubernetes) |
| Hardware | You need to start a vLLM backend on AMD or NVIDIA GPUs, or accelerate supported Router-side models. | [Hardware](#hardware) |

## Docker

`vllm-sr serve` starts the local Router, Envoy, Dashboard, and supporting
services. It is the fastest path for configuration work, evaluation, and a
single-host deployment.

The command does not provision the provider Models used by a custom Entrypoint.
Start those endpoints first, or connect endpoints that already exist. Built-in
Recipes remain model-free until a control plane assigns Models and publishes an
Entrypoint.

Follow [Deploy with Docker](docker) for the stack lifecycle and backend
networking. If you want a small local model server, add
[Ollama](ollama). Both guides keep the Router and model-server responsibilities
separate.

## Kubernetes

Choose the Kubernetes path that matches who should own the Router lifecycle:

- Use the [CLI and Helm workflow](configuration-workflows#helm) when your team
  already owns a complete canonical config and deploys releases through CLI or
  GitOps automation.
- Use the [Kubernetes Operator](k8s/operator) when Kubernetes should manage
  `SemanticRouter` resources, discovery, and lifecycle.
- Choose a supported [gateway](k8s/gateways) when Semantic Router must join an
  existing traffic data plane.
- Choose an [inference-platform integration](k8s/inference-platforms) when
  another platform owns model deployment and replica scheduling.

Gateway and inference-platform integrations do not replace Router policy. They
connect semantic model selection to infrastructure that already owns traffic
or model-serving lifecycle.

## Hardware

The Hardware guides prepare GPU-backed vLLM endpoints and explain when to run
supported Router-side signal models on a GPU:

- [AMD ROCm](amd-rocm) for vLLM on AMD Instinct GPUs;
- [NVIDIA CUDA](nvidia-cuda) for vLLM on NVIDIA GPUs and optional CUDA
  acceleration in the Router.

Hardware is not a separate Router topology. You can connect these model servers
to either Docker or Kubernetes. Keep the Router on CPU unless measurements show
that its local embeddings or classifiers benefit from sharing GPU capacity.

Whichever hardware path you choose, test the model endpoint directly before
testing it through the Router. Confirm that it supports the context, modality,
tool, and protocol requirements declared by the recipe.

## Before production

Review these areas before exposing a deployment:

- [Configuration](configuration) for canonical YAML and environment bindings;
- [Security Hardening](security-hardening) for trust boundaries and credentials;
- [Data and Storage](storage-overview) for persistence and retention; and
- [Upgrade and Rollback](upgrade-rollback) for version pinning and recovery.
