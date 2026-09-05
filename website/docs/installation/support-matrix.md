---
title: Deployment Support
description: See which deployment paths, integrations, examples, and hardware profiles the Semantic Router project maintains.
---

# Deployment Support

Use this page to check which Router deployment paths and integrations the
project maintains. It does not certify every platform, model server, model, or
accelerator combination.

If you are still choosing a topology, start with
[Choose a Deployment](deployment-options). For wire formats and endpoint
configuration, use [Protocol Compatibility](protocol-compatibility) and
[Backend Target Compatibility](backend-target-compatibility).

## What each status means

| Status | Meaning |
| --- | --- |
| Maintained reference stack | The project owns and tests the installation or lifecycle contract. |
| Supported integration | The project tests the Router-side connection; the external platform owns its lifecycle. |
| Experimental example | The files demonstrate a feature or test topology that you must qualify. |
| Deprecated | The option remains temporarily for a documented migration. |

Evidence tags show the strongest recurring check: **PR CI** runs an end-to-end
profile, **Contract** validates static contracts without the external platform,
and **Manual** requires an opt-in environment.

## Use one tested version set

For each linked option:

1. use an external platform version named in its guide; if none is named,
   treat your choice as unqualified until you test it;
2. take every Semantic Router artifact you use—the CLI, chart, CRDs,
   controller, and images—from one release; and
3. test that exact set in your environment before upgrading any one component.

Support covers that tested version set, not every version supported by the
external project.

## Maintained reference stacks

| Option | Classification | Project coverage |
| --- | --- | --- |
| [Helm chart](configuration-workflows#helm) | Maintained reference stack | **Contract.** Router, optional Dashboard, ingress, autoscaling, persistence, and observability resources. Gateways and storage remain external. |
| [Local deployment](docker) | Maintained reference stack | **PR CI.** The CLI manages Router, Envoy, Dashboard, and support services. You provide custom model endpoints and harden local defaults. |
| [Kubernetes Operator](k8s/operator) | Maintained reference stack | **PR CI + Contract.** The project owns CRDs, reconciliation, Router workloads, Services, and routing APIs. Kubernetes schedules workloads; your gateway carries traffic. |

## Supported integrations

| Integration | Classification | Project coverage |
| --- | --- | --- |
| [agentgateway](k8s/agentgateway) | Supported integration | **PR CI.** Router supplies ExtProc policy; agentgateway owns the data plane. Set request bodies to `FullDuplexStreamed`. |
| [Envoy AI Gateway](k8s/ai-gateway) | Supported integration | **PR CI.** Router supplies routing policy; the gateways own provider traffic. Verify your provider separately. |
| [AIBrix](k8s/aibrix) | Supported integration | **PR CI.** Router selects a model or pool; AIBrix owns deployment, autoscaling, and replicas. Use AIBrix's hardware support matrix. |
| [NVIDIA Dynamo](k8s/dynamo) | Supported integration | **Manual.** Router selects a target; Dynamo owns graphs, workers, and frontends. Use the guide version; the older fixture is test-only. |
| [Istio Gateway](k8s/istio) | Supported integration | **PR CI.** Router supplies ExtProc policy; Istio carries requests. Supplied GPU workloads are examples only. |
| [llm-d](k8s/llm-d) | Supported integration | **PR CI + Contract.** Router selects a model or pool; llm-d owns discovery and replica routing. Do not add a competing direct-Service route. |
| [Streaming with Envoy AI Gateway](k8s/streamed-extproc) | Supported integration | **PR CI.** The gateway streams transport; Router uses the configured ExtProc body mode. Test that mode explicitly. |
| [Valkey agentic memory](valkey-memory) | Supported integration | **Contract + Manual.** Router owns memory behavior; Valkey owns persistence and Search. You own security, retention, and backup. |
| [Responses API state with Redis](../tutorials/global/api-and-observability#response-api) | Supported integration | **Manual.** Router owns Responses behavior; Redis stores state. You own Redis security, persistence, and eviction. |
| [Response cache](../tutorials/plugin/response-cache) | Supported integration | **Contract + Manual.** Router owns cache behavior; your backend owns storage and availability. Treat cached data as sensitive. |
| [Valkey vector store](storage-overview) | Supported integration | **Manual.** Router owns store references; Valkey owns indexes and durability. Pin Valkey, Search, and the embedding model together. |

## Experimental examples

| Example | Classification | Use it for / not for |
| --- | --- | --- |
| KServe example | Experimental example | KServe integration smoke testing; not a qualified KServe or model-serving deployment. |
| OpenShift example | Experimental example | Adapting resources to Routes and security constraints; not a hardened OpenShift profile. |
| Anthropic-compatible backend fixture | Experimental example | Protocol integration tests; not a production model service. |
| Hallucination policy demo | Experimental example | Fact-check policy behavior; not a qualified guardrail or model. |
| Jailbreak error-handling demo | Experimental example | Classifier failure paths; not a secure production policy. |
| LLM Katan development backends | Experimental example | Lightweight OpenAI-compatible test backends; not production inference. |
| Observability demo | Experimental example | Prometheus, Grafana, alert, and Dashboard wiring; replace all example security and retention settings. |
| Response jailbreak demo | Experimental example | Response-classifier window behavior; not a production guardrail model. |
| Responses API Kubernetes demo | Experimental example | Redis persistence and restart behavior; not a hardened Redis deployment. |
| Route action demo | Experimental example | Focused route-action behavior; compose it into a maintained stack before production. |
| Router replay recovery demo | Experimental example | Replay and restart recovery; not a supported deployment platform. |
| Routing strategy demos | Experimental example | Focused policy examples; qualify routing quality and backend capacity with representative traffic. |
| Local tools database | Experimental example | Local tool definitions for examples and tests; use an authenticated, durable registry in production. |

No shipped option is currently **Deprecated**. See
[Upgrade and Rollback](upgrade-rollback) for migration guidance.

## Hardware overlays

Hardware support applies to a deployment stack; it is not a separate Router
topology. Router acceleration and backend model serving are also separate
choices.

| Hardware profile | Status | What is covered |
| --- | --- | --- |
| Linux x86-64 CPU | Maintained | The standard Router images, CLI, Helm chart, and Operator receive the broadest recurring coverage. Model-server requirements remain separate. |
| Linux Arm64 CPU | Build-qualified | Release workflows publish multi-architecture images where declared. This does not qualify every integration or optional native dependency on Arm64. |
| NVIDIA CUDA on Linux x86-64 | Supported integration | Follow [NVIDIA CUDA](nvidia-cuda) for supported Router-side models, or keep the Router on CPU and qualify a separate NVIDIA backend against vLLM's support matrix. |
| AMD ROCm on Linux x86-64 | Supported integration | Follow [AMD ROCm](amd-rocm) for supported Router-side models and qualify the Router, vLLM, ROCm, and model revisions as one set. |
| AMD AI PC/NPU | Experimental, not qualified | No maintained deployment contract yet; tracked by [issue #2373](https://github.com/vllm-project/semantic-router/issues/2373). |
| NVIDIA DGX Spark Arm64 | Experimental, not qualified | Arm64 images do not qualify CUDA or end-to-end inference on this platform; tracked by [issue #2374](https://github.com/vllm-project/semantic-router/issues/2374). |
| Other accelerators and operating systems | Not qualified | No maintained profile. Open a qualification issue with reproducible hardware, software, image, and test evidence. |

## Before production

Test the model endpoint directly, then exercise buffered, streaming, failure,
upgrade, and rollback paths through the real data plane. Review
[Security Hardening](security-hardening), [Data and Storage](storage-overview),
and [Upgrade and Rollback](upgrade-rollback) for controls outside this matrix.
