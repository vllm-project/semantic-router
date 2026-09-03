---
title: Deployment and Hardware Support Matrix
description: Understand which Semantic Router deployment options and hardware profiles are maintained, supported, experimental, or deprecated.
---

# Deployment and Hardware Support Matrix

Use this page to choose a deployment option and understand what the Semantic
Router project maintains. A documented option does not, by itself, make every
platform version, image, model, or accelerator combination supported.

For a task-oriented starting point, see [Choose a Deployment](deployment-options).

## Support levels

| Level | Project commitment |
| --- | --- |
| Maintained reference stack | The project owns the installation or lifecycle contract and protects it with repository tests. Pin a Semantic Router release for production. |
| Supported integration | The project maintains the Router-side attachment contract and documentation. The external platform keeps its own support matrix and lifecycle. |
| Experimental example | The option demonstrates a feature or test topology. Review and adapt it; compatibility and production hardening are not guaranteed. |
| Deprecated | The option remains temporarily for migration and has a documented replacement or removal path. |

Evidence labels describe the project's strongest recurring check. They do not
prove that an arbitrary downstream environment passed:

- **PR/full CI** means a registered end-to-end profile is selected in pull
  request or full-CI coverage.
- **Contract/build** means source, schema, rendering, image, or packaging
  checks protect the option without running every external platform.
- **Manual profile** means the project provides an opt-in test with external
  prerequisites. It does not mean the profile passed in another environment.
- **Documented example** means the project checks the files and documentation,
  but operators own end-to-end qualification.

## Maintained reference stacks

| Option | Classification | What the project maintains | Version, ownership, and evidence boundary |
| --- | --- | --- | --- |
| [Helm chart](configuration-workflows#helm) | Maintained reference stack | Router, optional Dashboard, ingress, autoscaling, persistence, and observability resources. | Use the chart from the same Semantic Router release and pin application images. The project owns the chart and Router lifecycle; the selected Gateway, Service, and storage remain separate. The [Helm safety gate](https://github.com/vllm-project/semantic-router/blob/main/tools/make/helm.mk) validates rendering, schema, and safety rules. |
| [Local deployment](docker) | Maintained reference stack | Router, Envoy, Dashboard, and selected support services through `vllm-sr serve`. | Use same-release CLI and images; pin image digests for controlled deployments. The CLI owns the local stack lifecycle, while model endpoints remain separate. The [CLI integration workflow](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/integration-test-vllm-sr-cli.yml) exercises the lifecycle with real containers. Local defaults require hardening before production. |
| [Kubernetes Operator](k8s/operator) | Maintained reference stack | Reconciliation of Router configuration, workload, Service, storage, autoscaling, and the project-owned `vllm.ai` routing APIs used by controllers and integrations. | Install the CRDs, controller, and Router images from one release. Kubernetes owns workload scheduling and the chosen gateway owns traffic. [Operator CI](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/operator-ci.yml) covers unit, generated-resource, image, API-schema, and cluster-reconciliation contracts. The routing APIs are components of these integrations, not a standalone deployment. |

## Supported integrations

Each linked guide is authoritative for its current external platform versions.
Pin that compatibility set together with the selected Semantic Router release.

| Integration | Classification | Ownership and version boundary | Evidence and limits |
| --- | --- | --- | --- |
| [agentgateway](k8s/agentgateway) | Supported integration | agentgateway owns the Gateway API data plane; Semantic Router supplies ExtProc policy. Use the versions pinned by the guide. | PR/full CI profile. agentgateway requires `FullDuplexStreamed` rather than `Streamed` request bodies. |
| [Envoy AI Gateway](k8s/ai-gateway) | Supported integration | Envoy AI Gateway and Envoy Gateway own provider traffic; Semantic Router supplies routing policy. Use the versions pinned by the guide. | Default and full-CI profile. Provider compatibility changes independently; verify the selected provider. |
| [AIBrix](k8s/aibrix) | Supported integration | AIBrix owns model deployment, autoscaling, and replica routing; Semantic Router selects the model or pool. | PR/full CI profile. Pin one AIBrix release and do not treat this integration as a replacement for AIBrix's cluster and accelerator support matrix. |
| [NVIDIA Dynamo](k8s/dynamo) | Supported integration | Dynamo owns graphs, workers, and frontend lifecycle; Semantic Router selects a model target. Use the versions pinned by the guide. | Manual profile because it requires a prepared Dynamo/GPU environment. The repository also retains a pinned Dynamo `0.6.1.post1` model-deployment fixture for historical testing; it is not compatible with the guide's current platform and is not a production deployment. |
| [Istio Gateway](k8s/istio) | Supported integration | Istio's Gateway API data plane carries requests; Semantic Router is the ExtProc policy service. Use the versions pinned by the guide. | PR/full CI profile. Supplied model workloads need NVIDIA GPU capacity and are examples, not a general GPU guarantee. |
| [llm-d](k8s/llm-d) | Supported integration | Semantic Router selects a logical model or pool; the Gateway route targets its `InferencePool`, and llm-d owns discovery, endpoint picking, and replica routing. The repository supplies both Router values and scheduler-manifest examples. | PR/full CI profile plus contract/build coverage. Match the APIs and images to one llm-d release, and do not apply competing direct-Service routes for the same traffic. |
| [Streaming with Envoy AI Gateway](k8s/streamed-extproc) | Supported integration | Envoy AI Gateway owns streamed transport; Semantic Router processes the configured ExtProc body mode. | PR/full CI streaming profile. Pin both releases and test the body mode because streaming capabilities differ across gateways. |
| [Valkey agentic memory](valkey-memory) | Supported integration | Semantic Router owns memory configuration; Valkey owns persistence, availability, and Search-module compatibility. | Contract and manual integration coverage. Credentials, retention, backup, tenant isolation, and end-to-end qualification remain operator responsibilities. |
| [Responses API state with Redis](../tutorials/global/api-and-observability#response-api) | Supported integration | Semantic Router owns Responses API configuration; Redis owns durable conversation state. | Manual Redis profiles. Validate the selected Redis topology; production authentication, encryption, persistence, and eviction policy are not supplied by the examples. |
| [Response cache](../tutorials/plugin/response-cache) | Supported integration | Semantic Router owns cache behavior and configuration; the selected external backend owns storage and availability. | Contract/manual coverage. Pin the backend and Router release, validate serialization and expiry behavior, and treat cached prompts and responses as sensitive data. |
| [Valkey vector store](storage-overview) | Supported integration | Semantic Router owns vector-store references; Valkey owns indexing, durability, and access control. | Manual vector-store profiles. Pin Valkey, its Search module, the embedding model, and the Router release as one tested set. |

## Experimental examples

| Example | Classification | Purpose | Production boundary |
| --- | --- | --- | --- |
| KServe example | Experimental example | Demonstrates KServe integration while KServe and the cluster own model serving. | Documented example and smoke helpers. Review platform versions, images, credentials, storage, and cleanup before use. |
| OpenShift example | Experimental example | Adapts Kubernetes resources to OpenShift Routes and security constraints. | Documented example. Review images, Route TLS, credentials, SCCs, caches, and persistent data. |
| Anthropic-compatible backend fixture | Experimental example | Provides an Anthropic-compatible backend for integration tests. | Test fixture only; it is not a production model service. |
| Hallucination policy demo | Experimental example | Exercises fact-check gating and warning behavior. | Manual profile with model prerequisites. Validate model quality and failure policy for the deployment domain. |
| Jailbreak error-handling demo | Experimental example | Exercises jailbreak-classifier failure behavior. | Manual failure-path profile. Do not treat the unreachable test endpoint or permissive policy as production hardening. |
| LLM Katan development backends | Experimental example | Supplies lightweight OpenAI-compatible development backends. | Test fixture only; it does not represent production model quality, capacity, or availability. |
| Observability demo | Experimental example | Demonstrates Prometheus, Grafana, alerts, and Dashboard wiring in a development cluster. | Documented example. Replace example hostnames, TLS material, credentials, retention, and insecure defaults. |
| Responses API Kubernetes demo | Experimental example | Demonstrates Responses API persistence and restart behavior with Redis. | Manual profile. Harden Redis networking, authentication, encryption, persistence, and eviction policy. |
| Route action demo | Experimental example | Demonstrates route-action behavior through a focused Kubernetes topology. | Test/example coverage only; compose it into a maintained stack before production use. |
| Router replay recovery demo | Experimental example | Exercises management-boundary replay and restart recovery. | Manual profile. It is a recovery test topology, not a supported deployment platform. |
| Routing strategy demos | Experimental example | Demonstrates focused routing-strategy configurations and test traffic. | Test/example coverage only. Validate policy quality and backend capacity with representative traffic. |
| Local tools database | Experimental example | Provides local tool definitions used by examples and tests. | Local example data only. Use an authenticated, durable tool registry for production. |

There are no currently shipped deployment options classified as **Deprecated**.
Removed legacy behavior and migration guidance are recorded in
[Upgrade and Rollback](upgrade-rollback). A future deprecated option must remain
in this matrix until its documented removal.

## Hardware overlays

Hardware support applies to a deployment stack; it does not replace one.
Semantic Router and the backend model server are also separate support
boundaries.

| Hardware profile | Status | Deployment guidance | Qualification boundary |
| --- | --- | --- | --- |
| Linux x86-64 CPU | Maintained | Use the standard Router images, local CLI, Helm chart, or Operator. | The normal build, unit, CLI, and Kubernetes profiles provide the broadest recurring coverage. Backend model-server requirements are separate. |
| Linux Arm64 CPU | Build-qualified | Standard Router, Dashboard, ExtProc, and Operator images are published as multi-architecture images where their release workflow declares Arm64. | Multi-architecture image publication is not a promise that every integration or optional native dependency passed on every Arm64 platform. |
| NVIDIA CUDA on Linux x86-64 | Supported integration | Use the `vllm-sr-cuda` image for supported Router-side ONNX models, or keep the Router on CPU and connect a separately qualified NVIDIA vLLM backend. The current image is built on CUDA `12.4.1`. | Follow [NVIDIA CUDA](nvidia-cuda), verify the CUDA execution provider, and pin the image digest. GPU model serving follows vLLM's NVIDIA support matrix; presence here does not qualify every NVIDIA architecture. |
| AMD ROCm on Linux x86-64 | Supported integration | Keep the Router on CPU or use the ROCm Router image for supported local models, and connect a separately qualified ROCm vLLM backend. | Follow [AMD ROCm](amd-rocm) and pin compatible Router, vLLM, ROCm, and model revisions. Hardware-specific runtime coverage is not a per-PR guarantee. |
| AMD AI PC/NPU | Experimental, not qualified | Tracked by [issue #2373](https://github.com/vllm-project/semantic-router/issues/2373). | No maintained deployment contract or compatibility promise yet. |
| NVIDIA DGX Spark Arm64 | Experimental, not qualified | Tracked by [issue #2374](https://github.com/vllm-project/semantic-router/issues/2374). | Arm64 image availability alone does not qualify CUDA, native dependencies, or end-to-end inference on this platform. |
| Other accelerators and operating systems | Not qualified | No maintained Router deployment option is currently declared. | Open a qualification issue with reproducible hardware, software, image, and test evidence before documenting support. |

## Configuration, dependencies, and security

Use one canonical Router YAML document. The CLI may translate it into Helm
values, and the Operator may reconcile it from a custom resource, but neither
option should create a second hand-maintained routing schema. Runtime
configuration examples are references consumed by complete configurations;
they are not deployment manifests.

External gateways, inference platforms, model servers, databases, storage
classes, identity systems, and accelerator runtimes retain their own release
and security policies. For production:

1. pin the Semantic Router release, images or digests, external platform
   versions, model revisions, and configuration together;
2. test the backend directly before testing it through the Router;
3. exercise health, streaming, failure, upgrade, and rollback behavior through
   the actual data plane;
4. move credentials into a secret manager and enable transport security,
   network policy, authentication, and least privilege; and
5. review [Security Hardening](security-hardening),
   [Data and Storage](storage-overview), and
   [Upgrade and Rollback](upgrade-rollback).
