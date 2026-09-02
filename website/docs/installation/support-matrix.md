---
title: Deployment and Hardware Support Matrix
description: Understand which Semantic Router deployment assets and hardware paths are maintained, supported, experimental, or deprecated.
---

# Deployment and Hardware Support Matrix

Use this page to determine what the Semantic Router project maintains and what
you must validate for your environment. An asset being present in the
repository does not, by itself, make every platform version, image, model, or
accelerator combination supported.

For a task-oriented starting point, see [Choose a Deployment](deployment-options).

## Support levels

| Level | Project commitment |
| --- | --- |
| Maintained reference stack | The project owns the installation or lifecycle contract and protects it with repository tests. Pin a Semantic Router release for production. |
| Supported integration | The project maintains the Router-side attachment contract and documentation. The external platform keeps its own support matrix and lifecycle. |
| Experimental example | The asset demonstrates a feature or test topology. Review and adapt it; compatibility and production hardening are not guaranteed. |
| Deprecated | The asset remains temporarily for migration and has a documented replacement or removal path. |

Evidence labels in the tables describe the repository's strongest recurring
check. They are not proof that an arbitrary downstream cluster passed:

- **PR/full CI** means a registered end-to-end profile is selected in pull
  request or full-CI coverage.
- **Contract/build** means source, schema, rendering, image, or packaging
  checks protect the asset without running every external platform.
- **Manual profile** means the repository provides an opt-in test with external
  prerequisites. It does not mean the profile passed in another environment.
- **Documented example** means the project checks the files and documentation,
  but operators own end-to-end qualification.

## Maintained reference stacks

| Asset path | Classification | Capabilities | Ownership and data plane | Version or image policy | Evidence and limits |
| --- | --- | --- | --- | --- | --- |
| `deploy/helm/` | Maintained reference stack | Router, optional Dashboard, ingress, autoscaling, persistence, and observability resources. | The project owns the chart and Router lifecycle; the selected Gateway or Service remains the data plane. | Use the chart from the same Semantic Router release and pin application images. Chart dependencies are declared by the chart. | The [Helm safety gate](https://github.com/vllm-project/semantic-router/blob/main/tools/make/helm.mk) validates rendering, schema, and safety rules. External gateways and storage still require their own tests. |
| `deploy/local/` | Maintained reference stack | Local Router, Envoy, Dashboard, and selected support services through `vllm-sr serve`. | The CLI owns the local stack lifecycle. Model endpoints remain separate. | Use same-release CLI and images; pin image digests for controlled deployments. | The [CLI integration workflow](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/integration-test-vllm-sr-cli.yml) exercises the lifecycle with real containers. Local defaults are for evaluation and development until hardened. |
| `deploy/operator/` | Maintained reference stack | Reconciliation of Router configuration, workload, Service, storage, autoscaling, and optional platform resources. | The Operator reconciles `SemanticRouter` resources; Kubernetes owns workload scheduling and the chosen gateway owns traffic. | Install the CRD, controller, and Router images from one release. Pin the controller image and inspect samples before use. | [Operator CI](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/operator-ci.yml) covers unit, generated-resource, image, and cluster reconciliation contracts. Sample custom resources are not production values. |
| `deploy/kubernetes/crds/` | Maintained reference stack | Kubernetes APIs for intelligent routing resources consumed by their controllers and integrations. | The project owns the API definitions; controllers or integrations own reconciliation and traffic. | Apply CRDs from the same release as the consuming controller or integration. | The [operator test contract](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/test-domain-registry.yaml) protects generated schemas. The directory is not a standalone deployment. |

## Supported integrations

| Asset path | Classification | Ownership and data plane | Version or image policy | Evidence and limits |
| --- | --- | --- | --- | --- |
| `deploy/kubernetes/agentgateway/` | Supported integration | agentgateway owns the Gateway API data plane; Semantic Router supplies ExtProc policy. | The guide pins agentgateway `v1.4.1` and Gateway API `v1.6.0`; pin a Router release instead of `0.0.0-latest`. | PR/full CI profile. agentgateway requires `FullDuplexStreamed` rather than `Streamed` request bodies. |
| `deploy/kubernetes/ai-gateway/` | Supported integration | Envoy AI Gateway and Envoy Gateway own provider traffic; Semantic Router supplies routing policy. | The guide pins Envoy AI Gateway `v1.0.0`, Envoy Gateway `v1.8.1`, Gateway API `v1.5.x`, and Kubernetes `1.32+`. | Default and full-CI profile. Provider compatibility changes independently; verify the selected provider. |
| `deploy/kubernetes/aibrix/` | Supported integration | AIBrix owns model deployment, autoscaling, and replica routing; Semantic Router selects the model or pool. | Follow one supported AIBrix release and pin Router images and configuration to a tested set. | PR/full CI profile. The project does not replace AIBrix's cluster and accelerator support matrix. |
| `deploy/kubernetes/dynamo/` | Supported integration | NVIDIA Dynamo owns graphs, workers, and frontend lifecycle; Semantic Router selects a model target. | The guide covers Dynamo `1.4.0`, Envoy Gateway `v1.9.0`, Gateway API `v1.6.1`, and Kubernetes `1.33`-`1.36`. | Manual profile because it requires a prepared Dynamo/GPU environment. Validate the selected Dynamo release separately. |
| `deploy/kubernetes/istio/` | Supported integration | Istio's Gateway API data plane carries requests; Semantic Router is the ExtProc policy service. | The guide pins Istio `1.29.6`, Gateway API `v1.5.1`, and Kubernetes `1.31`-`1.35`. | PR/full CI profile. Supplied model workloads need NVIDIA GPU capacity and are examples, not a general GPU guarantee. |
| `deploy/kubernetes/llm-d/` | Supported integration | llm-d owns model services, discovery, and replica routing; Semantic Router owns semantic model selection. | Choose one supported llm-d release and its matching APIs, gateway, and images; do not mix copied manifests across releases. | PR/full CI profile. llm-d's release-specific deployment and hardware requirements remain authoritative. |
| `deploy/kubernetes/llmd-base/` | Supported integration | The llm-d Inference Scheduler chooses a replica after Semantic Router selects an `InferencePool`. | Match the `InferencePool` API and names to the selected llm-d release and pin all images. | Contract/build and shared gateway coverage. Do not apply competing direct-Service routes for the same traffic. |
| `deploy/kubernetes/streaming/` | Supported integration | The selected Gateway owns streamed transport; Semantic Router processes the configured ExtProc body mode. | Pin the Router and gateway releases and use a body mode supported by that gateway. | PR/full CI streaming profile. Transport capabilities differ across gateways. |
| `config/runtime/memory/` | Supported integration | Semantic Router owns memory configuration; Milvus or Valkey owns persistence and availability. | Pin the external service and client-compatible schema used by the selected Router release. | Contract and manual integration coverage. Credentials, retention, backup, and tenant isolation are operator responsibilities. |
| `config/runtime/response-api/` | Supported integration | Semantic Router owns Response API configuration; Redis owns durable conversation state. | Use a Redis version and topology tested with the selected release. | Manual Redis profile. Production authentication, encryption, persistence, and eviction policy are not supplied by the example. |
| `config/runtime/response-cache/` | Supported integration | Semantic Router owns cache configuration; the external cache owns storage and availability. | Pin the backend and Router release and validate serialization and expiry behavior. | Contract/manual coverage. Treat cached prompts and responses as sensitive data. |
| `config/runtime/vector-store/` | Supported integration | Semantic Router owns vector-store references; the selected store owns indexing, durability, and access control. | Pin the store version, embedding model, and Router release as one tested set. | Manual vector-store profiles. Index compatibility and data migration remain operator responsibilities. |

## Experimental examples

| Asset path | Classification | Purpose and boundary | Evidence and production cautions |
| --- | --- | --- | --- |
| `deploy/kserve/` | Experimental example | Demonstrates KServe integration; KServe and the cluster own model serving. | Documented example and smoke helpers. Review platform versions, images, credentials, storage, and cleanup before use. |
| `deploy/openshift/` | Experimental example | Adapts Kubernetes resources to OpenShift Routes and security constraints. | Documented example. Review images, Route TLS, credentials, SCCs, caches, and persistent data. |
| `deploy/kubernetes/anthropic-backend/` | Experimental example | Provides an Anthropic-compatible backend fixture for integration tests. | Test fixture only; it is not a production model service. |
| `deploy/kubernetes/hallucination/` | Experimental example | Exercises fact-check gating and warning behavior. | Manual profile with model prerequisites. Validate model quality and failure policy for the deployment domain. |
| `deploy/kubernetes/jailbreak-onerror/` | Experimental example | Exercises jailbreak classifier error handling. | Manual failure-path profile. Do not treat the unreachable test endpoint or permissive policy as production hardening. |
| `deploy/kubernetes/llm-katan/` | Experimental example | Supplies lightweight OpenAI-compatible development backends. | Test fixture only; it does not represent production model quality, capacity, or availability. |
| `deploy/kubernetes/observability/` | Experimental example | Demonstrates Prometheus, Grafana, alerts, and Dashboard wiring in a development cluster. | Documented example. Replace example hostnames, TLS material, credentials, retention, and insecure defaults. |
| `deploy/kubernetes/response-api/` | Experimental example | Demonstrates Response API persistence and restart behavior with Redis. | Manual profile. Harden Redis networking, authentication, encryption, persistence, and eviction policy. |
| `deploy/kubernetes/route-action/` | Experimental example | Demonstrates route-action behavior through a focused Kubernetes topology. | Test/example coverage only; compose it into a maintained stack before production use. |
| `deploy/kubernetes/router-replay/` | Experimental example | Exercises management-boundary replay and restart recovery. | Manual profile. It is a recovery test topology, not a supported deployment platform. |
| `deploy/kubernetes/routing-strategies/` | Experimental example | Demonstrates focused routing-strategy configurations and test traffic. | Test/example coverage only. Validate policy quality and backend capacity with representative traffic. |
| `config/runtime/tools/` | Experimental example | Provides a local tools database used by examples and tests. | Local example data only. Use an authenticated, durable tool registry for production. |

There are no currently shipped deployment directories classified as
**Deprecated**. Removed legacy behavior and migration guidance are recorded in
[Upgrade and Rollback](upgrade-rollback). A future deprecated asset must remain
in this matrix until its documented removal.

## Hardware overlays

Hardware support applies to a deployment stack; it does not replace one.
Semantic Router and the backend model server are also separate support
boundaries.

| Hardware profile | Status | Maintained path and constraints | Qualification boundary |
| --- | --- | --- | --- |
| Linux x86-64 CPU | Maintained | Use the standard Router images, local CLI, Helm chart, or Operator. | The normal build, unit, CLI, and Kubernetes profiles provide the broadest recurring coverage. Backend model-server requirements are separate. |
| Linux Arm64 CPU | Build-qualified | Standard Router, Dashboard, ExtProc, and Operator images are published as multi-architecture images where their release workflow declares Arm64. | Multi-architecture image publication is not a promise that every integration or optional native dependency passed on every Arm64 platform. |
| NVIDIA CUDA on Linux x86-64 | Supported integration | Use the `vllm-sr-cuda` image for supported Router-side ONNX models, or keep the Router on CPU and connect a separately qualified NVIDIA vLLM backend. The current image is built on CUDA `12.4.1`. | Follow [NVIDIA CUDA](nvidia-cuda), verify the CUDA execution provider, and pin the image digest. GPU model serving follows vLLM's NVIDIA support matrix; presence here does not qualify every NVIDIA architecture. |
| AMD ROCm on Linux x86-64 | Supported integration | Keep the Router on CPU or use the ROCm Router image for supported local models, and connect a separately qualified ROCm vLLM backend. | Follow [AMD ROCm](amd-rocm) and pin compatible Router, vLLM, ROCm, and model revisions. Hardware-specific runtime coverage is not a per-PR guarantee. |
| AMD AI PC/NPU | Experimental, not qualified | Tracked by [issue #2373](https://github.com/vllm-project/semantic-router/issues/2373). | No maintained deployment contract or compatibility promise yet. |
| NVIDIA DGX Spark Arm64 | Experimental, not qualified | Tracked by [issue #2374](https://github.com/vllm-project/semantic-router/issues/2374). | Arm64 image availability alone does not qualify CUDA, native dependencies, or end-to-end inference on this platform. |
| Other accelerators and operating systems | Not qualified | No maintained Router deployment path is currently declared. | Open a qualification issue with reproducible hardware, software, image, and test evidence before documenting support. |

## Configuration, dependencies, and security

Use one canonical Router YAML document. The CLI may translate it into Helm
values, and the Operator may reconcile it from a custom resource, but neither
path should create a second hand-maintained routing schema. The assets under
`config/runtime/` are references consumed by complete configurations; they are
not deployment manifests.

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

## Keeping the matrix complete

Every direct child of `deploy/`, every direct child of `deploy/kubernetes/`,
and every runtime integration under `config/runtime/` must appear exactly once
in the tables above. When adding, removing, or changing one of those assets,
update its classification and evidence here.

Run the lightweight inventory check from the repository root:

```bash
python3 tools/ci/check_deployment_support_matrix.py
```

The repository pre-commit configuration runs the same check when deployment
assets, runtime integration assets, or this page change.
