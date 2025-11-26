---
id: gateway-integrations
title: Gateway Integrations
sidebar_label: Gateway Integrations
description: How the Semantic Router plugs into Envoy AI Gateway, Istio, AIBrix, LLM-D, and the vLLM Production Stack, plus what each integration adds.
---

The Semantic Router ships with multiple gateway profiles. This page shows **Which gateway plugs in**, **What SR adds**, and **What’s already validated**.

## High-level topology

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="System Architecture Overview" defaultZoom={5.5}>
{`
flowchart LR
    C[Client / SDK]
    GW["Gateway<br/>(Envoy | Istio | AIBrix | LLM-D | Prod Stack)"]
    SR["Semantic Router<br/>(ExtProc gRPC)"]
    SC["Semantic Cache<br/>(Milvus)"]
    OBS["Telemetry<br/>(OTel → Prom/Grafana)"]
    B1["Cloud LLMs<br/>(OpenAI, Anthropic, ...)"]
    B2["Self-hosted<br/>vLLM workers"]

    C --> GW
    GW -- ExtProc <br/> Inference Extension --> SR
    SR -->|headers: model, safety| GW
    SR --> SC
    SR --> OBS
    GW --> B1
    GW --> B2
    B1 --> OBS
    B2 --> OBS

    style SR fill:#1f2937,stroke:#0ea5e9,stroke-width:2,color:#e5e7eb
    style GW fill:#0f172a,stroke:#a855f7,stroke-width:2,color:#e5e7eb
`}
</ZoomableMermaid>

## Supported Profiles

| Gateway profile      | Integration path                           | SR  adds                                                                            | CI status                                                                                                                                                                                                                                                              | Manifests / config                                                                                                       |
| -------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Envoy AI Gateway** | ExtProc gRPC (Envoy AI Gateway → SR)       | Classification → model header, PII/jailbreak, semantic cache, observability headers | [![integration-test-k8s](https://github.com/vllm-project/semantic-router/actions/workflows/integration-test-k8s.yml/badge.svg)](https://github.com/vllm-project/semantic-router/actions/workflows/integration-test-k8s.yml)<br/>**Validates:**<br/>• **Features:** Classification, Cache, PII, Jailbreak<br/>• **Routing:** Priority, Fallback, Keyword<br/>• **Traffic:** Chat API, Stress tests<br/><br/>[![integration-test-helm](https://github.com/vllm-project/semantic-router/actions/workflows/integration-test-helm.yml/badge.svg)](https://github.com/vllm-project/semantic-router/actions/workflows/integration-test-helm.yml)<br/>**Validates:** Install, Upgrade, Rollback | [`deploy/kubernetes/ai-gateway`](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/ai-gateway) |
| **Istio Gateway**    | Gateway API Inference Extension + ExtProc  | Same as above; demo with dual vLLM backends                                         | Manual guide | [`deploy/kubernetes/istio`](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/istio)           |
| **AIBrix Gateway**   | Envoy Gateway API resources + ExtProc      | SR intelligence in front of AIBrix autoscaler and distributed KV                    | Helm + AIBrix manifests; <br /> follows Envoy ExtProc; <br /> Planned E2E                                                                                                                                                                                                                    | [`deploy/kubernetes/aibrix`](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/aibrix)         |
| **LLM-D Gateway**    | Istio Gateway + LLM-D schedulers + ExtProc | Semantic routing feeds pool selection in LLM-D                                      | Covered by Istio flow; <br /> Planned E2E                                                                                                                                                                                                                                         | [`deploy/kubernetes/llmd-base`](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/llmd-base)   |

> **Reading map**: pick your gateway, open the install guide, then jump to the manifests to see the exact resources the diagram refers to.

## Request Flow

<ZoomableMermaid title="System Architecture Overview" defaultZoom={5.5}>
{`
sequenceDiagram
    autonumber
    participant Client
    participant Gateway
    participant SR as Semantic Router
    participant Cache as Semantic Cache
    participant Upstream as LLM Backends

    Client->>Gateway: OpenAI-compatible request
    Gateway->>SR: ExtProc gRPC (headers/body)
    SR->>SR: PII / jailbreak / category classification
    SR->>Cache: Semantic lookup
    alt cache hit
        SR-->>Gateway: Headers + cached response
    else miss
        SR-->>Gateway: Route headers (model, policy flags)
        Gateway->>Upstream: Forward to chosen backend
        Upstream-->>Gateway: LLM response
        Gateway-->>SR: Response headers/body (optional)
        SR->>Cache: Write entry
    end
    Gateway-->>Client: Final response
`}
</ZoomableMermaid>

## Where to go next

- **Envoy AI Gateway install**: [installation/k8s/ai-gateway](../../installation/k8s/ai-gateway)
- **Istio Gateway install**: [installation/k8s/istio](../../installation/k8s/istio)
- **AIBrix Gateway install**: [installation/k8s/aibrix](../../installation/k8s/aibrix)
- **LLM-D Gateway install**: [installation/k8s/llm-d](../../installation/k8s/llm-d)
