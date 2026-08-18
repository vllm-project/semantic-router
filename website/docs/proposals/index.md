---
title: Proposals
description: Design proposals, proof-of-concept explorations, implementation records, and architectural decisions for vLLM Semantic Router.
---

This collection records ideas and design decisions that need more context than a
feature guide. The status on each page distinguishes proposed work from experiments,
implemented contracts, and decisions that intentionally limit the product scope.

## Routing & Selection

How the router understands requests, bounds model choice, and improves selection
without taking over serving-layer scheduling.

| Proposal | Created | Status | Scope |
| --- | --- | --- | --- |
| [Routing Scope: Per-Query and Capacity-Aware Routing](./batch-and-capacity-aware-routing) | 2026-07-14 | Decision record | Keeps semantic routing per-query and capacity handling in the serving layer. |
| [Router Learning](./router-learning-memory-and-adaptations) | 2026-06-20 | Implemented | Online adaptation, route protection, and offline recipe improvement. |
| [Prompt Classification Routing](./prompt-classification-routing) | 2025-10-08 | Proposal | Keyword, regex, embedding, and classifier signal fusion. |

## Workflows, Memory & Tools

Multi-step execution and contextual capabilities that coordinate models, retrieval,
memory, and tools around a routed request.

| Proposal | Created | Status | Scope |
| --- | --- | --- | --- |
| [Router Flow Workflows](./router-flow-workflows) | 2026-06-30 | Implemented | Bounded static and dynamic multi-model workflows. |
| [Deliberation Algorithms](./deliberation-algorithms) | 2026-06-17 | Proposal | Grounding-aware multi-model synthesis. |
| [Agentic Memory](./agentic-memory) | 2026-02-09 | Proof of concept | Cross-session memory retrieval and persistence. |
| [OpenAI RAG Integration](./agentic-rag) | 2026-01-23 | Implemented | Retrieval through OpenAI Files and Vector Stores. |
| [Advanced Tool Filtering](./advanced-tool-filtering) | 2026-01-14 | Implemented | Explainable filtering and reranking of tool candidates. |

## Safety & Resilience

Failure, eligibility, and response-quality boundaries that keep routing behavior
explicit when a model cannot or should not handle a request.

| Proposal | Created | Status | Scope |
| --- | --- | --- | --- |
| [Model Execution Fallback](./model-execution-fallback) | 2026-08-10 | Proposal | Safe ownership boundary for cross-model fallback. |
| [PRISM](./Prism-153key) | 2026-03-20 | Proposal | Model qualification and legitimacy checks. |
| [TruthLens](./hallucination-mitigation-milestone) | 2025-12-02 | Proposal | Gateway-level hallucination detection and mitigation. |

## Configuration & Protocols

Shared contracts for authoring router behavior and reaching the routing engine from
different client and transport protocols.

| Proposal | Created | Status | Scope |
| --- | --- | --- | --- |
| [Unified Config Contract v0.3](./unified-config-contract-v0-3) | 2026-03-17 | Implemented | One configuration contract across authoring and deployment surfaces. |
| [Multi-Protocol Adapter Architecture](./multi-protocol-adaptor) | 2026-02-18 | Proposal | Protocol-independent access to the routing engine. |

## Serving Integrations

Layering contracts that connect semantic model choice to infrastructure-owned model
serving, replica selection, and execution.

| Proposal | Created | Status | Scope |
| --- | --- | --- | --- |
| [vLLM Production Stack Integration](./production-stack-integration) | 2025-10-13 | Proposal | Layered semantic and infrastructure routing. |
| [NVIDIA Dynamo Integration](./nvidia-dynamo-integration) | 2025-10-09 | Proposal | Semantic routing above Dynamo's worker-level routing. |

Statuses describe the document's current role:

- **Proposal**: a design that is not presented as fully shipped.
- **Proof of concept**: an experiment with explicit production limitations.
- **Implemented**: a contract represented in the current repository.
- **Decision record**: an architectural choice, including work intentionally kept
  outside the router.
