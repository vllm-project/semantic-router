---
sidebar_position: 1
---

# vLLM Semantic Router

**System-Level Intelligence for Mixture-of-Models (MoM)** - An intelligent routing layer that brings collective intelligence to LLM systems. Acting as an Envoy External Processor (ExtProc), it uses a **signal-driven decision engine** and **plugin chain architecture** to capture missing signals, make better routing decisions, and secure your LLM infrastructure.

Use this page as the public docs entrypoint. The site is organized around a few primary journeys:

- understand the concepts and architecture
- bootstrap or configure a local deployment
- deploy to gateways, operators, and platform integrations
- implement routing, cache, safety, and response-api capabilities
- operate, troubleshoot, and tune the system
- integrate against APIs and CRDs
- contribute code, docs, and translations

## Project Goals

We are building the **System Level Intelligence** for Mixture-of-Models (MoM), bringing **Collective Intelligence** into **LLM systems**, answering:

1. **How to capture the missing signals** in request, response and context?
2. **How to combine the signals** to make better decisions?
3. **How to collaborate more efficiently** between different models?
4. **How to secure** the real world and LLM system from jailbreaks, PII leaks, hallucinations?
5. **How to collect valuable signals** and build a self-learning system?

## Core Architecture

### Signal-Driven Decision Engine

Captures and combines multiple request and runtime signals to make intelligent routing decisions:

| Signal family | Description | Use case |
|------------|-------------|----------|
| **keyword** | Pattern matching with AND/OR operators | Fast rule-based routing for specific terms |
| **embedding** | Semantic similarity using embeddings | Intent detection and semantic understanding |
| **domain** | Domain and category classification | Academic and professional domain routing |
| **fact_check / user_feedback / preference** | Quality and user-intent signals | Handle verification needs, corrections, and preferences |
| **language / context / complexity** | Language and request-shape signals | Match queries to suitable models and policies |
| **jailbreak / pii / modality / authz** | Safety, modality, and access signals | Enforce security, multimodal routing, and RBAC-style controls |

**How it works**: Signals are extracted from requests, combined using AND/OR operators in decision rules, and used to select the best model and configuration.

### Processing Surfaces

Extensible request/response processing surfaces can be configured around the decision engine:

| Processing surface | Description | Use case |
|------------|-------------|----------|
| **semantic-cache** | Semantic similarity-based caching | Reduce latency and costs for similar queries |
| **system_prompt** | Dynamic system prompt injection | Add context-aware instructions per route |
| **header_mutation** | HTTP header manipulation | Control routing and backend behavior |
| **hallucination** | Token-level hallucination detection | Real-time fact verification during generation |
| **router_replay / fast_response** | Response shaping and replay flows | Support replay, fast-path responses, and diagnostics |
| **memory / rag** | Retrieval-backed request enrichment | Add memory and document retrieval to routed requests |

## Architecture Overview

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="Signal-Driven Decision + Plugin Chain Architecture" defaultZoom={3.5}>
{`graph TB
    Client[Client Request] --> Envoy[Envoy Proxy]
    Envoy --> Router[Semantic Router ExtProc]

    subgraph "Signal Extraction Layer"
        direction TB
        Keyword[Keyword Signals<br/>Pattern Matching]
        Embedding[Embedding Signals<br/>Semantic Similarity]
        Domain[Domain Signals<br/>Category Classification]
        FactCheck[Fact Check Signals<br/>Verification Need]
        Feedback[User Feedback Signals<br/>Satisfaction Analysis]
        Preference[Preference Signals<br/>LLM-based Matching]
        Language[Language Signals<br/>Multi-language Detection]
        Context[Context Signals<br/>Token Count]
        Complexity[Complexity Signals<br/>Difficulty Classification]
        Safety[Safety Signals<br/>Jailbreak / PII]
        Access[Modality and Authz<br/>Routing Constraints]
    end

    subgraph "Decision Engine"
        Rules[Decision Rules<br/>AND/OR Operators]
        ModelSelect[Model Selection<br/>Priority/Confidence/Algorithms]
    end

    subgraph "Plugin Chain"
        direction LR
        Cache[Semantic Cache]
        Memory[Memory / RAG]
        SysPrompt[System Prompt]
        HeaderMut[Header Mutation]
        Hallucination[Hallucination Detection]
        Replay[Replay / Fast Response]
    end

    Router --> Keyword
    Router --> Embedding
    Router --> Domain
    Router --> FactCheck
    Router --> Feedback
    Router --> Preference
    Router --> Language
    Router --> Context
    Router --> Complexity
    Router --> Safety
    Router --> Access

    Keyword --> Rules
    Embedding --> Rules
    Domain --> Rules
    FactCheck --> Rules
    Feedback --> Rules
    Preference --> Rules
    Language --> Rules
    Context --> Rules
    Complexity --> Rules
    Safety --> Rules
    Access --> Rules

    Rules --> ModelSelect
    ModelSelect --> Cache
    Cache --> Memory
    Memory --> SysPrompt
    SysPrompt --> HeaderMut
    HeaderMut --> Hallucination
    Hallucination --> Replay

    Replay --> Backend[Backend Models]
    Backend --> Math[Math Model]
    Backend --> Creative[Creative Model]
    Backend --> Code[Code Model]
    Backend --> General[General Model]`}
</ZoomableMermaid>

## Key Benefits

### Intelligent Routing

- **Signal Fusion**: Combine multiple signals (keyword + embedding + domain) for accurate routing
- **Adaptive Decisions**: Use AND/OR operators to create complex routing logic
- **Model Specialization**: Route math to math models, code to code models, etc.

### Security & Compliance

- **Multi-layer Protection**: PII detection, jailbreak prevention, hallucination detection
- **Policy Enforcement**: Model-specific PII policies and security rules
- **Audit Trail**: Complete logging of all security decisions

### Performance & Cost

- **Semantic Caching**: 10-100x latency reduction for similar queries
- **Smart Model Selection**: Use smaller models for simple tasks, larger for complex
- **Tool Optimization**: Auto-select relevant tools to reduce token usage

### Flexibility & Extensibility

- **Plugin Architecture**: Add custom processing logic without modifying core
- **Signal Extensibility**: Define new signal types for your use cases
- **Configuration-Driven**: Change routing behavior without code changes

## Use Cases

- **Enterprise API Gateways**: Intelligent routing with security and compliance
- **Multi-tenant Platforms**: Per-tenant routing policies and model selection
- **Development Environments**: Cost optimization through smart model selection
- **Production Services**: High-performance routing with comprehensive monitoring
- **Regulated Industries**: Compliance-ready with PII detection and audit trails

## Documentation Structure

Choose the path that matches your job:

### [Concepts](overview/goals.md)

Understand the goals, mental model, and core routing concepts before changing a deployment.

### [Get Started](installation/installation.md)

Start with the local CLI flow, configuration model, Docker Compose helpers, and local vector-store prerequisites.

### [Deploy & Integrate](installation/k8s/operator.md)

Use the Kubernetes operator, gateway integrations, and platform-specific deployment guides when you need a cluster-facing setup.

### [Capabilities](tutorials/intelligent-route/keyword-routing.md)

Implement routing, semantic cache, response API, safety, and model-selection behaviors using task-oriented tutorials.

### [Operations](tutorials/observability/metrics.md)

Monitor, tune, and troubleshoot the router using observability guides, performance notes, and error references.

### [Reference](api/router.md)

Use the API and CRD reference when you need stable request, schema, or platform contracts.

### [Research & Roadmap](training/training-overview.md)

Review training workflows and forward-looking proposals separately from day-to-day user documentation.

### [Contribute](community/overview.md)

Use the contributor docs for development workflow, documentation maintenance, translation guidance, and code style.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/vllm-project/semantic-router/blob/main/LICENSE) file for details.
