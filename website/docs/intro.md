---
sidebar_position: 1
---

# vLLM Semantic Router

**System-Level Intelligence for Mixture-of-Models (MoM)** - An intelligent routing layer that brings collective intelligence to LLM systems. Acting as an Envoy External Processor (ExtProc), it uses a **signal-driven decision engine** and **plugin chain architecture** to capture missing signals, make better routing decisions, and secure your LLM infrastructure.

## Project Goals

We are building the **System Level Intelligence** for Mixture-of-Models (MoM), bringing **Collective Intelligence** into **LLM systems**, answering:

1. **How to capture the missing signals** in request, response and context?
2. **How to combine the signals** to make better decisions?
3. **How to collaborate more efficiently** between different models?
4. **How to secure** the real world and LLM system from jailbreaks, PII leaks, hallucinations?
5. **How to collect valuable signals** and build a self-learning system?

## Core Architecture

### Signal-Driven Decision Engine

Captures and combines **8 types of signals** to make intelligent routing decisions:

| Signal Type | Description | Use Case |
|------------|-------------|----------|
| **keyword** | Pattern matching with AND/OR operators | Fast rule-based routing for specific terms |
| **embedding** | Semantic similarity using embeddings | Intent detection and semantic understanding |
| **domain** | MMLU domain classification (14 categories) | Academic and professional domain routing |
| **fact_check** | ML-based fact-checking requirement detection | Identify queries needing fact verification |
| **user_feedback** | User satisfaction and feedback classification | Handle follow-up messages and corrections |
| **preference** | LLM-based route preference matching | Complex intent analysis via external LLM |
| **language** | Multi-language detection (100+ languages) | Route queries to language-specific models |
| **latency** | TPOT-based latency evaluation | Route latency-sensitive queries to faster models based on real-time TPOT |

**How it works**: Signals are extracted from requests, combined using AND/OR operators in decision rules, and used to select the best model and configuration.

### Plugin Chain Architecture

Extensible plugin system for request/response processing:

| Plugin Type | Description | Use Case |
|------------|-------------|----------|
| **semantic-cache** | Semantic similarity-based caching | Reduce latency and costs for similar queries |
| **jailbreak** | Adversarial prompt detection | Block prompt injection and jailbreak attempts |
| **pii** | Personally identifiable information detection | Protect sensitive data and ensure compliance |
| **system_prompt** | Dynamic system prompt injection | Add context-aware instructions per route |
| **header_mutation** | HTTP header manipulation | Control routing and backend behavior |
| **hallucination** | Token-level hallucination detection | Real-time fact verification during generation |

**How it works**: Plugins form a processing chain, each plugin can inspect/modify requests and responses, with configurable enable/disable per decision.

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
        Domain[Domain Signals<br/>MMLU Classification]
        FactCheck[Fact Check Signals<br/>Verification Need]
        Feedback[User Feedback Signals<br/>Satisfaction Analysis]
        Preference[Preference Signals<br/>LLM-based Matching]
        Language[Language Signals<br/>Multi-language Detection]
        Latency[Latency Signals<br/>TPOT-based Routing]
    end

    subgraph "Decision Engine"
        Rules[Decision Rules<br/>AND/OR Operators]
        ModelSelect[Model Selection<br/>Priority/Confidence]
    end

    subgraph "Plugin Chain"
        direction LR
        Cache[Semantic Cache]
        Jailbreak[Jailbreak Guard]
        PII[PII Detector]
        SysPrompt[System Prompt]
        HeaderMut[Header Mutation]
        Hallucination[Hallucination Detection]
    end

    Router --> Keyword
    Router --> Embedding
    Router --> Domain
    Router --> FactCheck
    Router --> Feedback
    Router --> Preference
    Router --> Language
    Router --> Latency

    Keyword --> Rules
    Embedding --> Rules
    Domain --> Rules
    FactCheck --> Rules
    Feedback --> Rules
    Preference --> Rules
    Language --> Rules
    Latency --> Rules

    Rules --> ModelSelect
    ModelSelect --> Cache
    Cache --> Jailbreak
    Jailbreak --> PII
    PII --> SysPrompt
    SysPrompt --> HeaderMut
    HeaderMut --> Hallucination

    Hallucination --> Backend[Backend Models]
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

## Quick Links

- [**Installation**](installation/installation.md) - Setup and installation guide
- [**Overview**](overview/goals.md) - Project goals and core concepts
- [**Configuration**](installation/configuration.md) - Configure signals and routing decisions
- [**Tutorials**](tutorials/intelligent-route/keyword-routing.md) - Step-by-step guides

## Documentation Structure

This documentation is organized into the following sections:

### [Overview](overview/goals.md)

Learn about our goals, semantic routing concepts, collective intelligence, and signal-driven decisions.

### [Installation & Configuration](installation/installation.md)

Get started with installation and learn how to configure signals, decisions, and plugins.

### [Tutorials](tutorials/intelligent-route/keyword-routing.md)

Step-by-step guides for implementing intelligent routing, semantic caching, content safety, and observability.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/vllm-project/semantic-router/blob/main/LICENSE) file for details.
