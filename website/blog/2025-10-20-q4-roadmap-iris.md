---
slug: q4-roadmap-iris
title: "Semantic Router Q4 2025 Roadmap: Journey to Iris"
authors: [Xunzhuo, rootfs, wangchen615, yuezhu1]
tags: [roadmap, release, iris, v0.1]
---

# Semantic Router Q4 2025 Roadmap: Journey to Iris

As we approach the end of 2025, we're excited to share our Q4 2025 roadmap for vLLM Semantic Router. This quarter marks a significant milestone in our project's evolution as we prepare for our first major release: **v0.1, codename "Iris"**, expected in late 2025 to early 2026.

![iris](/img/q4.png)

<!-- truncate -->

## About Our Release Naming Convention

Starting with v0.1, each major release of vLLM Semantic Router will carry a codename inspired by figures from Greek mythology. These names reflect the essence and purpose of each release, connecting ancient wisdom with modern AI infrastructure.

Our inaugural release is named **Iris** (Ἶρις), after the Greek goddess of the rainbow and divine messenger of the Olympian gods. In mythology, Iris served as the swift-footed messenger who bridged the gap between gods and mortals, traveling on the arc of the rainbow to deliver messages across vast distances. She personified the connection between heaven and earth, ensuring that communication flowed seamlessly across different realms.

This symbolism perfectly captures the essence of vLLM Semantic Router: a system that bridges the gap between users and diverse AI models, intelligently routing requests across different LLM providers and architectures. Just as Iris connected different worlds through her rainbow bridge, our router connects applications to the right models through intelligent semantic understanding. The rainbow itself—a spectrum of colors working in harmony—mirrors our vision of orchestrating multiple models in a unified, efficient system.

With the Iris release, we're establishing the foundation for reliable, intelligent, and secure AI model routing that will serve as the bridge for modern AI applications.

## Q4 2025 Focus Areas

Our Q4 roadmap centers on seven critical pillars that will transform vLLM Semantic Router from an experimental project into a production-ready platform. These initiatives address the most pressing needs identified by our community and represent the essential groundwork for v0.1.

### 1. Semantic Chain for Fusion Intelligent Routing

**The Challenge**

Current routing relies exclusively on ModernBERT classification for semantic understanding. While powerful, this approach has limitations: it cannot perform deterministic routing based on specific keywords, lacks pattern-based detection for safety and compliance, and misses opportunities for specialized domain classification that could enhance routing accuracy and flexibility.

**The Innovation**

We're introducing a **unified content scanning and routing framework** that extends semantic routing with four complementary signal sources, all integrated through a Signal Fusion Layer:

**1. Keyword-Based Routing**

- Deterministic, fast Boolean logic for exact term matching
- Route queries containing "kubernetes" or "CVE-" patterns directly to specialized models
- Eliminate unnecessary ML inference for technology-specific queries

**2. Regex Content Scanning**

- Pattern-based detection for safety, compliance, and structured data
- Guaranteed blocking of PII patterns (SSN, credit cards) with no ML false negatives
- RE2 engine with ReDoS protection for security-critical applications

**3. Embedding Similarity Scanning**

- Semantic concept detection robust to paraphrasing
- Detect "multi-step reasoning" intent even when phrased as "explain thoroughly"
- Reuses existing BERT embedder for zero additional model overhead

**4. Domain Classification**

- **In-Tree BERT Classification**: Lightweight BERT-based domain classifiers running directly in the router process for low-latency intent detection
- **Out-of-Tree MCP Classification**: Advanced domain-specific classifiers deployed as MCP servers for specialized routing scenarios (legal, medical, financial domains)
- Hierarchical classification with confidence scoring for multi-domain queries

**Dual Execution Paths**

- **In-Tree Path**: Low-latency signal providers running directly in the router process
- **Out-of-Tree Path**: MCP (Model Context Protocol) servers for massive rule sets, custom matching engines (Aho-Corasick, Hyperscan), and domain-specific algorithms

**Signal Fusion Layer**

The decision-making engine that combines all signals into actionable routing decisions:

- **Priority-based policy evaluation**: Safety blocks (200) → Routing overrides (150) → Category boosting (100) → Consensus (50) → Default (0)
- **Boolean expressions**: Combine multiple signals with AND, OR, NOT operators
- **Flexible actions**: Block, route to specific models, boost category weights, or fallthrough to BERT

**Impact**

This framework enables:

- Fast deterministic routing for technology-specific queries
- Guaranteed compliance with safety and regulatory requirements
- Semantic intent detection that complements BERT classification
- Specialized domain classification for vertical-specific routing (legal, medical, financial)
- Flexible deployment options with both in-tree and out-of-tree execution paths
- Graceful degradation and backward compatibility with existing routing

The Semantic Chain for Fusion Intelligent Routing represents a fundamental shift from pure ML-based routing to a hybrid approach that leverages the best of deterministic, pattern-based, semantic, and domain-specific classification methods.

### 2. Extensible Serving Architecture: Modular Candle-Binding for MoM Family

**The Challenge**

Our Rust-based candle-binding codebase has grown organically into a 2,600+ line monolithic structure. This architecture was designed for a handful of models, but now faces a critical challenge: supporting the entire **MoM (Mixture of Models) Family** with its diverse model architectures, specialized classifiers, and LoRA-adapted variants. The current monolithic design makes it nearly impossible to efficiently serve multiple model types simultaneously.

**The Vision**

We're restructuring the candle-binding into an **extensible serving architecture** specifically designed to support the MoM Family's diverse model ecosystem. This modular design enables seamless addition of new MoM models without code changes, efficient multi-model serving, and clear separation between model architectures and serving logic.

**Layered Architecture for MoM Models**

- **Core Layer**: Unified error handling, configuration management, device initialization, and weight loading shared across all MoM models
- **Model Architectures Layer**: Modular implementations of BERT (mom-similarity-flash, mom-pii-flash, mom-jailbreak-flash), ModernBERT, and Qwen3 (mom-brain-pro/max, mom-expert-* series) with extensible traits for future MoM additions
- **Classifiers Layer**: Specialized implementations for sequence classification (intent routing), token classification (PII/jailbreak detection), and LoRA support (fine-tuned MoM experts)
- **FFI Layer**: Centralized memory safety checks and C-compatible interfaces for Go integration

**Impact**

This extensible architecture enables:

- **Rapid MoM Model Deployment**: Add new MoM models (mom-expert-math-flash, mom-brain-max) by implementing standard traits
- **Efficient Multi-Model Serving**: Serve multiple MoM models simultaneously with shared infrastructure
- **LoRA Support**: Native support for LoRA-adapted MoM experts with high-confidence routing
- **Backward Compatibility**: Existing Go bindings continue to work without changes

This transformation positions the serving layer as a scalable foundation for the entire MoM Family ecosystem, enabling us to rapidly expand our model offerings while maintaining performance and reliability.

### 3. Model Unification: The MoM (Mixture of Models) Family

**The Challenge**

Despite developing a comprehensive family of specialized routing models, our codebase still references legacy models scattered across configuration files. This fragmentation creates confusion, inconsistent performance, and a steep learning curve for new users.

**The Solution**

We're migrating the entire system to use the **MoM Family** as the primary built-in models:

- **🧠 Intelligent Routing**: mom-brain-flash/pro/max for intent classification with clear latency-accuracy trade-offs
- **🔍 Similarity Search**: mom-similarity-flash for semantic matching
- **🔒 Prompt Guardian**: mom-jailbreak-flash and mom-pii-flash for security and privacy
- **🎯 SLM Experts**: Specialized models for math, science, social sciences, humanities, law, and general tasks

**Key Features**

- **Centralized Registry**: Single source of truth for all MoM models with metadata, capabilities, and recommended use cases
- **Simplified Configuration**: Reference models by name (`mom-brain-flash`) instead of complex paths
- **Auto-Discovery**: Intelligent model detection and validation
- **Performance Optimization**: All MoM models are specifically trained and optimized for vLLM-SR routing tasks

This unification provides users with a clear, consistent model selection experience while ensuring optimal performance for every routing scenario.

### 4. Architectural Evolution: Model-Based Routing Core

**The Challenge**

Our current routing implementation, inherited from traditional cluster-based approaches, has reached its architectural limits. The tight coupling between routing logic and cluster management prevents us from supporting the diverse LLM deployment scenarios that modern AI applications demand—from hybrid cloud deployments to multi-provider orchestration.

**The Vision**

We're reimagining our routing architecture with a clean separation of concerns: semantic routing focuses purely on intelligent model selection, while traffic management is delegated to the AI Gateway layer. This modular approach transforms the semantic router into a global external processor that operates transparently within the gateway infrastructure.

**Key Capabilities**

- **Universal Connectivity**: Support for HTTPS, HTTP, IP-based, and DNS-based connections to any LLM provider
- **Hybrid Routing**: Seamlessly route between in-cluster services and external providers (Claude, Gemini, DeepSeek, etc.)
- **Advanced Traffic Management**: Model-level failover, weighted distribution, circuit breaking, and health checks
- **Enterprise Features**: Built-in authentication, retry mechanisms, and token-based rate limiting

This architectural shift enables vLLM Semantic Router to scale from single-cluster deployments to global, multi-cloud AI infrastructures while maintaining the simplicity and performance that users expect.

### 5. Next-Generation API: OpenAI Responses API Support

**The Challenge**

The traditional Chat Completions API (`/v1/chat/completions`) is stateless and designed for single-turn interactions. Modern AI applications—especially agents, multi-turn conversations, and agentic workflows—require stateful interactions, advanced tool orchestration, and long-running background tasks. Without Responses API support, vLLM Semantic Router cannot intelligently route these next-generation workloads.

**The Vision**

Add comprehensive support for the OpenAI Responses API (`/v1/responses`), enabling intelligent routing for stateful, multi-turn, and agentic LLM workflows while preserving all advanced features of the API.

**Key Capabilities**

**Stateful Conversations**

- Built-in conversation state management with `previous_response_id` chaining
- Automatic context preservation across multiple turns
- Intelligent routing that maintains conversation context and intent classification history

**Advanced Tool Orchestration**

- Native support for code interpreter with container management
- Function calling and tool execution routing
- Image generation and editing capabilities
- MCP (Model Context Protocol) server integration for external tools
- File uploads and processing (PDFs, images, structured data)

**Agentic Workflows**

- Background task processing for long-running agent operations
- Asynchronous execution with polling support for complex reasoning tasks
- Resumable streaming with sequence tracking for dropped connections
- Support for reasoning models (o1, o3, o4-mini) with encrypted reasoning items

**Semantic Routing Integration**

- Extract and classify intent from Responses API `input` field (text, messages, or mixed content)
- Apply intelligent model selection based on conversation history and tool requirements
- Route multi-turn conversations to models optimized for stateful interactions
- Preserve VSR (vLLM Semantic Router) headers for routing metadata across response chains

**Impact**

Responses API support positions vLLM Semantic Router at the forefront of agentic AI infrastructure:

- Enable routing for modern agent frameworks and multi-turn applications
- Support complex workflows requiring code execution, file processing, and external tool integration
- Provide intelligent model selection for reasoning-heavy tasks and long-running operations
- Maintain semantic router's value proposition (cost optimization, latency reduction) for next-generation LLM APIs

This capability is essential for vLLM Semantic Router to remain relevant as the industry shifts from simple chat completions to sophisticated, stateful, tool-augmented AI agents.

### 6. Intelligent MCP Gateway: Smart Tool Management and Selection

**The Challenge**

As AI agents increasingly rely on external tools and services through the Model Context Protocol (MCP), managing and selecting the right tools for each task becomes critical. Current approaches lack intelligent tool discovery, selection optimization, and centralized management, leading to inefficient tool usage and increased latency in agentic workflows.

**The Innovation**

We're introducing an **Intelligent MCP Gateway** that serves as a unified control plane for MCP tools with smart selection capabilities:

**MCP Tool Management**

- **Centralized Registry**: Unified catalog of available MCP servers and tools with metadata, capabilities, and performance characteristics
- **Dynamic Discovery**: Automatic detection and registration of MCP servers in the cluster
- **Health Monitoring**: Real-time health checks and availability tracking for all registered MCP tools
- **Version Management**: Support for multiple versions of MCP tools with seamless upgrades and rollbacks

**Intelligent Tool Selection**

- **Semantic Matching**: Analyze user intent and task requirements to automatically select the most appropriate tools
- **Context-Aware Routing**: Consider conversation history, user preferences, and task complexity for tool selection
- **Performance Optimization**: Route tool requests based on latency, cost, and success rate metrics
- **Fallback Strategies**: Automatic failover to alternative tools when primary options are unavailable

**Integration with Fusion Routing**

- Seamlessly integrate with the Semantic Chain for unified routing decisions
- Combine tool selection with model selection for optimal agentic workflows
- Support both in-tree (low-latency) and out-of-tree (MCP server) tool execution paths

**Impact**

The Intelligent MCP Gateway enables:

- Simplified tool management for complex agentic applications
- Reduced latency through intelligent tool selection and caching
- Improved reliability with automatic failover and health monitoring
- Enhanced developer experience with centralized tool discovery and configuration
- Cost optimization by routing to the most efficient tools for each task

This gateway positions vLLM Semantic Router as a comprehensive orchestration layer for modern AI agents, managing not just model selection but also the tools and services that agents rely on.

### 7. Enterprise Readiness: Production Deployment Tools

**The Challenge**

While vLLM Semantic Router works well for experimental deployments, production adoption requires professional-grade deployment tools, comprehensive monitoring, and intuitive management interfaces.

**The Deliverables**

#### Helm Chart Support

Professional Kubernetes deployment with:

- Templated manifests for all resources
- Values-driven configuration for different environments
- Built-in versioning and rollback capabilities
- Best practices for security, scaling, and resource management

#### Modern Management Dashboard

A comprehensive web-based control plane featuring:

- **Visual Route Builder**: Drag-and-drop interface for creating SemanticRoute configurations
- **Interactive Playground**: Test routing decisions, compare models, and visualize filter chains
- **Real-time Monitoring**: Live metrics, request tracing, and health status
- **Analytics & Insights**: Cost analysis, performance benchmarks, and routing effectiveness
- **User Management**: Role-based access control, API key management, and audit logs

These enterprise features will dramatically lower the barrier to entry, improve operational efficiency, and make vLLM Semantic Router accessible to organizations of all sizes.

## Ecosystem Integration

Beyond the seven core pillars, we're actively exploring integrations with key platforms in the AI infrastructure ecosystem. These five integrations are **work-in-progress and good-to-have** features that will expand vLLM Semantic Router's reach and interoperability:

### vLLM Production Stack

[vLLM Production Stack](https://docs.vllm.ai/projects/production-stack) is vLLM's reference system for Kubernetes-native cluster-wide deployment with community-driven performance optimization. It provides a reference implementation on how to build an inference stack on top of vLLM with Helm chart-based deployment.

Deep integration with the vLLM Production Stack will enable seamless model serving, monitoring, and orchestration. This integration will provide native support for vLLM's advanced features like PagedAttention, continuous batching, and optimized CUDA kernels, ensuring maximum performance for production workloads.

### llm-d

[llm-d](https://llm-d.ai/) is a Kubernetes-native high-performance distributed LLM inference framework built on vLLM. Founded by Red Hat, Google Cloud, CoreWeave, and IBM Research, with contributions from NVIDIA, Hugging Face, Intel, Lambda, and Mistral AI, llm-d provides well-lit paths for anyone to serve large generative AI models at scale with distributed inference capabilities.

Integration with llm-d will bring intelligent semantic routing to Kubernetes-native distributed inference deployments. This partnership will enable llm-d users to leverage MoM models and fusion routing for efficient model selection across distributed inference clusters, optimizing resource utilization and performance in cloud-native environments.

### NVIDIA Dynamo

[NVIDIA Dynamo](https://developer.nvidia.com/dynamo) is NVIDIA's high-performance, low-latency inference platform that supports all major frameworks including TensorRT-LLM. It delivers scalable, efficient inference with GPU optimization and includes intelligent inference optimizations that boost token generation performance by over 30x per GPU, with support for advanced features like disaggregated serving.

Integration with NVIDIA Dynamo will leverage cutting-edge GPU acceleration and optimization frameworks to deliver industry-leading latency and throughput for semantic routing operations. This partnership will enable seamless deployment of MoM models on NVIDIA-accelerated infrastructure with optimal performance.

### vLLM AIBrix

[vLLM AIBrix](https://github.com/vllm-project/aibrix) is an open-source initiative designed to provide essential building blocks to construct scalable GenAI inference infrastructure. As a cloud-native framework, AIBrix serves as an infrastructure orchestrator and workload control plane, offering cost-efficient and pluggable components for large-scale LLM serving with simplified deployment and management.

Collaboration with vLLM AIBrix will enable unified control planes, advanced observability, and streamlined deployment workflows across hybrid and multi-cloud environments. This integration will make it easier for enterprises to adopt and scale vLLM Semantic Router with production-ready infrastructure components.

---

These ecosystem integrations represent our commitment to building an open, interoperable platform that works seamlessly with the broader AI infrastructure landscape. While not required for the v0.1 release, they demonstrate our vision for vLLM Semantic Router as a foundational component in modern AI stacks.

## Timeline and Release Plan

**v0.1 "Iris" Release (Late 2025 - Early 2026):**

- All P0 priority issues resolved
- Seven foundational pillars fully implemented
- Comprehensive documentation and migration guides
- Production-ready deployment tools (Helm charts, dashboard)
- Full Responses API, Intelligent MCP Gateway, and Semantic Chain for Fusion Intelligent Routing support
- Community celebration and feedback collection

## Looking Beyond Iris

The Iris release establishes the foundation, but our vision extends far beyond v0.1. Future releases will introduce:

- Advanced multi-model orchestration strategies
- Federated routing across distributed clusters
- Enhanced reasoning capabilities and chain-of-thought routing
- Deeper integration with the broader vLLM ecosystem

Each release will carry its own mythological codename, reflecting the unique character and capabilities it brings to the project.

![iris](/img/code.png)

## Get Involved

This roadmap represents our commitment to building production-ready AI infrastructure, but we can't do it alone. We invite the community to:

- **Review and provide feedback** on the P0 issues
- **Contribute code** to any of the initiatives
- **Test early releases** and share your experiences
- **Suggest improvements** to the roadmap

Together, we're building the bridge that will connect the next generation of AI applications to the models they need—just as Iris connected the realms of gods and mortals.

---

**Follow our progress:**

- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Issues: [P0 Priority Issues](https://github.com/vllm-project/semantic-router/issues?q=is%3Aissue+state%3Aopen+label%3Apriority%2FP0)

*The rainbow bridge awaits. Let's build it together.* 🌈
