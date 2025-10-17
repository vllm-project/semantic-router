---
slug: mom-family
title: "MoM: Specialized Models for Intelligent Routing"
authors: [Xunzhuo]
tags: [mom, models, routing, announcement]
---

![MoM Family](/img/mom-family.png)

**One fabric. Many minds.** We're introducing **MoM** (Mixture of Models)—a family of specialized routing models that power vLLM-SR's intelligent decision-making.

<!-- truncate -->

## Why MoM?

vLLM-SR solves a critical problem: **how to route LLM requests to the right model at the right time**. Not every query needs the same resources—"What's the weather?" shouldn't cost as much as "Analyze this legal contract."

## MoM System Card

A quick overview of all MoM models:

| Category | Model | Size | Base Model | Latency | Purpose |
|----------|-------|------|------------|---------|---------|
| **🧠 Intelligent Routing** | mom-brain-flash | Flash | ModernBERT | &lt;10ms | Ultra-fast intent classification |
| | mom-brain-pro | Pro | Qwen 0.6B | ~30-50ms | Balanced routing with reasoning |
| | mom-brain-max | Max | Qwen 1.7B | ~50-100ms | Maximum accuracy for complex decisions |
| **🔍 Similarity Search** | mom-similarity-flash | Flash | ModernBERT | &lt;10ms | Semantic similarity matching |
| **🔒 Prompt Guardian** | mom-jailbreak-flash | Flash | ModernBERT | &lt;10ms | Jailbreak/attack detection |
| | mom-pii-flash | Flash | ModernBERT | &lt;10ms | PII detection & privacy protection |
| **🎯 SLM Experts** | mom-expert-math-flash | Flash | Qwen 0.6B | ~30-50ms | Mathematics routing |
| | mom-expert-math-pro | Pro | Qwen 1.7B | ~50-100ms | Advanced math with reasoning |

**Key Insights:**

- **4 Categories** × **3 Size Variants** = Flexible routing architecture
- **ModernBERT** (encoder-only) → Sub-10ms latency for high-throughput scenarios
- **Qwen** (decoder-only) → Explainable decisions with reasoning capabilities
- **Flash** models achieve 10,000+ QPS on commodity hardware

## The Evolution: From Encoder-Only to Mixture-of-Models

### Where We Started: ModernBERT Foundation

vLLM-SR initially built its routing intelligence entirely on **ModernBERT** (encoder-only models):

**Advantages**:

- ⚡ **Blazing fast**: Sub-10ms inference latency
- 📊 **High throughput**: 10,000+ QPS on commodity hardware
- 💰 **Cost-effective**: Minimal compute requirements
- 🎯 **Proven accuracy**: Strong performance on classification tasks

**Limitations**:

- ❌ **Black box decisions**: No explanation for routing choices
- ❌ **Limited reasoning**: Cannot handle complex, multi-step logic
- ❌ **Fixed capabilities**: Hard to extend with new behaviors
- ❌ **No tool integration**: Cannot leverage external tools or APIs

### Why We're Evolving: Decoder-Only Models

As vLLM-SR adoption grew, we encountered more diverse scenarios and requirements:

- **Explainability**: Users need to understand *why* a query was routed to a specific model
- **Complex reasoning**: Some routing decisions require multi-step analysis
- **Agentic workflows**: Integration with tool calling, function execution, and external APIs
- **Advanced techniques**: Reinforcement learning (RL), sophisticated post-training methods
- **Domain expertise**: Specialized routing for legal, medical, scientific domains

**The Solution**: Expand to decoder-only models while keeping encoder speed where it matters.

### The MoM Architecture: Best of Both Worlds

**Mixture-of-Models (MoM)** is both a philosophy and an architecture:

1. **Backend LLM Architecture** — Route requests to the optimal downstream model (GPT-4, Claude, Llama, etc.)
2. **Router Internal Design** — The router itself uses multiple specialized models working together

Our MoM approach combines encoder and decoder strengths:

- ⚡ **Encoders** — Fast classification (sub-10ms latency) for high-throughput scenarios
- 🧠 **Decoders** — Explainable decisions with reasoning for transparency
- 🎯 **Domain Agents** — Expert routing with specialized knowledge

This hybrid architecture lets you choose the right tool for each job: speed when you need it, reasoning when it matters.

**Key Insight**: Just as vLLM-SR routes to different backend LLMs, the router itself is powered by a mixture of specialized models—each optimized for specific routing tasks (security, similarity, intent classification, domain expertise).

## The MoM Model Family

We organize MoM models into **four categories** with **three size variants** (Flash, Pro, Max):

### 🧠 Intelligent Routing

Smart routing models with three size variants:

| Model | Size | Base Model | Purpose |
|-------|------|------------|---------|
| **mom-brain-flash** | Flash | ModernBERT | Ultra-fast intent classification (sub-10ms latency) |
| **mom-brain-pro** | Pro | Qwen 0.6B | Balanced performance with reasoning capabilities |
| **mom-brain-max** | Max | Qwen 1.7B | Maximum accuracy for complex routing decisions |

**Architecture**: Flash is based on ModernBERT (encoder-only), while Pro and Max are based on Qwen 0.6B and 1.7B (decoder-only) models.

### 🔍 Similarity Search

Semantic similarity and vector search:

| Model | Size | Base Model | Purpose |
|-------|------|------------|---------|
| **mom-similarity-flash** | Flash | ModernBERT | Fast semantic similarity matching for route selection |

**Architecture**: Based on ModernBERT (encoder-only) for high-speed embedding generation.

### 🔒 Prompt Guardian

Security and safety checks before routing:

| Model | Size | Base Model | Purpose |
|-------|------|------------|---------|
| **mom-jailbreak-flash** | Flash | ModernBERT | Jailbreak/attack detection (security) |
| **mom-pii-flash** | Flash | ModernBERT | PII detection (privacy protection) |

**Architecture**: Both based on ModernBERT (encoder-only) for ultra-fast security checks.

### 🎯 SLM Experts

Specialized small language models for domain-specific routing:

| Model | Size | Base Model | Domain |
|-------|------|------------|--------|
| **mom-expert-math-flash** | Flash | Qwen 0.6B | Mathematics (algebra, calculus, statistics) |
| **mom-expert-math-pro** | Pro | Qwen 1.7B | Advanced mathematics with reasoning |

**Architecture**: Based on Qwen models (decoder-only) for domain-specific reasoning capabilities.

## Design Principles

**Safety-First**: Prompt Guardian models (PII, jailbreak detection) run before routing—security at the edge.

**Speed ↔ Capability**: Choose Flash for sub-10ms latency, Pro for balanced performance, or Max for maximum accuracy. Different sizes, different SLAs.

**Domain Expertise**: SLM Expert models achieve 15-25% better accuracy on domain-specific tasks vs. generalist routing. Math queries go to math experts.

## How vLLM-SR Uses MoM

MoM operates at **two levels** in vLLM-SR:

### Level 1: Router Internal Architecture (MoM Inside)

The router itself is a mixture of specialized models working together in a pipeline:

1. **Security Check** → `mom-jailbreak-flash` and `mom-pii-flash` filter malicious/sensitive requests
2. **Intent Classification** → `mom-brain-*` models (flash/pro/max) determine query type and routing decisions
3. **Similarity Search** → `mom-similarity-flash` finds semantically similar routes
4. **Domain Routing** → `mom-expert-*` models route specialized queries to optimal downstream models

Each stage uses the **right model for the right task**: fast encoders for security checks, reasoning decoders for complex decisions, domain experts for specialized queries.

### Level 2: Backend LLM Orchestration (MoM Outside)

The router then directs requests to the optimal backend LLM:

- **Simple queries** → Lightweight models (Llama 3.2, Qwen 2.5)
- **Complex queries** → Premium models (GPT-4, Claude 3.5)
- **Domain-specific** → Specialized models (Code Llama, Mistral Math)

This dual-level MoM architecture achieves **2x+ cost reduction** while maintaining quality, similar to [RouteLLM](https://arxiv.org/abs/2406.18665).

**The Philosophy**: Mixture-of-Models all the way down—from the router's internal decision-making to the backend LLM selection.

## What's Next: Exploring Frontier Techniques

The move to decoder-only models opens exciting possibilities for vLLM-SR:

### 🤖 Agentic Routing

Decoder models can act as intelligent agents that:

- Dynamically select and orchestrate multiple models
- Make multi-step routing decisions with tool calling
- Adapt routing strategies based on feedback

### 🎯 Reinforcement Learning (RL)

Apply RL techniques to optimize routing decisions:

- Learn from user feedback and model performance
- Discover optimal routing policies through trial and error
- Continuously improve cost-quality trade-offs

### 🔧 Advanced Post-Training

Leverage cutting-edge post-training methods:

- **Distillation**: Transfer knowledge from large models to efficient routers
- **Preference learning**: Train on human feedback (RLHF, DPO)
- **Domain adaptation**: Fine-tune for specific industries or use cases

### 🛠️ Tool Integration

Enable routers to:

- Call external APIs for context-aware routing
- Query databases for historical routing patterns
- Integrate with monitoring systems for real-time optimization

**The vision**: vLLM-SR routers that not only classify but *reason*, *learn*, and *adapt*.

## Model Naming Convention

```text
mom-{category}-{size}
mom-expert-{domain}-{size}
```

### Four Categories

1. **Intelligent Routing**: `mom-brain-{flash|pro|max}`
2. **Similarity Search**: `mom-similarity-{flash}`
3. **Prompt Guardian**: `mom-{jailbreak|pii}-{flash}`
4. **SLM Experts**: `mom-expert-{domain}-{flash|pro}`

### Three Size Variants

- **flash**: ModernBERT-based (for brain/similarity/guardian) or Qwen 0.6B (for experts) — fastest, sub-10ms latency
- **pro**: Qwen 0.6B (for brain) or Qwen 1.7B (for experts) — balanced performance with reasoning
- **max**: Qwen 1.7B (for brain) — maximum accuracy and capabilities

### Architecture Summary

- **Intelligent Routing**: Flash (ModernBERT) + Pro/Max (Qwen 0.6B/1.7B)
- **Similarity Search**: Flash (ModernBERT)
- **Prompt Guardian**: Flash (ModernBERT)
- **SLM Experts**: Flash/Pro (Qwen 0.6B/1.7B)

## Get Started

All MoM models are available on [Hugging Face](https://huggingface.co/LLM-Semantic-Router).

**Resources**:

- [GitHub](https://github.com/vllm-project/semantic-router)
- [Documentation](https://vllm-semantic-router.com)
- [Quick Start Guide](https://vllm-semantic-router.com/docs/installation)

---

**vLLM-SR · Route with intent. Think with reason.**
