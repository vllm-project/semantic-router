# What is MoM Model Family?

The **MoM (Mixture of Models) Model Family** is a curated collection of specialized, lightweight models designed for intelligent routing, content safety, and semantic understanding. These models power the core capabilities of Semantic Router, enabling fast, accurate, and privacy-preserving AI operations.

## Overview

The MoM family consists of purpose-built models that handle specific tasks in the routing pipeline:

- **Classification Models**: Domain detection, PII identification, jailbreak detection
- **Embedding Models**: Semantic similarity, caching, retrieval
- **Safety Models**: Hallucination detection, content moderation
- **Feedback Models**: User intent understanding, conversation analysis

All MoM models are:

- **Lightweight**: 33M-600M parameters for fast inference
- **Specialized**: Fine-tuned for specific routing tasks
- **Efficient**: Many use LoRA adapters for minimal memory footprint
- **Open Source**: Available on HuggingFace for transparency and customization

## Model Categories

### 1. Classification Models

#### Domain/Intent Classifier

- **Model ID**: `models/mmbert32k-intent-classifier-merged`
- **HuggingFace**: `llm-semantic-router/mmbert32k-intent-classifier-merged`
- **Purpose**: Classify user queries into 14 MMLU categories (math, science, history, etc.)
- **Architecture**: mmBERT-32K merged classifier (307M)
- **Use Case**: Route queries to domain-specific models or experts

#### PII Detector

- **Model ID**: `models/mmbert32k-pii-detector-merged`
- **HuggingFace**: `llm-semantic-router/mmbert32k-pii-detector-merged`
- **Purpose**: Detect 17 PII entity types across 35 BIO labels
- **Architecture**: mmBERT-32K merged token classifier (307M)
- **Use Case**: Privacy protection, compliance, data masking

#### Jailbreak Detector

- **Model ID**: `models/mmbert32k-jailbreak-detector-merged`
- **HuggingFace**: `llm-semantic-router/mmbert32k-jailbreak-detector-merged`
- **Purpose**: Detect prompt injection and jailbreak attempts
- **Architecture**: mmBERT-32K merged classifier (307M)
- **Use Case**: Content safety, prompt security

#### Feedback Detector

- **Model ID**: `models/mmbert32k-feedback-detector-merged`
- **HuggingFace**: `llm-semantic-router/mmbert32k-feedback-detector-merged`
- **Purpose**: Classify user feedback into 4 types (satisfied, need clarification, wrong answer, want different)
- **Architecture**: mmBERT-32K merged classifier (307M)
- **Use Case**: Adaptive routing, conversation improvement

### 2. Embedding Models

#### Embedding Pro (High Quality)

- **Model ID**: `models/mom-embedding-pro`
- **HuggingFace**: `Qwen/Qwen3-Embedding-0.6B`
- **Purpose**: High-quality embeddings with 32K context support
- **Architecture**: Qwen3 (600M parameters)
- **Embedding Dimension**: 1024
- **Use Case**: Long-context semantic search, high-accuracy caching

#### Embedding Flash (Balanced)

- **Model ID**: `models/mom-embedding-flash`
- **HuggingFace**: `google/embeddinggemma-300m`
- **Purpose**: Fast embeddings with Matryoshka support
- **Architecture**: Gemma (300M parameters)
- **Embedding Dimension**: 768 (supports 512/256/128 via Matryoshka)
- **Use Case**: Balanced speed/quality, multilingual support

#### Embedding Ultra (Default)

- **Model ID**: `models/mom-embedding-ultra`
- **HuggingFace**: `llm-semantic-router/mmbert-embed-32k-2d-matryoshka`
- **Purpose**: Long-context multilingual semantic similarity with 2D Matryoshka support
- **Architecture**: mmBERT 2D Matryoshka (307M parameters)
- **Embedding Dimension**: 768 (supports lower dimensions via Matryoshka)
- **Use Case**: Default semantic caching, retrieval, and tools similarity

### 3. Hallucination Detection Models

#### Halugate Sentinel

- **Model ID**: `models/mom-halugate-sentinel`
- **HuggingFace**: `LLM-Semantic-Router/halugate-sentinel`
- **Purpose**: First-stage hallucination screening
- **Architecture**: BERT-base (110M)
- **Use Case**: Fast hallucination detection, pre-filtering

#### Halugate Detector

- **Model ID**: `models/mom-halugate-detector`
- **HuggingFace**: `KRLabsOrg/lettucedect-base-modernbert-en-v1`
- **Purpose**: Accurate hallucination verification
- **Architecture**: ModernBERT-base (149M)
- **Context Length**: 8192 tokens
- **Use Case**: Factual accuracy verification, grounding check

#### Halugate Explainer

- **Model ID**: `models/mom-halugate-explainer`
- **HuggingFace**: `tasksource/ModernBERT-base-nli`
- **Purpose**: Explain hallucination reasoning via NLI
- **Architecture**: ModernBERT-base (149M)
- **Classes**: 3 (entailment/neutral/contradiction)
- **Use Case**: Explainable AI, hallucination analysis

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Domain routing | mmbert32k-intent-classifier-merged | 14 MMLU categories, 32K context |
| Privacy protection | mmbert32k-pii-detector-merged | 17 entity types, 35 BIO labels, 32K context |
| Content safety | mmbert32k-jailbreak-detector-merged | Prompt injection detection with merged mmBERT |
| Semantic caching | mom-embedding-ultra | Default 32K multilingual embeddings |
| Long-context search | mom-embedding-pro | 32K context, 1024-dim |
| Hallucination check | mom-halugate-detector | ModernBERT, 8K context |
| User feedback | mmbert32k-feedback-detector-merged | 4 feedback types, merged mmBERT |

### By Performance Requirements

| Requirement | Model Tier | Examples |
|-------------|-----------|----------|
| Ultra-fast (&lt;10ms) | Light | mom-embedding-flash, mmbert32k-jailbreak-detector-merged |
| Balanced (10-50ms) | Default | mom-embedding-ultra, mmbert32k-intent-classifier-merged |
| High-quality (50-200ms) | Pro | mom-embedding-pro, mom-halugate-detector |

## Configuration

### Using MoM Models in Router

MoM models are configured through the canonical `global.model_catalog` block, with module-level settings living under `global.model_catalog.modules`:

```yaml
global:
  model_catalog:
    system:
      domain_classifier: "models/mmbert32k-intent-classifier-merged"
      pii_classifier: "models/mmbert32k-pii-detector-merged"
      prompt_guard: "models/mmbert32k-jailbreak-detector-merged"
    modules:
      classifier:
        domain:
          model_ref: "domain_classifier"
          threshold: 0.6
          use_cpu: true
        pii:
          model_ref: "pii_classifier"
          threshold: 0.9
          use_cpu: true
      prompt_guard:
        model_ref: "prompt_guard"
        threshold: 0.7
        use_cpu: true
```

### Custom System Bindings

Override the built-in system-model bindings in your `config.yaml`:

```yaml
global:
  model_catalog:
    system:
      domain_classifier: "models/your-domain-classifier"
      pii_classifier: "models/your-pii-classifier"
      prompt_guard: "models/your-prompt-guard"
```

## Model Architecture

### LoRA-Based Models

Many MoM models use LoRA (Low-Rank Adaptation) for efficiency:

- **Base Model**: BERT-base-uncased (110M parameters)
- **LoRA Adapters**: &lt;1M parameters per task
- **Memory Footprint**: ~440MB base + ~4MB per adapter
- **Inference Speed**: Same as base model (~10-20ms on CPU)

### ModernBERT Models

Newer models use ModernBERT for better performance:

- **Architecture**: ModernBERT-base (149M parameters)
- **Context Length**: 8192 tokens (vs 512 for BERT)
- **Performance**: Better accuracy on long-context tasks
- **Use Cases**: Hallucination detection, feedback classification

## Next Steps

- **[Signal-Driven Decisions](./signal-driven-decisions)** - Learn how MoM models power routing decisions
- **[Domain](../tutorials/signal/learned/domain)** - Use mmbert32k-intent-classifier-merged for routing
- **[PII](../tutorials/signal/learned/pii)** - Configure mmbert32k-pii-detector-merged
- **[RAG](../tutorials/plugin/rag)** - Use MoM embedding models for route-local retrieval
