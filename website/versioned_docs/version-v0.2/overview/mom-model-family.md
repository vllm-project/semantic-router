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

- **Model ID**: `models/mom-domain-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model`
- **Purpose**: Classify user queries into 14 MMLU categories (math, science, history, etc.)
- **Architecture**: BERT-base (110M) + LoRA adapters
- **Use Case**: Route queries to domain-specific models or experts

#### PII Detector

- **Model ID**: `models/mom-pii-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model`
- **Purpose**: Detect 35 types of personally identifiable information
- **Architecture**: BERT-base (110M) + LoRA adapters
- **Use Case**: Privacy protection, compliance, data masking

#### Jailbreak Detector

- **Model ID**: `models/mom-jailbreak-classifier`
- **HuggingFace**: `LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model`
- **Purpose**: Detect prompt injection and jailbreak attempts
- **Architecture**: BERT-base (110M) + LoRA adapters
- **Use Case**: Content safety, prompt security

#### Feedback Detector

- **Model ID**: `models/mom-feedback-detector`
- **HuggingFace**: `llm-semantic-router/feedback-detector`
- **Purpose**: Classify user feedback into 4 types (satisfied, need clarification, wrong answer, want different)
- **Architecture**: ModernBERT-base (149M)
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

#### Embedding Light (Fast)

- **Model ID**: `models/mom-embedding-light`
- **HuggingFace**: `sentence-transformers/all-MiniLM-L12-v2`
- **Purpose**: Lightweight semantic similarity
- **Architecture**: MiniLM (33M parameters)
- **Embedding Dimension**: 384
- **Use Case**: Fast semantic caching, low-latency retrieval

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
| Domain routing | mom-domain-classifier | 14 MMLU categories, LoRA efficient |
| Privacy protection | mom-pii-classifier | 35 PII types, token-level detection |
| Content safety | mom-jailbreak-classifier | Prompt injection detection |
| Semantic caching | mom-embedding-light | Fast, 384-dim, low latency |
| Long-context search | mom-embedding-pro | 32K context, 1024-dim |
| Hallucination check | mom-halugate-detector | ModernBERT, 8K context |
| User feedback | mom-feedback-detector | 4 feedback types, ModernBERT |

### By Performance Requirements

| Requirement | Model Tier | Examples |
|-------------|-----------|----------|
| Ultra-fast (&lt;10ms) | Light | mom-embedding-light, mom-jailbreak-classifier |
| Balanced (10-50ms) | Flash | mom-embedding-flash, mom-domain-classifier |
| High-quality (50-200ms) | Pro | mom-embedding-pro, mom-halugate-detector |

## Configuration

### Using MoM Models in Router

MoM models are pre-configured in `router-defaults.yaml`:

```yaml
# Domain classification
classifier:
  category_model:
    model_id: "models/mom-domain-classifier"
    threshold: 0.6
    use_cpu: true

# PII detection
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    threshold: 0.9
    use_cpu: true

# Jailbreak protection
prompt_guard:
  model_id: "models/mom-jailbreak-classifier"
  threshold: 0.7
  use_cpu: true
```

### Custom Model Registry

Override the default registry in your `config.yaml`:

```yaml
mom_registry:
  "models/mom-domain-classifier": "your-org/custom-domain-classifier"
  "models/mom-pii-classifier": "your-org/custom-pii-detector"
  "models/mom-embedding-pro": "your-org/custom-embeddings"
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

- **[Signal-Driven Decisions](./signal-driven-decisions.md)** - Learn how MoM models power routing decisions
- **[Domain Routing](../tutorials/intelligent-route/domain-routing.md)** - Use mom-domain-classifier for routing
- **[PII Detection](../tutorials/content-safety/pii-detection.md)** - Configure mom-pii-classifier
- **[Semantic Cache](../tutorials/semantic-cache/in-memory-cache.md)** - Use MoM embedding models
