# Domain Based Routing

This guide shows you how to use fine-tuned classification models for intelligent routing based on academic and professional domains. Domain routing uses specialized models (ModernBERT, Qwen3-Embedding, EmbeddingGemma) with LoRA adapters to classify queries into categories like math, physics, law, business, and more.

## Key Advantages

- **Efficient**: Fine-tuned models with LoRA adapters provide fast inference (5-20ms) with high accuracy
- **Specialized**: Multiple model options (ModernBERT for English, Qwen3 for multilingual/long-context, Gemma for small footprint)
- **Multi-task**: LoRA enables running multiple classification tasks (domain + PII + jailbreak) with shared base model
- **Cost-effective**: Lower latency than LLM-based classification, no API costs

## What Problem Does It Solve?

Generic classification approaches struggle with domain-specific terminology and nuanced differences between academic/professional fields. Domain routing provides:

- **Accurate domain detection**: Fine-tuned models distinguish between math, physics, chemistry, law, business, etc.
- **Multi-task efficiency**: LoRA adapters enable simultaneous domain classification, PII detection, and jailbreak detection with one base model pass
- **Long-context support**: Qwen3-Embedding handles up to 32K tokens (vs ModernBERT's 8K limit)
- **Multilingual routing**: Qwen3 trained on 100+ languages, ModernBERT optimized for English
- **Resource optimization**: Expensive reasoning only enabled for domains that benefit (math, physics, chemistry)

## When to Use

- **Educational platforms** with diverse subject areas (STEM, humanities, social sciences)
- **Professional services** requiring domain expertise (legal, medical, financial)
- **Enterprise knowledge bases** spanning multiple departments
- **Research assistance** tools needing academic domain awareness
- **Multi-domain products** where classification accuracy is critical

## Configuration

Configure the domain classifier in your `config.yaml`:

```yaml
# Define domain signals
signals:
  domains:
    - name: "math"
      description: "Mathematics and quantitative reasoning"
      mmlu_categories: ["math"]

    - name: "physics"
      description: "Physics and physical sciences"
      mmlu_categories: ["physics"]

    - name: "computer_science"
      description: "Programming and computer science"
      mmlu_categories: ["computer_science"]

    - name: "business"
      description: "Business and management"
      mmlu_categories: ["business"]

    - name: "health"
      description: "Health and medical information"
      mmlu_categories: ["health"]

    - name: "law"
      description: "Legal and regulatory information"
      mmlu_categories: ["law"]

    - name: "other"
      description: "General queries"
      mmlu_categories: ["other"]

# Define decisions using domain signals
decisions:
  - name: math
    description: "Route mathematical queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a mathematics expert. Provide step-by-step solutions with clear explanations."

  - name: physics
    description: "Route physics queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "physics"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a physics expert with deep understanding of physical laws. Explain concepts clearly with mathematical derivations."

  - name: computer_science
    description: "Route computer science queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a computer science expert with knowledge of algorithms and data structures. Provide clear code examples."

  - name: business
    description: "Route business queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "business"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a senior business consultant and strategic advisor."

  - name: health
    description: "Route health queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a health and medical information expert."
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95

  - name: law
    description: "Route legal queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "law"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a knowledgeable legal expert."

  - name: general_route
    description: "Default fallback route for general queries"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.85
```

## Supported Domains

Academic: math, physics, chemistry, biology, computer science, engineering

Professional: business, law, economics, health, psychology

General: philosophy, history, other

## Features

- **PII Detection**: Automatically detects and handles sensitive information
- **Semantic Caching**: Cache similar queries for faster responses
- **Reasoning Control**: Enable/disable reasoning per domain
- **Custom Thresholds**: Adjust cache sensitivity per category

## Example Requests

```bash
# Math query (reasoning enabled)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}]
  }'

# Business query (reasoning disabled)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "What is a SWOT analysis?"}]
  }'

# Health query (high cache threshold)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "What are symptoms of diabetes?"}]
  }'
```

## Real-World Use Cases

### 1. Multi-Task Classification with LoRA (Efficient)

**Problem**: Need domain classification + PII detection + jailbreak detection on every request
**Solution**: LoRA adapters run all 3 tasks with one base model pass instead of 3 separate models
**Impact**: 3x faster than running 3 full models, &lt;1% parameter overhead per task

### 2. Long Document Analysis (Specialized - Qwen3)

**Problem**: Research papers and legal documents exceed 8K token limit of ModernBERT
**Solution**: Qwen3-Embedding supports up to 32K tokens without truncation
**Impact**: Accurate classification on full documents, no information loss from truncation

### 3. Multilingual Education Platform (Specialized - Qwen3)

**Problem**: Students ask questions in 100+ languages, ModernBERT limited to English
**Solution**: Qwen3-Embedding trained on 100+ languages handles multilingual routing
**Impact**: Single model serves global users, consistent quality across languages

### 4. Edge Deployment (Specialized - Gemma)

**Problem**: Mobile/IoT devices can't run large classification models
**Solution**: EmbeddingGemma-300M with Matryoshka embeddings (128-768 dims)
**Impact**: 5x smaller model, runs on edge devices with &lt;100MB memory

### 5. STEM Tutoring Platform (Efficient Reasoning Control)

**Problem**: Math/physics need reasoning, but history/literature don't
**Solution**: Domain classifier routes STEM → reasoning models, humanities → fast models
**Impact**: 2x better STEM accuracy, 60% cost savings on non-STEM queries

## Domain-Specific Optimizations

### STEM Domains (Reasoning Enabled)

```yaml
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # Step-by-step solutions
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a mathematics expert. Provide step-by-step solutions."

- name: physics
  description: "Route physics queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "physics"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # Derivations and proofs
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a physics expert. Explain concepts clearly with mathematical derivations."

- name: chemistry
  description: "Route chemistry queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "chemistry"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # Reaction mechanisms
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a chemistry expert. Explain reaction mechanisms clearly."
```

### Professional Domains (PII + Caching)

```yaml
decisions:
- name: health
  description: "Route health queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "health"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a health and medical information expert."
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.95  # Very strict

- name: law
  description: "Route legal queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "law"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a knowledgeable legal expert."
```

### General Domains (Fast + Cached)

```yaml
decisions:
- name: business
  description: "Route business queries"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "business"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false  # Fast responses
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "You are a senior business consultant and strategic advisor."

- name: general_route
  description: "Default fallback route for general queries"
  priority: 50
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.75  # Relaxed
```

## Performance Characteristics

| Domain | Reasoning | Cache Threshold | Avg Latency | Use Case |
|--------|-----------|-----------------|-------------|----------|
| Math | ✅ | 0.85 | 2-5s | Step-by-step solutions |
| Physics | ✅ | 0.85 | 2-5s | Derivations |
| Chemistry | ✅ | 0.85 | 2-5s | Mechanisms |
| Health | ❌ | 0.95 | 500ms | Safety-critical |
| Law | ❌ | 0.85 | 500ms | Compliance |
| Business | ❌ | 0.80 | 300ms | Fast insights |
| Other | ❌ | 0.75 | 200ms | General queries |

## Cost Optimization Strategy

1. **Reasoning Budget**: Enable only for STEM (30% of queries) → 60% cost reduction
2. **Caching Strategy**: High threshold for sensitive domains → 70% hit rate
3. **Model Selection**: Lower scores for low-value domains → cheaper models
4. **PII Detection**: Only for health/law → reduced processing overhead

## Reference

See [bert_classification.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/bert_classification.yaml) for complete configuration.
