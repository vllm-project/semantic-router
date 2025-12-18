# Embedding Based Routing

This guide shows you how to route requests using semantic similarity with embedding models. Embedding routing matches user queries to predefined categories based on meaning rather than exact keywords, making it ideal for handling diverse phrasings and rapidly evolving categories.

## Key Advantages

- **Scalable**: Handles unlimited categories without retraining models
- **Fast**: 10-50ms inference with efficient embedding models (Qwen3, Gemma)
- **Flexible**: Add/remove categories by updating keyword lists, no model retraining
- **Semantic**: Captures meaning beyond exact keyword matching

## What Problem Does It Solve?

Keyword matching fails when users phrase questions differently. Embedding routing solves:

- **Paraphrase handling**: "How to install?" matches "installation guide" without exact words
- **Intent detection**: Routes based on semantic meaning, not surface patterns
- **Fuzzy matching**: Handles typos, abbreviations, informal language
- **Dynamic categories**: Add new categories without retraining classification models
- **Multilingual support**: Embeddings capture cross-lingual semantics

## When to Use

- **Customer support** with diverse query phrasings
- **Product inquiries** where users ask the same thing many different ways
- **Technical support** needing semantic understanding of error descriptions
- **Rapidly evolving categories** where you need to add/update categories frequently
- **Moderate latency tolerance** (10-50ms acceptable for better semantic accuracy)

## Configuration

Add embedding rules to your `config.yaml`:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"

embedding_models:
  qwen3_model_path: "models/Qwen3-Embedding-0.6B"
  gemma_model_path: "models/embeddinggemma-300m"
  use_cpu: true

embedding_rules:
  - category: "technical_support"
    threshold: 0.75
    keywords:
      - "how to configure the system"
      - "installation guide"
      - "troubleshooting steps"
      - "error message explanation"
    aggregation_method: "max"
    model: "auto"
    dimension: 768
    quality_priority: 0.7
    latency_priority: 0.3
  
  - category: "product_inquiry"
    threshold: 0.70
    keywords:
      - "product features and specifications"
      - "pricing information"
      - "availability and stock"
    aggregation_method: "avg"
    model: "gemma"
    dimension: 768
  
  - category: "account_management"
    threshold: 0.72
    keywords:
      - "password reset"
      - "account settings"
      - "subscription management"
    aggregation_method: "max"
    model: "qwen3"
    dimension: 1024

categories:
  - name: technical_support
    system_prompt: "You are a technical support specialist."
    model_scores:
      - model: qwen3
        score: 0.9
        use_reasoning: true
    jailbreak_enabled: true
    pii_enabled: true
  
  - name: product_inquiry
    system_prompt: "You are a product specialist."
    model_scores:
      - model: qwen3
        score: 0.85
        use_reasoning: false
```

## Embedding Models

- **qwen3**: High quality, 1024-dim, 32K context
- **gemma**: Balanced, 768-dim, 8K context, Matryoshka support (128/256/512/768)
- **auto**: Automatically selects based on quality/latency priorities

## Aggregation Methods

- **max**: Uses highest similarity score
- **avg**: Uses average similarity across keywords
- **any**: Matches if any keyword exceeds threshold

## Example Requests

```bash
# Technical support query
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "How do I troubleshoot connection errors?"}]
  }'

# Product inquiry
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "What are the pricing options?"}]
  }'
```

## Real-World Use Cases

### 1. Customer Support (Scalable Categories)

**Problem**: Need to add new support categories weekly without retraining models
**Solution**: Add new categories by updating keyword lists, embeddings handle semantic matching
**Impact**: Deploy new categories in minutes vs weeks for model retraining

### 2. E-commerce Support (Fast Semantic Matching)

**Problem**: "Where's my order?" vs "track package" vs "shipping status" all mean the same
**Solution**: Gemma embeddings (10-20ms) route all variations to order tracking category
**Impact**: 95% accuracy with 10-20ms latency, handles 5K+ queries/sec

### 3. SaaS Product Inquiries (Flexible Routing)

**Problem**: Users ask about pricing in 100+ different ways
**Solution**: Semantic similarity matches all variations to "pricing information" keywords
**Impact**: Single category handles all pricing queries without explicit rules

### 4. Startup Iteration (Rapid Category Updates)

**Problem**: Product evolves rapidly, need to adjust categories daily
**Solution**: Update embedding keywords in config, no model retraining required
**Impact**: Category updates in seconds vs days for fine-tuning

### 5. Multilingual Platform (Semantic Understanding)

**Problem**: Same question in English, Spanish, Chinese needs same routing
**Solution**: Embeddings capture cross-lingual semantics automatically
**Impact**: Single category definition works across languages

## Model Selection Strategy

### Auto Mode (Recommended)

```yaml
model: "auto"
quality_priority: 0.7  # Favor accuracy
latency_priority: 0.3  # Accept some latency
```

- Automatically selects Qwen3 (high quality) or Gemma (fast) based on priorities
- Balances accuracy vs speed per request

### Qwen3 (High Quality)

```yaml
model: "qwen3"
dimension: 1024
```

- Best for: Complex queries, subtle distinctions, high-value interactions
- Latency: ~30-50ms per query
- Use case: Account management, financial queries

### Gemma (Fast)

```yaml
model: "gemma"
dimension: 768  # or 512, 256, 128 for Matryoshka
```

- Best for: High-throughput, simple categorization, cost-sensitive
- Latency: ~10-20ms per query
- Use case: Product inquiries, general support

## Performance Characteristics

| Model | Dimension | Latency | Accuracy | Memory |
|-------|-----------|---------|----------|--------|
| Qwen3 | 1024 | 30-50ms | Highest | 600MB |
| Gemma | 768 | 10-20ms | High | 300MB |
| Gemma | 512 | 8-15ms | Medium | 300MB |
| Gemma | 256 | 5-10ms | Lower | 300MB |

## Reference

See [embedding.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/embedding.yaml) for complete configuration.
