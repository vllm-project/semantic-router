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
# Define embedding signals
signals:
  embeddings:
    - name: "technical_support"
      threshold: 0.75
      candidates:
        - "how to configure the system"
        - "installation guide"
        - "troubleshooting steps"
        - "error message explanation"
      aggregation_method: "max"

    - name: "product_inquiry"
      threshold: 0.70
      candidates:
        - "product features and specifications"
        - "pricing information"
        - "availability and stock"
      aggregation_method: "avg"

    - name: "account_management"
      threshold: 0.72
      candidates:
        - "password reset"
        - "account settings"
        - "subscription management"
      aggregation_method: "max"

# Define decisions using embedding signals
decisions:
  - name: technical_support
    description: "Route technical support queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "technical_support"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a technical support specialist with deep knowledge of system configuration and troubleshooting."

  - name: product_inquiry
    description: "Route product inquiry queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "product_inquiry"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a product specialist with comprehensive knowledge of features, pricing, and availability."
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.85

  - name: account_management
    description: "Route account management queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "account_management"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are an account management specialist. Handle user account queries with care and security."
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
| Qwen3 | 1024 | 30-50ms | Highest | ~2.4GB |
| Gemma | 768 | 10-20ms | High | ~1.2GB |
| Gemma | 512 | 8-15ms | Medium | ~1.2GB |
| Gemma | 256 | 5-10ms | Lower | ~1.2GB |

## Reference

See [embedding.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/embedding.yaml) for complete configuration.
