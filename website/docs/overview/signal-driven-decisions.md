---
sidebar_position: 4
---

# What is Signal-Driven Decision?

**Signal-Driven Decision** is the core architecture that enables intelligent routing by extracting multiple signals from requests and combining them to make better routing decisions.

## The Core Idea

Traditional routing uses a single signal:

```yaml
# Traditional: Single classification model
if classifier(query) == "math":
    route_to_math_model()
```

Signal-driven routing uses multiple signals:

```yaml
# Signal-driven: Multiple signals combined
if (keyword_match AND domain_match) OR high_embedding_similarity:
    route_to_math_model()
```

**Why this matters**: Multiple signals voting together make more accurate decisions than any single signal.

## The 10 Signal Types

### 1. Keyword Signals

- **What**: Fast pattern matching with AND/OR operators
- **Latency**: Less than 1ms
- **Use Case**: Deterministic routing, compliance, security

```yaml
signals:
  keywords:
    - name: "math_keywords"
      operator: "OR"
      keywords: ["calculate", "equation", "solve", "derivative"]
```

**Example**: "Calculate the derivative of x^2" → Matches "calculate" and "derivative"

### 2. Embedding Signals

- **What**: Semantic similarity using embeddings
- **Latency**: 10-50ms
- **Use Case**: Intent detection, paraphrase handling

```yaml
signals:
  embeddings:
    - name: "code_debug"
      threshold: 0.70
      candidates:
        - "My code isn't working, how do I fix it?"
        - "Help me debug this function"
```

**Example**: "Need help debugging this function" → 0.78 similarity → Match!

### 3. Domain Signals

- **What**: MMLU domain classification (14 categories)
- **Latency**: 50-100ms
- **Use Case**: Academic and professional domain routing

```yaml
signals:
  domains:
    - name: "mathematics"
      mmlu_categories: ["abstract_algebra", "college_mathematics"]
```

**Example**: "Prove that the square root of 2 is irrational" → Mathematics domain

### 4. Fact Check Signals

- **What**: ML-based detection of queries needing fact verification
- **Latency**: 50-100ms
- **Use Case**: Healthcare, financial services, education

```yaml
signals:
  fact_checks:
    - name: "factual_queries"
      threshold: 0.75
```

**Example**: "What is the capital of France?" → Needs fact checking

### 5. User Feedback Signals

- **What**: Classification of user feedback and corrections
- **Latency**: 50-100ms
- **Use Case**: Customer support, adaptive learning

```yaml
signals:
  user_feedbacks:
    - name: "negative_feedback"
      feedback_types: ["correction", "dissatisfaction"]
```

**Example**: "That's wrong, try again" → Negative feedback detected

### 6. Preference Signals

- **What**: LLM-based route preference matching
- **Latency**: 200-500ms
- **Use Case**: Complex intent analysis

```yaml
signals:
  preferences:
    - name: "creative_writing"
      llm_endpoint: "http://localhost:8000/v1"
      model: "gpt-4"
      routes:
        - name: "creative"
          description: "Creative writing, storytelling, poetry"
```

**Example**: "Write a story about dragons" → Creative route preferred

### 7. Language Signals

- **What**: Multi-language detection (100+ languages)
- **Latency**: Less than 1ms
- **Use Case**: Route queries to language-specific models or apply language-specific policies

```yaml
signals:
  language:
    - name: "en"
      description: "English language queries"
    - name: "es"
      description: "Spanish language queries"
    - name: "zh"
      description: "Chinese language queries"
    - name: "ru"
      description: "Russian language queries"
```

- **Example 1**: "Hola, ¿cómo estás?" → Spanish (es) → Spanish model
- **Example 2**: "你好，世界" → Chinese (zh) → Chinese model

### 8. Latency Signals - Percentile-based Routing

**What**: Model latency evaluation using TPOT (Time Per Output Token) and TTFT (Time To First Token) percentiles
**Latency**: Typically 2-5ms for 10 models (runs asynchronously) - percentile calculation with O(n log n) complexity where n = observations per model (typically 10-100, max 1000)
**Use Case**: Route latency-sensitive queries to faster models based on adaptive percentile thresholds

```yaml
signals:
  latency:
    - name: "low_latency_comprehensive"
      tpot_percentile: 10  # 10th percentile for TPOT (top 10% fastest token generation)
      ttft_percentile: 10  # 10th percentile for TTFT (top 10% fastest first token)
      description: "For real-time applications - fast start and fast generation"
    - name: "balanced_latency"
      tpot_percentile: 50  # Median TPOT
      ttft_percentile: 10  # Top 10% TTFT (prioritize fast start)
      description: "Prioritize fast start, accept moderate generation speed"
```

**Example**: Real-time chat query → low_latency_comprehensive signal → Route to model meeting both TPOT and TTFT percentile thresholds

**How it works**:

- TPOT and TTFT are automatically tracked from each response
- Percentile-based thresholds adapt to each model's actual performance distribution
- Works with any number of observations: uses average for 1-2 observations, percentile calculation for 3+
- When both TPOT and TTFT percentiles are set, model must meet BOTH thresholds (AND logic)
- **Recommendation**: Use both TPOT and TTFT percentiles for comprehensive latency evaluation

### 9. Context Signals

- **What**: Token-count based routing for short/long request handling
- **Latency**: 1ms (calculated during processing)
- **Use Case**: Route long-context requests to models with larger context windows
- **Metrics**: Tracks input token counts with `llm_context_token_count` histogram

```yaml
signals:
  context_rules:
    - name: "low_token_count"
      min_tokens: "0"
      max_tokens: "1K"
      description: "Short requests"
    - name: "high_token_count"
      min_tokens: "1K"
      max_tokens: "128K"
      description: "Long requests requiring large context window"
```

**Example**: A request with 5,000 tokens → Matches "high_token_count" → Routes to `claude-3-opus`

### 10. Complexity Signals

- **What**: Embedding-based query complexity classification (hard/easy/medium)
- **Latency**: 50-100ms (embedding computation)
- **Use Case**: Route complex queries to powerful models, simple queries to efficient models
- **Logic**: Two-step classification:
  1. Find best matching rule by comparing query to rule descriptions
  2. Classify difficulty within that rule using hard/easy candidate embeddings

```yaml
signals:
  complexity:
    - name: "code_complexity"
      threshold: 0.1
      description: "Detects code complexity level"
      hard:
        candidates:
          - "design distributed system"
          - "implement consensus algorithm"
          - "optimize for scale"
      easy:
        candidates:
          - "print hello world"
          - "loop through array"
          - "read file"
```

**Example**: "How do I implement a distributed consensus algorithm?" → Matches "code_complexity" rule → High similarity to hard candidates → Returns "code_complexity:hard"

**How it works**:

1. Query embedding is compared to each rule's description
2. Best matching rule is selected (highest description similarity)
3. Within that rule, query is compared to hard and easy candidates
4. Difficulty signal = max_hard_similarity - max_easy_similarity
5. If signal > threshold: "hard", if signal < -threshold: "easy", else: "medium"

## How Signals Combine

### AND Operator - All Must Match

```yaml
decisions:
  - name: "advanced_math"
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "mathematics"
```

- **Logic**: Route to advanced_math **only if** both keyword AND domain match
- **Use Case**: High-confidence routing (reduce false positives)

### OR Operator - Any Can Match

```yaml
decisions:
  - name: "code_help"
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "code_keywords"
        - type: "embedding"
          name: "code_debug"
```

- **Logic**: Route to code_help **if** keyword OR embedding matches
- **Use Case**: Broad coverage (reduce false negatives)

### Nested Logic - Complex Rules

```yaml
decisions:
  - name: "verified_math"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - operator: "OR"
          conditions:
            - type: "keyword"
              name: "proof_keywords"
            - type: "fact_check"
              name: "factual_queries"
```

- **Logic**: Route if (mathematics domain) AND (proof keywords OR needs fact checking)
- **Use Case**: Complex routing scenarios

## Real-World Example

### User Query

```text
"Prove that the square root of 2 is irrational"
```

### Signal Extraction

```yaml
signals_detected:
  keyword: true          # "prove", "square root", "irrational"
  embedding: 0.89        # High similarity to math queries
  domain: "mathematics"  # MMLU classification
  fact_check: true       # Proof requires verification
```

### Decision Process

```yaml
decision: "advanced_math"
reason: "All math signals agree (keyword + embedding + domain + fact_check)"
confidence: 0.95
selected_model: "qwen-math"
```

### Why This Works

- **Multiple signals agree**: High confidence
- **Fact checking enabled**: Quality assurance
- **Specialized model**: Best for mathematical proofs

## Next Steps

- [Configuration Guide](../installation/configuration.md) - Configure signals and decisions
- [Keyword Routing Tutorial](../tutorials/intelligent-route/keyword-routing.md) - Learn keyword signals
- [Embedding Routing Tutorial](../tutorials/intelligent-route/embedding-routing.md) - Learn embedding signals
- [Domain Routing Tutorial](../tutorials/intelligent-route/domain-routing.md) - Learn domain signals
- [Context Routing Tutorial](../tutorials/intelligent-route/context-routing.md) - Learn context signals
- [Complexity Routing Tutorial](../tutorials/intelligent-route/complexity-routing.md) - Learn complexity signals
