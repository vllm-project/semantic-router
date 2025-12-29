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

## The 6 Signal Types

### 1. Keyword Signals

**What**: Fast pattern matching with AND/OR operators
**Latency**: Less than 1ms
**Use Case**: Deterministic routing, compliance, security

```yaml
signals:
  keywords:
    - name: "math_keywords"
      operator: "OR"
      keywords: ["calculate", "equation", "solve", "derivative"]
```

**Example**: "Calculate the derivative of x^2" → Matches "calculate" and "derivative"

### 2. Embedding Signals

**What**: Semantic similarity using embeddings
**Latency**: 10-50ms
**Use Case**: Intent detection, paraphrase handling

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

**What**: MMLU domain classification (14 categories)
**Latency**: 50-100ms
**Use Case**: Academic and professional domain routing

```yaml
signals:
  domains:
    - name: "mathematics"
      mmlu_categories: ["abstract_algebra", "college_mathematics"]
```

**Example**: "Prove that the square root of 2 is irrational" → Mathematics domain

### 4. Fact Check Signals

**What**: ML-based detection of queries needing fact verification
**Latency**: 50-100ms
**Use Case**: Healthcare, financial services, education

```yaml
signals:
  fact_checks:
    - name: "factual_queries"
      threshold: 0.75
```

**Example**: "What is the capital of France?" → Needs fact checking

### 5. User Feedback Signals

**What**: Classification of user feedback and corrections
**Latency**: 50-100ms
**Use Case**: Customer support, adaptive learning

```yaml
signals:
  user_feedbacks:
    - name: "negative_feedback"
      feedback_types: ["correction", "dissatisfaction"]
```

**Example**: "That's wrong, try again" → Negative feedback detected

### 6. Preference Signals

**What**: LLM-based route preference matching
**Latency**: 200-500ms
**Use Case**: Complex intent analysis

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

**Logic**: Route to advanced_math **only if** both keyword AND domain match

**Use Case**: High-confidence routing (reduce false positives)

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

**Logic**: Route to code_help **if** keyword OR embedding matches

**Use Case**: Broad coverage (reduce false negatives)

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

**Logic**: Route if (mathematics domain) AND (proof keywords OR needs fact checking)

**Use Case**: Complex routing scenarios

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
