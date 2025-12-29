# Fact Check Signal Routing

This guide shows you how to route requests based on whether they require fact verification. The fact_check signal helps identify factual queries that need hallucination detection or fact-checking.

## Key Advantages

- **Automatic Detection**: ML-based detection of factual vs creative/code queries
- **Hallucination Prevention**: Route factual queries to models with verification
- **Resource Optimization**: Apply expensive fact-checking only when needed
- **Compliance**: Ensure factual accuracy for regulated industries

## What Problem Does It Solve?

Not all queries require fact verification:

- **Factual queries**: "What is the capital of France?" → Needs verification
- **Creative queries**: "Write a story about dragons" → No verification needed
- **Code queries**: "Write a Python function" → No verification needed

The fact_check signal automatically identifies which queries need fact verification, allowing you to:

1. Route factual queries to models with hallucination detection
2. Enable fact-checking plugins only for factual queries
3. Optimize costs by avoiding unnecessary verification

## Configuration

### Basic Configuration

Define fact check signals in your `config.yaml`:

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "Query contains factual claims that should be verified against context"

    - name: no_fact_check_needed
      description: "Query is creative, code-related, or opinion-based - no fact verification needed"
```

### Use in Decision Rules

```yaml
decisions:
  - name: factual_queries
    description: "Route factual queries with verification"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a factual information specialist. Provide accurate, verifiable information with sources when possible."
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.7
```

## Use Cases

### 1. Healthcare - Medical Information

**Problem**: Medical queries must be factually accurate to avoid harm

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "Query contains factual claims that should be verified"

  domains:
    - name: "health"
      description: "Medical and health queries"
      mmlu_categories: ["health"]

decisions:
  - name: verified_medical
    description: "Medical queries with fact verification"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a medical information specialist. Provide accurate, evidence-based health information."
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.8  # High threshold for medical
```

**Example Queries**:

- "What are the symptoms of diabetes?" → ✅ Routed with verification
- "Write a story about a doctor" → ❌ Creative, no verification

### 2. Financial Services - Investment Information

**Problem**: Financial advice must be accurate to comply with regulations

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "Query contains factual claims that should be verified"

  keywords:
    - name: "financial_keywords"
      operator: "OR"
      keywords: ["stock", "investment", "portfolio", "dividend"]
      case_sensitive: false

decisions:
  - name: verified_financial
    description: "Financial queries with verification"
    priority: 200
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "financial_keywords"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a financial information specialist. Provide accurate financial information with appropriate disclaimers."
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.8
```

**Example Queries**:

- "What is the current P/E ratio of Apple?" → ✅ Factual, verified
- "Explain investment strategies" → ❌ General advice, no verification

### 3. Education - Historical Facts

**Problem**: Educational content must be factually accurate

```yaml
signals:
  fact_check:
    - name: needs_fact_check
      description: "Query contains factual claims that should be verified"

  domains:
    - name: "history"
      description: "Historical queries"
      mmlu_categories: ["history"]

decisions:
  - name: verified_history
    description: "Historical queries with verification"
    priority: 150
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "history"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a history specialist. Provide accurate historical information with proper context."
      - type: "hallucination"
        configuration:
          enabled: true
          threshold: 0.7
```

**Example Queries**:

- "When did World War II end?" → ✅ Factual, verified
- "Write a historical fiction story" → ❌ Creative, no verification

## Performance Characteristics

| Aspect | Value |
|--------|-------|
| Latency | 20-50ms |
| Accuracy | 80-90% |
| False Positives | 5-10% (creative marked as factual) |
| False Negatives | 5-10% (factual marked as creative) |

## Best Practices

### 1. Combine with Domain Signals

Use both fact_check and domain signals for better accuracy:

```yaml
rules:
  operator: "AND"
  conditions:
    - type: "domain"
      name: "science"
    - type: "fact_check"
      name: "needs_verification"
```

### 2. Set Appropriate Priorities

Factual queries should have higher priority:

```yaml
decisions:
  - name: verified_factual
    priority: 100  # High priority
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_verification"
```

### 3. Enable Hallucination Detection

Always enable hallucination plugin for factual queries:

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      threshold: 0.7
```

### 4. Monitor False Positives/Negatives

Track queries that are misclassified:

```yaml
logging:
  level: debug
  fact_check: true
```

## Reference

See [Signal-Driven Decision Architecture](../../overview/signal-driven-decisions.md) for complete signal architecture.
