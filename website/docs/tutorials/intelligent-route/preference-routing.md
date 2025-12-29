# Preference Signal Routing

This guide shows you how to route requests using LLM-based preference matching. The preference signal uses an external LLM to analyze complex intent and make nuanced routing decisions.

## Key Advantages

- **Complex Intent Analysis**: Use LLM reasoning for nuanced routing decisions
- **Flexible Logic**: Define routing preferences in natural language
- **High Accuracy**: 90-98% accuracy for complex intent detection
- **Extensible**: Add new preferences without retraining models

## What Problem Does It Solve?

Some routing decisions are too complex for simple pattern matching or classification:

- **Nuanced Intent**: "Explain the philosophical implications of quantum mechanics"
- **Multi-faceted Queries**: "Compare and contrast utilitarianism and deontology"
- **Context-dependent**: "What's the best approach for this problem?"

The preference signal uses an external LLM to analyze these complex queries and match them to routing preferences, allowing you to:

1. Handle complex intent that other signals miss
2. Make nuanced routing decisions based on LLM reasoning
3. Define routing logic in natural language
4. Adapt to new use cases without retraining

## Configuration

### Basic Configuration

```yaml
signals:
  preferences:
    - name: "complex_reasoning"
      description: "Requires deep reasoning and analysis"
      llm_endpoint: "http://localhost:11434"
      model: "llama3"  # Optional: specify model
      temperature: 0.1  # Optional: low temperature for consistency
```

### Use in Decision Rules

```yaml
decisions:
  - name: deep_reasoning
    description: "Route complex reasoning queries"
    priority: 20
    rules:
      operator: "OR"
      conditions:
        - type: "preference"
          name: "complex_reasoning"
    modelRefs:
      - model: reasoning-specialist
        weight: 1.0
        use_reasoning: true
```

## How It Works

### 1. Query Analysis

The external LLM analyzes the query:

```
Query: "Explain the philosophical implications of quantum mechanics"

LLM Analysis:
- Requires deep reasoning: YES
- Complexity level: HIGH
- Domain: Philosophy + Physics
- Reasoning type: Analytical, conceptual
```

### 2. Preference Matching

The LLM matches the query to defined preferences:

```yaml
preferences:
  - name: "complex_reasoning"
    description: "Requires deep reasoning and analysis"
    # LLM evaluates: Does this query require deep reasoning?
    # Result: YES (confidence: 0.95)
```

### 3. Routing Decision

Based on the match, the query is routed:

```
Preference matched: complex_reasoning (0.95)
Decision: deep_reasoning
Model: reasoning-specialist
```

## Use Cases

### 1. Academic Research - Complex Analysis

**Problem**: Research queries require deep reasoning and analysis

```yaml
signals:
  preferences:
    - name: "research_analysis"
      description: "Academic research requiring deep analysis and critical thinking"
      llm_endpoint: "http://localhost:11434"

  domains:
    - name: "philosophy"
      description: "Philosophical queries"
      mmlu_categories: ["philosophy", "formal_logic"]

decisions:
  - name: academic_research
    description: "Route academic research queries"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "philosophy"
        - type: "preference"
          name: "research_analysis"
    modelRefs:
      - model: research-specialist
        weight: 1.0
        use_reasoning: true
```

**Example Queries**:

- "Analyze the epistemological implications of Kant's Critique" → ✅ Complex analysis
- "What is philosophy?" → ❌ Simple definition

### 2. Business Strategy - Decision Making

**Problem**: Strategic queries need nuanced analysis

```yaml
signals:
  preferences:
    - name: "strategic_thinking"
      description: "Business strategy requiring multi-faceted analysis"
      llm_endpoint: "http://localhost:11434"

  keywords:
    - name: "business_keywords"
      operator: "OR"
      keywords: ["strategy", "market", "competition", "growth"]

decisions:
  - name: strategic_analysis
    description: "Route strategic business queries"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "business_keywords"
        - type: "preference"
          name: "strategic_thinking"
    modelRefs:
      - model: business-strategist
        weight: 1.0
```

**Example Queries**:

- "Analyze our competitive position and recommend growth strategies" → ✅ Strategic
- "What is our revenue?" → ❌ Simple query

### 3. Technical Architecture - Design Decisions

**Problem**: Architecture decisions require deep technical reasoning

```yaml
signals:
  preferences:
    - name: "architecture_design"
      description: "Technical architecture requiring design thinking and trade-off analysis"
      llm_endpoint: "http://localhost:11434"

  keywords:
    - name: "architecture_keywords"
      operator: "OR"
      keywords: ["architecture", "design", "scalability", "performance"]

decisions:
  - name: architecture_analysis
    description: "Route architecture design queries"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "architecture_keywords"
        - type: "preference"
          name: "architecture_design"
    modelRefs:
      - model: architecture-specialist
        weight: 1.0
```

**Example Queries**:

- "Design a scalable microservices architecture with trade-offs" → ✅ Design thinking
- "What is microservices?" → ❌ Simple definition

## Performance Characteristics

| Aspect | Value |
|--------|-------|
| Latency | 100-500ms (depends on LLM) |
| Accuracy | 90-98% |
| Cost | Higher (external LLM call) |
| Scalability | Limited by LLM endpoint |

## Best Practices

### 1. Use as Last Resort

Preference signals are expensive. Use other signals first:

```yaml
decisions:
  - name: simple_math
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "math_keywords"  # Fast, cheap
    
  - name: complex_reasoning
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "preference"
          name: "complex_reasoning"  # Slow, expensive
```

### 2. Combine with Other Signals

Use AND operator to reduce false positives:

```yaml
rules:
  operator: "AND"
  conditions:
    - type: "domain"
      name: "philosophy"  # Fast pre-filter
    - type: "preference"
      name: "complex_reasoning"  # Expensive verification
```

### 3. Cache LLM Responses

Enable caching to reduce latency and cost:

```yaml
preferences:
  - name: "complex_reasoning"
    description: "Requires deep reasoning"
    llm_endpoint: "http://localhost:11434"
    cache_enabled: true
    cache_ttl: 3600  # 1 hour
```

### 4. Set Appropriate Timeouts

Prevent slow LLM calls from blocking:

```yaml
preferences:
  - name: "complex_reasoning"
    description: "Requires deep reasoning"
    llm_endpoint: "http://localhost:11434"
    timeout: 2000  # 2 seconds
    fallback_on_timeout: false  # Don't match if timeout
```

### 5. Monitor Performance

Track LLM call latency and accuracy:

```yaml
logging:
  level: info
  preference_signals: true
  llm_latency: true
```

## Advanced Configuration

### Multiple LLM Endpoints

Use different LLMs for different preferences:

```yaml
signals:
  preferences:
    - name: "complex_reasoning"
      description: "Deep reasoning"
      llm_endpoint: "http://localhost:11434"
      model: "llama3-70b"  # Large model for complex reasoning
    
    - name: "simple_classification"
      description: "Simple intent classification"
      llm_endpoint: "http://localhost:11435"
      model: "llama3-8b"  # Small model for simple tasks
```

### Custom Prompts

Customize the LLM prompt for better accuracy:

```yaml
preferences:
  - name: "complex_reasoning"
    description: "Requires deep reasoning"
    llm_endpoint: "http://localhost:11434"
    prompt_template: |
      Analyze the following query and determine if it requires deep reasoning and analysis.
      Query: {query}
      Answer with YES or NO and explain why.
```

## Reference

See [Signal-Driven Decision Architecture](../../overview/signal-driven-decisions.md) for complete signal architecture.
