---
sidebar_position: 2
---

# What is Semantic Router?

**Semantic Router** is an intelligent routing layer that dynamically selects the most suitable language model for each query based on multiple signals extracted from the request.

## The Problem

Traditional LLM deployments use a single model for all tasks:

```text
User Query → Single LLM → Response
```

**Problems**:

- High cost for simple queries
- Suboptimal performance for specialized tasks
- No security or compliance controls
- Poor resource utilization

## The Solution

Semantic Router uses **signal-driven decision making** to route queries intelligently:

```text
User Query → Signal Extraction → Decision Engine → Best Model → Response
```

**Benefits**:

- Cost-effective routing (use smaller models for simple tasks)
- Better quality (use specialized models for their strengths)
- Built-in security (jailbreak detection, PII filtering)
- Flexible and extensible (plugin architecture)

## How It Works

### 1. Signal Extraction

The router extracts multiple types of signals from each request:

| Signal Type | What It Detects | Example |
|------------|----------------|---------|
| **keyword** | Specific terms and patterns | "calculate", "prove", "debug" |
| **embedding** | Semantic meaning | Math intent, code intent, creative intent |
| **domain** | Knowledge domain | Mathematics, computer science, history |
| **fact_check** | Need for verification | Factual claims, medical advice |
| **user_feedback** | User satisfaction | "That's wrong", "try again" |
| **preference** | Route preference | Complex intent matching |

### 2. Decision Making

Signals are combined using logical rules to make routing decisions:

```yaml
decisions:
  - name: math_routing
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "domain"
          name: "mathematics"
    modelRefs:
      - model: qwen-math
        weight: 1.0
```

**How it works**: If the query contains math keywords **AND** is classified as mathematics domain, route to the math model.

### 3. Model Selection

Based on the decision, the router selects the best model:

- **Math queries** → Math-specialized model (e.g., Qwen-Math)
- **Code queries** → Code-specialized model (e.g., DeepSeek-Coder)
- **Creative queries** → Creative model (e.g., Claude)
- **Simple queries** → Lightweight model (e.g., Llama-3-8B)

### 4. Plugin Chain

Before and after model execution, plugins process the request/response:

```yaml
plugins:
  - type: "semantic-cache"    # Check cache first
  - type: "jailbreak"         # Detect adversarial prompts
  - type: "pii"               # Filter sensitive data
  - type: "system_prompt"     # Add context
  - type: "hallucination"     # Verify facts
```

## Key Concepts

### Mixture of Models (MoM)

Unlike Mixture of Experts (MoE) which operates within a single model, Mixture of Models operates at the **system level**:

| Aspect | Mixture of Experts (MoE) | Mixture of Models (MoM) |
|--------|-------------------------|------------------------|
| **Scope** | Within a single model | Across multiple models |
| **Routing** | Internal gating network | External semantic router |
| **Models** | Shared architecture | Independent models |
| **Flexibility** | Fixed at training time | Dynamic at runtime |
| **Use Case** | Model efficiency | System intelligence |

### Signal-Driven Decisions

Traditional routing uses simple rules:

```yaml
# Traditional: Simple keyword matching
if "math" in query:
    route_to_math_model()
```

Signal-driven routing uses multiple signals:

```yaml
# Signal-driven: Multiple signals combined
if (has_math_keywords AND is_math_domain) OR has_high_math_embedding:
    route_to_math_model()
```

**Benefits**:

- More accurate routing
- Handles edge cases better
- Adapts to context
- Reduces false positives

## Real-World Example

**User Query**: "Prove that the square root of 2 is irrational"

**Signal Extraction**:

- keyword: ["prove", "square root", "irrational"] ✓
- embedding: 0.89 similarity to math queries ✓
- domain: "mathematics" ✓

**Decision**: Route to `qwen-math` (all math signals agree)

**Plugins Applied**:

- semantic-cache: Cache miss, proceed
- jailbreak: No adversarial patterns
- system_prompt: Added "Provide rigorous mathematical proof"
- hallucination: Enabled for verification

**Result**: High-quality mathematical proof from specialized model

## Next Steps

- [What is Collective Intelligence?](collective-intelligence.md) - How signals create system intelligence
- [What is Signal-Driven Decision?](signal-driven-decisions.md) - Deep dive into the decision engine
- [Configuration Guide](../installation/configuration.md) - Set up your semantic router
