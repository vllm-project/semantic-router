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
User Query → Signal Extraction → Projection Coordination → Decision Engine → Plugins + Model Dispatch → Response
```

**Benefits**:

- Cost-effective routing (use smaller models for simple tasks)
- Better quality (use specialized models for their strengths)
- Built-in security (jailbreak detection, PII filtering)
- Flexible and extensible (projection + plugin architecture)

## How It Works

### 1. Signal Extraction

The router extracts 14 maintained signal families from each request:

| Signal family group | Families                                                                                                         | Example role                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Heuristic**       | `authz`, `context`, `keyword`, `language`, `structure`                                                           | Cheap policy, request-shape, and locale gating       |
| **Learned**         | `complexity`, `domain`, `embedding`, `modality`, `fact-check`, `jailbreak`, `pii`, `preference`, `user-feedback` | Semantic, safety, and response-quality understanding |

### 2. Projection Coordination

Projections coordinate raw signal matches into reusable routing facts:

```yaml
routing:
  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: complexity
            name: hard
            weight: 0.4
    mappings:
      - name: difficulty_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: balance_reasoning
            gte: 0.6
```

### 3. Decision Making

Signals and projection outputs are combined using logical rules to make routing
decisions:

```yaml
decisions:
  - name: math_routing
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - type: "projection"
          name: "balance_reasoning"
    modelRefs:
      - model: qwen-math
        weight: 1.0
```

**How it works**: If the query is classified as mathematics **and** the
projection layer marks it as reasoning-heavy, route to the math model.

### 4. Model Selection

Based on the decision, the router selects the best model:

- **Math queries** → Math-specialized model (e.g., Qwen-Math)
- **Code queries** → Code-specialized model (e.g., DeepSeek-Coder)
- **Creative queries** → Creative model (e.g., Claude)
- **Simple queries** → Lightweight model (e.g., Llama-3-8B)

### 5. Plugin Chain

Before and after model execution, plugins process the request/response:

```yaml
routing:
  decisions:
    - name: "guarded-route"
      plugins:
        - type: "semantic-cache" # Check cache first
        - type: "jailbreak" # Detect adversarial prompts
        - type: "pii" # Filter sensitive data
        - type: "system_prompt" # Add context
        - type: "hallucination" # Verify facts
```

## Key Concepts

### Mixture of Models (MoM)

Unlike Mixture of Experts (MoE) which operates within a single model, Mixture of Models operates at the **system level**:

| Aspect          | Mixture of Experts (MoE) | Mixture of Models (MoM)  |
| --------------- | ------------------------ | ------------------------ |
| **Scope**       | Within a single model    | Across multiple models   |
| **Routing**     | Internal gating network  | External semantic router |
| **Models**      | Shared architecture      | Independent models       |
| **Flexibility** | Fixed at training time   | Dynamic at runtime       |
| **Use Case**    | Model efficiency         | System intelligence      |

### Signal-Driven Decisions

Traditional routing uses simple rules:

```yaml
# Traditional: Simple keyword matching
if "math" in query: route_to_math_model()
```

Signal-driven routing uses multiple signals:

```yaml
# Signal-driven: Multiple signals combined
if (has_math_keywords AND is_math_domain) OR has_high_math_embedding: route_to_math_model()
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

- [What is Collective Intelligence?](collective-intelligence) - How signals create system intelligence
- [What is Signal-Driven Decision?](signal-driven-decisions) - Deep dive into the decision engine
- [Configuration Guide](../installation/configuration) - Set up your semantic router
