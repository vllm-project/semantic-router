---
sidebar_position: 3
---

# What is Collective Intelligence?

**Collective Intelligence** is the emergent intelligence that arises when multiple models, signals, and decision-making processes work together as a unified system.

## The Core Idea

Just as a team of specialists can solve problems better than any individual expert, a system of specialized LLMs can provide better results than any single model.

### Traditional Approach: Single Model

```
User Query → Single LLM → Response
```

**Limitations**:

- One model tries to be good at everything
- No specialization or optimization
- Same model for simple and complex tasks
- No learning from patterns

### Collective Intelligence Approach: System of Models

```
User Query → Signal Extraction → Projection Coordination → Decision Engine → Plugins + Model Dispatch → Response
              ↓                    ↓                         ↓                         ↓
        14 Signal Families   Partitions / Scores / Mappings  Boolean Policies     Specialized Models
```

**Benefits**:

- Each model focuses on what it does best
- System learns from patterns across all interactions
- Adaptive routing based on multiple signals
- Emergent intelligence from signal fusion

## How Collective Intelligence Emerges

### 1. Signal Diversity

Different signals capture different aspects of intelligence:

| Signal family group                                                                                                            | Intelligence aspect                                      |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **Heuristic** (`authz`, `context`, `keyword`, `language`, `structure`)                                                         | Fast request-shape, locale, and policy gating            |
| **Learned** (`complexity`, `domain`, `embedding`, `modality`, `fact-check`, `jailbreak`, `pii`, `preference`, `user-feedback`) | Semantic, safety, modality, and preference understanding |

**Collective benefit**: The combination of signals provides a richer understanding than any single signal.

### 2. Projection Coordination

Signals become more useful when the router coordinates them into reusable
intermediate facts:

```yaml
projections:
  partitions:
    - name: balance_domain_partition
      semantics: exclusive
      members: [mathematics, coding, creative]
      default: creative
  scores:
    - name: reasoning_pressure
      method: weighted_sum
      inputs:
        - type: complexity
          name: hard
          weight: 0.6
        - type: embedding
          name: math_intent
          weight: 0.4
  mappings:
    - name: reasoning_band
      source: reasoning_pressure
      method: threshold_bands
      outputs:
        - name: balance_reasoning
          gte: 0.5
```

**Collective benefit**: Projections turn many weak or competing signals into
named routing facts that multiple decisions can reuse.

### 3. Decision Fusion

Signals are combined using logical operators:

```yaml
# Example: Math routing with multiple signals
decisions:
  - name: advanced_math
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "mathematics"
        - type: "projection"
          name: "balance_reasoning"
```

**Collective benefit**: Multiple signals voting together make more accurate decisions than any single signal.

### 4. Model Specialization

Different models contribute their strengths:

```yaml
modelRefs:
  - model: qwen-math # Best at mathematical reasoning
    weight: 1.0
  - model: deepseek-coder # Best at code generation
    weight: 1.0
  - model: claude-creative # Best at creative writing
    weight: 1.0
```

**Collective benefit**: System-level intelligence emerges from routing to the right specialist.

### 5. Plugin Collaboration

Plugins work together to enhance responses:

```yaml
routing:
  decisions:
    - name: "protected-route"
      plugins:
        - type: "semantic-cache" # Speed optimization
        - type: "jailbreak" # Security layer
        - type: "pii" # Privacy protection
        - type: "system_prompt" # Context injection
        - type: "hallucination" # Quality assurance
```

**Collective benefit**: Multiple layers of processing create a more robust and secure system.

## Real-World Example

Let's see collective intelligence in action:

### User Query

```
"Prove that the square root of 2 is irrational"
```

### Signal Extraction

```yaml
signals_detected:
  keyword: ["prove", "square root", "irrational"] # Math keywords detected
  embedding: 0.89 # High similarity to math queries
  domain: "mathematics" # MMLU classification
  fact_check: true # Proof requires verification
```

### Projection Coordination

```yaml
projection_outputs:
  balance_domain_partition: "mathematics"
  balance_reasoning: true
```

### Decision Process

```yaml
decision_made: "advanced_math"
reason: "Math domain plus projection-driven reasoning pressure"
confidence: 0.95
```

### Model Selection

```yaml
selected_model: "qwen-math"
reason: "Specialized in mathematical proofs"
```

### Plugin Chain

```yaml
plugins_applied:
  - semantic-cache: "Cache miss, proceeding"
  - jailbreak: "No adversarial patterns detected"
  - system_prompt: "Added: 'Provide rigorous mathematical proof'"
  - hallucination: "Enabled for fact verification"
```

### Result

- **Accurate**: Routed to math specialist
- **Fast**: Checked cache first
- **Safe**: Verified no jailbreak attempts
- **High-quality**: Hallucination detection enabled

**This is collective intelligence**: No single component made the decision.
The intelligence emerged from the collaboration of signals, projections, rules,
models, and plugins.

## Benefits of Collective Intelligence

### 1. Better Accuracy

- Multiple signals reduce false positives
- Specialized models perform better in their domains
- Signal fusion catches edge cases

### 2. Improved Robustness

- System continues working even if one signal fails
- Multiple security layers provide defense in depth
- Fallback mechanisms ensure reliability

### 3. Continuous Learning

- System learns from patterns across all interactions
- Feedback signals improve future routing
- Collective knowledge grows over time

### 4. Emergent Capabilities

- System can handle cases no single component was designed for
- New patterns emerge from signal combinations
- Intelligence scales with system complexity

## Next Steps

- [What is Signal-Driven Decision?](signal-driven-decisions) - Deep dive into the decision engine
- [Configuration Guide](../installation/configuration) - Set up your own collective intelligence system
- [Signal Tutorials](../tutorials/signal/overview) - Learn to configure signals and decisions
