# Category Configuration

This guide covers how to configure categories in vLLM Semantic Router, including YAML syntax, parameter explanations, and best practices for optimal routing performance.

## Configuration Overview

Categories are configured in the main `config.yaml` file under the `categories` section. Each category defines how queries of that type should be handled, including model preferences, reasoning settings, and routing behavior.

## Basic Configuration Structure

```yaml
categories:
  - name: "category_name"
    description: "Optional description"
    use_reasoning: true|false
    reasoning_description: "Why reasoning is needed"
    reasoning_effort: "low|medium|high"
    model_scores:
      - model: "model_name"
        score: 0.0-1.0
```

## Configuration Parameters

### Core Parameters

#### `name` (Required)

- **Type**: String
- **Description**: Unique identifier for the category
- **Valid Values**: Any string matching supported categories
- **Example**: `"math"`, `"computer science"`, `"business"`

```yaml
categories:
  - name: "math"
```

#### `description` (Optional)

- **Type**: String
- **Description**: Human-readable description of the category
- **Purpose**: Documentation and debugging
- **Example**: `"Mathematical problems and calculations"`

```yaml
categories:
  - name: "math"
    description: "Mathematical problems requiring step-by-step solutions"
```

### Reasoning Configuration

#### `use_reasoning` (Required)

- **Type**: Boolean
- **Description**: Whether to enable reasoning mode for this category
- **Default**: `false`
- **Impact**: Enables step-by-step problem solving

```yaml
categories:
  - name: "math"
    use_reasoning: true  # Enable reasoning for math problems
```

#### `reasoning_description` (Optional)

- **Type**: String
- **Description**: Explanation of why reasoning is needed
- **Purpose**: Documentation and model context
- **Best Practice**: Provide clear justification

```yaml
categories:
  - name: "chemistry"
    use_reasoning: true
    reasoning_description: "Chemical reactions require systematic analysis"
```

#### `reasoning_effort` (Optional)

- **Type**: String
- **Valid Values**: `"low"`, `"medium"`, `"high"`
- **Default**: `"medium"`
- **Description**: Controls the depth of reasoning

```yaml
categories:
  - name: "math"
    use_reasoning: true
    reasoning_effort: "high"  # Maximum reasoning depth
```

**Reasoning Effort Levels**:

- **Low**: Basic step-by-step thinking (1-3 steps)
- **Medium**: Moderate analysis (3-7 steps)
- **High**: Deep reasoning (7-15 steps)

### Model Scoring

#### `model_scores` (Required)

- **Type**: Array of model-score pairs
- **Description**: Defines model preferences for this category
- **Purpose**: Intelligent model selection based on domain expertise

```yaml
categories:
  - name: "math"
    model_scores:
      - model: "phi4"
        score: 1.0      # Highest preference
      - model: "mistral-small3.1"
        score: 0.8      # Second choice
      - model: "gemma3:27b"
        score: 0.6      # Fallback option
```

**Score Guidelines**:

- **1.0**: Perfect match, primary choice
- **0.8-0.9**: Excellent capability
- **0.6-0.7**: Good capability
- **0.4-0.5**: Adequate capability
- **0.0-0.3**: Poor capability, avoid if possible

## Complete Configuration Examples

### Example 1: STEM Category (Reasoning Enabled)

```yaml
categories:
  - name: "math"
    description: "Mathematical problems requiring step-by-step reasoning"
    use_reasoning: true
    reasoning_description: "Mathematical problems require systematic analysis"
    reasoning_effort: "high"
    model_scores:
      - model: "phi4"
        score: 1.0
      - model: "mistral-small3.1"
        score: 0.8
      - model: "gemma3:27b"
        score: 0.6
```

### Example 2: Professional Category (Reasoning Disabled)

```yaml
categories:
  - name: "business"
    description: "Business strategy and management discussions"
    use_reasoning: false
    reasoning_description: "Business content is typically conversational"
    reasoning_effort: "low"
    model_scores:
      - model: "phi4"
        score: 0.8
      - model: "gemma3:27b"
        score: 0.4
      - model: "mistral-small3.1"
        score: 0.2
```

### Example 3: Multi-Category Configuration

```yaml
categories:
  # Technical categories with reasoning
  - name: "computer science"
    use_reasoning: true
    reasoning_description: "Programming requires logical analysis"
    reasoning_effort: "medium"
    model_scores:
      - model: "gemma3:27b"
        score: 0.6
      - model: "mistral-small3.1"
        score: 0.6
      - model: "phi4"
        score: 0.0

  - name: "physics"
    use_reasoning: true
    reasoning_description: "Physics concepts need systematic thinking"
    reasoning_effort: "medium"
    model_scores:
      - model: "gemma3:27b"
        score: 0.4
      - model: "phi4"
        score: 0.4
      - model: "mistral-small3.1"
        score: 0.4

  # General categories without reasoning
  - name: "history"
    use_reasoning: false
    reasoning_description: "Historical content is narrative-based"
    model_scores:
      - model: "mistral-small3.1"
        score: 0.8
      - model: "phi4"
        score: 0.6
      - model: "gemma3:27b"
        score: 0.4

  - name: "other"
    use_reasoning: false
    reasoning_description: "General content doesn't require reasoning"
    model_scores:
      - model: "gemma3:27b"
        score: 0.8
      - model: "phi4"
        score: 0.6
      - model: "mistral-small3.1"
        score: 0.6
```

## Configuration Best Practices

### 1. Model Score Optimization

**Principle**: Assign scores based on actual model performance for each domain.

```yaml
# Good: Scores reflect model strengths
categories:
  - name: "math"
    model_scores:
      - model: "phi4"
        score: 1.0    # Excellent at math
      - model: "mistral-small3.1"
        score: 0.8    # Good at math
      - model: "gemma3:27b"
        score: 0.6    # Adequate at math

# Avoid: Uniform scores don't leverage model strengths
categories:
  - name: "math"
    model_scores:
      - model: "phi4"
        score: 0.8    # Underutilizes phi4's math strength
      - model: "mistral-small3.1"
        score: 0.8
      - model: "gemma3:27b"
        score: 0.8
```

### 2. Reasoning Configuration

**Enable reasoning for complex domains**:

```yaml
# Reasoning recommended for:
categories:
  - name: "math"
    use_reasoning: true
    reasoning_effort: "high"

  - name: "computer science"
    use_reasoning: true
    reasoning_effort: "medium"

  - name: "chemistry"
    use_reasoning: true
    reasoning_effort: "high"

# Reasoning not needed for:
categories:
  - name: "business"
    use_reasoning: false

  - name: "history"
    use_reasoning: false
```

### 3. Performance Tuning

**Balance accuracy vs. latency**:

```yaml
# High-performance setup (lower latency)
categories:
  - name: "math"
    use_reasoning: true
    reasoning_effort: "medium"  # Reduced from "high"
    model_scores:
      - model: "phi4"
        score: 1.0
      - model: "mistral-small3.1"
        score: 0.6  # Larger gap for faster selection

# High-accuracy setup (higher latency)
categories:
  - name: "math"
    use_reasoning: true
    reasoning_effort: "high"    # Maximum reasoning
    model_scores:
      - model: "phi4"
        score: 1.0
      - model: "mistral-small3.1"
        score: 0.9  # Close scores for better fallback
```

## Classifier Configuration

Categories work with the classifier configuration:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6              # Classification confidence threshold
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
```

### Threshold Tuning

- **Higher threshold (0.7-0.9)**: More conservative, fewer false positives
- **Lower threshold (0.4-0.6)**: More aggressive, better coverage
- **Recommended**: Start with 0.6 and adjust based on performance

## Validation and Testing

### Configuration Validation

```bash
# Test configuration syntax
make build

# Validate category mappings
curl -X GET http://localhost:8080/api/v1/models
```

### Performance Testing

```bash
# Test category classification
curl -X POST http://localhost:8080/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "Solve x^2 + 5x + 6 = 0"}'

# Expected response for math category
{
  "classification": {
    "category": "math",
    "confidence": 0.95,
    "processing_time_ms": 45
  }
}
```

## Common Issues and Solutions

### Issue 1: Low Classification Accuracy

**Symptoms**: Queries routed to wrong categories

**Solutions**:

```yaml
# Increase classification threshold
classifier:
  category_model:
    threshold: 0.7  # Increase from 0.6

# Add more specific model scores
categories:
  - name: "math"
    model_scores:
      - model: "phi4"
        score: 1.0
      - model: "other-models"
        score: 0.0  # Explicitly avoid poor performers
```

### Issue 2: High Latency

**Symptoms**: Slow response times

**Solutions**:

```yaml
# Reduce reasoning effort
categories:
  - name: "math"
    reasoning_effort: "medium"  # Reduce from "high"

# Optimize model selection
categories:
  - name: "math"
    model_scores:
      - model: "fast-model"
        score: 0.9
      - model: "slow-model"
        score: 0.1  # Deprioritize slow models
```

### Issue 3: Poor Model Selection

**Symptoms**: Suboptimal model choices

**Solutions**:

```yaml
# Review and adjust model scores based on benchmarks
categories:
  - name: "computer science"
    model_scores:
      - model: "code-specialized-model"
        score: 1.0  # Use specialized models
      - model: "general-model"
        score: 0.3  # Reduce general model preference
```

## Migration Guide

### From Legacy Configuration

```yaml
# Old format (deprecated)
routing_rules:
  - category: "math"
    model: "phi4"
    reasoning: true

# New format (current)
categories:
  - name: "math"
    use_reasoning: true
    reasoning_effort: "high"
    model_scores:
      - model: "phi4"
        score: 1.0
```

## Next Steps

- [**Supported Categories**](supported-categories.md) - Review all available categories
- [**Technical Details**](technical-details.md) - Understand the implementation
- [**Category Overview**](overview.md) - Learn about the category system
