# Category Configuration

This guide covers how to configure categories in vLLM Semantic Router, including YAML syntax, parameter explanations, and best practices for optimal routing performance.

## Configuration Overview

Categories are configured in the main `config.yaml` file under the `categories` section. Each category defines how queries of that type should be handled, including model preferences, system prompts, reasoning settings, and routing behavior.

## Basic Configuration Structure

```yaml
categories:
  - name: "category_name"
    description: "Optional description"
    system_prompt: "Category-specific system prompt"
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

#### `system_prompt` (Optional)

- **Type**: String
- **Description**: Category-specific system prompt automatically injected into requests
- **Behavior**: Replaces existing system messages or adds new one at the beginning
- **Runtime Control**: Can be enabled/disabled via API when `--enable-system-prompt-api` flag is used
- **Example**: `"You are a mathematics expert. Provide step-by-step solutions."`

```yaml
categories:
  - name: "math"
    system_prompt: "You are a mathematics expert. Provide step-by-step solutions, show your work clearly, and explain mathematical concepts in an understandable way."
```

**Runtime Management**: System prompts can be dynamically controlled via REST API endpoints when the server is started with `--enable-system-prompt-api` flag:

```bash
# Start server with system prompt API enabled
./semantic-router --enable-system-prompt-api

# Toggle system prompt for specific category
curl -X PUT http://localhost:8080/config/system-prompts \
  -H "Content-Type: application/json" \
  -d '{"category": "math", "enabled": false}'

# Set injection mode (replace/insert)
curl -X PUT http://localhost:8080/config/system-prompts \
  -H "Content-Type: application/json" \
  -d '{"category": "math", "mode": "insert"}'
```

### Reasoning Configuration

#### `jailbreak_enabled` (Optional)

- **Type**: Boolean
- **Description**: Whether to enable jailbreak detection for this category
- **Default**: Inherits from global `prompt_guard.enabled` setting
- **Impact**: Enables or disables jailbreak protection for this specific category

```yaml
categories:
  - name: customer_support
    jailbreak_enabled: true  # Explicitly enable for public-facing
    model_scores:
      - model: qwen3
        score: 0.8

  - name: code_generation
    jailbreak_enabled: false  # Disable for internal tools
    model_scores:
      - model: qwen3
        score: 0.9

  - name: general
    # No jailbreak_enabled - inherits from global prompt_guard.enabled
    model_scores:
      - model: qwen3
        score: 0.5
```

#### `jailbreak_threshold` (Optional)

- **Type**: Float (0.0-1.0)
- **Description**: Confidence threshold for jailbreak detection
- **Default**: Inherits from global `prompt_guard.threshold` setting
- **Impact**: Controls sensitivity of jailbreak detection for this category
- **Tuning**: Higher values = stricter (fewer false positives), Lower values = more sensitive (catches more attacks)

```yaml
categories:
  - name: customer_support
    jailbreak_enabled: true
    jailbreak_threshold: 0.9  # Strict detection for public-facing
    model_scores:
      - model: qwen3
        score: 0.8

  - name: code_generation
    jailbreak_enabled: true
    jailbreak_threshold: 0.5  # Relaxed to reduce false positives on code
    model_scores:
      - model: qwen3
        score: 0.9

  - name: general
    # No jailbreak_threshold - inherits from global prompt_guard.threshold
    model_scores:
      - model: qwen3
        score: 0.5
```

**Threshold Guidelines**:

- **0.8-0.95**: High-security categories (customer support, business)
- **0.6-0.8**: Standard categories (general queries)
- **0.4-0.6**: Technical categories (code generation, development tools)

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

#### `lora_name` (Optional)

- **Type**: String
- **Description**: LoRA adapter name to use for this model
- **Purpose**: Enable intent-aware LoRA routing
- **Validation**: Must be defined in the model's `loras` list in `model_config`

When specified, the `lora_name` becomes the final model name in requests to vLLM, enabling automatic routing to LoRA adapters based on classified intent.

```yaml
# First, define available LoRA adapters in model_config
model_config:
  "llama2-7b":
    reasoning_family: "llama2"
    preferred_endpoints: ["vllm-primary"]
    loras:
      - name: "technical-lora"
        description: "Optimized for technical questions"
      - name: "medical-lora"
        description: "Specialized for medical domain"

# Then reference them in categories
categories:
  - name: "technical"
    model_scores:
      - model: "llama2-7b"        # Base model (for endpoint selection)
        lora_name: "technical-lora" # LoRA adapter name (final model name)
        score: 1.0
```

**How LoRA Routing Works**:

1. LoRA adapters are defined in `model_config` under the base model
2. Request is classified into a category (e.g., "technical")
3. Router selects the best `ModelScore` for that category
4. Configuration validator ensures `lora_name` is defined in model's `loras` list
5. If `lora_name` is specified, it replaces the base model name
6. Request is sent to vLLM with `model="technical-lora"`
7. vLLM automatically routes to the appropriate LoRA adapter

**Prerequisites**:

- vLLM server must be started with `--enable-lora` flag
- LoRA adapters must be registered using `--lora-modules` parameter
- LoRA names must be defined in `model_config` before use in `model_scores`

**Benefits**:

- **Domain Expertise**: Fine-tuned adapters for specific domains
- **Cost Efficiency**: Share base model weights across adapters
- **Easy A/B Testing**: Compare adapter versions by adjusting scores
- **Flexible Deployment**: Add/remove adapters without router restart
- **Configuration Validation**: Prevents typos and missing LoRA definitions

See [LoRA Routing Example](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/lora_routing.yaml) for complete configuration.

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

### Example 3: Intent-Aware LoRA Routing

```yaml
# Define LoRA adapters in model_config first
model_config:
  "llama2-7b":
    reasoning_family: "llama2"
    preferred_endpoints: ["vllm-primary"]
    loras:
      - name: "technical-lora"
        description: "Optimized for technical questions"
      - name: "medical-lora"
        description: "Specialized for medical domain"

# Then reference them in categories
categories:
  - name: "technical"
    description: "Programming and technical questions"
    model_scores:
      - model: "llama2-7b"
        lora_name: "technical-lora"  # Routes to technical LoRA adapter
        score: 1.0
        use_reasoning: true

  - name: "medical"
    description: "Medical and healthcare questions"
    model_scores:
      - model: "llama2-7b"
        lora_name: "medical-lora"    # Routes to medical LoRA adapter
        score: 1.0
        use_reasoning: true

  - name: "general"
    description: "General questions"
    model_scores:
      - model: "llama2-7b"           # No lora_name - uses base model
        score: 0.8
        use_reasoning: false
```

### Example 4: Security-Focused Configuration (Jailbreak Protection)

```yaml
categories:
  # High-security public-facing category with strict threshold
  - name: "customer_support"
    description: "Customer support and general inquiries"
    jailbreak_enabled: true  # Strict jailbreak protection
    jailbreak_threshold: 0.9  # High threshold for public-facing
    use_reasoning: false
    model_scores:
      - model: "phi4"
        score: 0.9
      - model: "mistral-small3.1"
        score: 0.7

  # Technical category with relaxed threshold
  - name: "code_generation"
    description: "Code generation for developers"
    jailbreak_enabled: true  # Keep enabled
    jailbreak_threshold: 0.5  # Lower threshold to reduce false positives on code
    use_reasoning: true
    reasoning_effort: "medium"
    model_scores:
      - model: "gemma3:27b"
        score: 0.9
      - model: "phi4"
        score: 0.7

  # General category using global default
  - name: "general"
    description: "General queries"
    # jailbreak_enabled not specified - inherits from global prompt_guard.enabled
    use_reasoning: false
    model_scores:
      - model: "phi4"
        score: 0.6
      - model: "mistral-small3.1"
        score: 0.6
```

### Example 4: Multi-Category Configuration

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
    system_prompt: "You are a mathematics expert. Provide step-by-step solutions."
    use_reasoning: true
    reasoning_effort: "high"
    model_scores:
      - model: "phi4"
        score: 1.0
```

## Complete Configuration Example

```yaml
categories:
  - name: "math"
    description: "Mathematical problems and calculations"
    system_prompt: "You are a mathematics expert. Provide step-by-step solutions, show your work clearly, and explain mathematical concepts in an understandable way."
    use_reasoning: true
    reasoning_effort: "high"
    model_scores:
      - model: "openai/gpt-oss-20b"
        score: 0.9
        use_reasoning: true

  - name: "computer science"
    description: "Programming and software engineering"
    system_prompt: "You are a computer science expert. Provide clear, practical solutions with code examples when helpful."
    use_reasoning: true
    reasoning_effort: "medium"
    model_scores:
      - model: "openai/gpt-oss-20b"
        score: 0.8
        use_reasoning: true

  - name: "business"
    description: "Business strategy and management"
    system_prompt: "You are a professional business consultant. Provide practical, actionable advice."
    use_reasoning: false
    model_scores:
      - model: "openai/gpt-oss-20b"
        score: 0.7
        use_reasoning: false
```

## Next Steps

- [**Supported Categories**](supported-categories.md) - Review all available categories
- [**Keyword Classifier Configuration**](keyword-configuration.md) - Learn how to configure keyword-based routing rules
- [**Technical Details**](technical-details.md) - Understand the implementation
- [**Category Overview**](overview.md) - Learn about the category system
