---
title: Categories Configuration
sidebar_label: Categories & Routing
---

# Categories Configuration

This guide provides quick configuration recipes for domain categories, model scoring, and reasoning control in vLLM Semantic Router. Use these examples to get started quickly with common routing patterns.

## Basic Category Definition

Define categories with MMLU mappings for domain classification:

```yaml
categories:
  - name: math
    description: "Mathematics and quantitative reasoning"
    mmlu_categories: ["math"]
  - name: physics
    description: "Physics and physical sciences"
    mmlu_categories: ["physics"]
  - name: business
    description: "Business and management queries"
    mmlu_categories: ["business"]
  - name: other
    description: "General knowledge topics"
    mmlu_categories: ["other"]
```

:::tip
Use `use_reasoning: true` for STEM domains (math, physics, chemistry) that benefit from step-by-step thinking.
:::

> See: [config.yaml#categories](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L73-L116) AND [config.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go).

## Enable Reasoning for STEM Domains

Enable step-by-step reasoning for domains that benefit from it:

```yaml
decisions:
  - name: "math_decision"
    description: "Mathematical queries with reasoning"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "qwen3"
        use_reasoning: true # Enable chain-of-thought
```

> See: [config.yaml#decisions](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L120-L411).

## Disable Reasoning for Fast Domains

Disable reasoning for domains requiring quick responses:

```yaml
decisions:
  - name: "business_decision"
    description: "Business queries - fast response"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "business"
    modelRefs:
      - model: "qwen3"
        use_reasoning: false # Fast responses
```

## Priority-Based Routing

Use priority to control decision order (higher = more priority):

```yaml
decisions:
  - name: "math_decision"
    priority: 100 # High priority for specific domains
    # ...

  - name: "general_decision"
    priority: 50 # Lower priority as fallback
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
    # ...
```

## Adding Custom System Prompts

Inject domain-specific system prompts:

```yaml
decisions:
  - name: "law_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "law"
    modelRefs:
      - model: "qwen3"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a legal expert. Provide accurate legal information while stating this is for informational purposes only."
```

## Combining Multiple Conditions

Use operators to combine conditions:

```yaml
decisions:
  - name: "stem_with_pii"
    rules:
      operator: "AND" # All conditions must match
      conditions:
        - type: "domain"
          name: "math"
        - type: "pii"
          detected: false
```

## Reasoning Family Configuration

Configure reasoning behavior per model family:

```yaml
model_config:
  "qwen3":
    reasoning_family: "qwen3" # Uses enable_thinking parameter
    preferred_endpoints: ["endpoint1"]

reasoning_families:
  qwen3:
    type: "chat_template_kwargs"
    parameter: "enable_thinking"

  deepseek:
    type: "chat_template_kwargs"
    parameter: "thinking"

  gpt:
    type: "reasoning_effort"
    parameter: "reasoning_effort"

default_reasoning_effort: high
```

> See: [config.yaml#model_config](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L53-L56) AND [config.yaml#reasoning_families](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L462-L477).

## Multimodal Category Configuration

Enable multimodal routing for categories that handle image inputs. Multimodal classification uses embedding-based similarity matching with CLIP vision transformer and BERT text embeddings.

### Basic Multimodal Category

```yaml
categories:
  - name: visual_analysis
    description: "Image analysis, object detection, visual reasoning, scene understanding"
    multimodal_enabled: true
    model_scores:
      - model: llava:7b
        score: 1.0
        capabilities: ["text", "image"]
```

### Image Generation Category

```yaml
categories:
  - name: image_generation
    description: "Generate images, create pictures, draw illustrations"
    model_scores:
      - model: image-generator
        score: 1.0
        capabilities: ["image"]
```

### Key Configuration Fields

- `multimodal_enabled: true` - Enables embedding-based multimodal classification for this category
  - Uses CLIP vision transformer for image embeddings
  - Fuses text and image embeddings via weighted combination
  - Classifies using cosine similarity against category descriptions

- `description` - Critical for multimodal classification
  - Used to generate category embeddings for similarity matching
  - Should be descriptive and capture the category's purpose
  - Example: "Image analysis, object detection, visual reasoning" vs "Analyze images"

- `capabilities: ["text", "image"]` - Specifies model capabilities
  - Used for routing decisions
  - Ensures requests are routed to models that support the required capabilities

## Quick Reference

| Domain Type            | use_reasoning | Priority | Use Case               |
| ---------------------- | ------------- | -------- | ---------------------- |
| Math/Physics/Chemistry | `true`        | 100      | Step-by-step solutions |
| Law/Health             | `false`       | 100      | Compliance, safety     |
| Business               | `false`       | 100      | Fast insights          |
| General/Other          | `false`       | 50       | Fallback routing       |

| Multimodal Category    | multimodal_enabled | Capabilities | Use Case                    |
| ---------------------- | ------------------ | ------------ | --------------------------- |
| visual_analysis        | `true`             | text, image  | Image understanding         |
| image_generation       | `false`            | image        | Image generation            |