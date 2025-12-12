---
title: Classifier Tuning
sidebar_label: Classifier Thresholds
---

# Classifier Tuning

This guide provides quick configuration recipes for tuning classification thresholds in vLLM Semantic Router. Adjust these settings to balance precision and recall based on your specific use case.

## Category Classifier Threshold

Adjust the confidence threshold for domain classification:

```yaml
classifier:
  category_model:
    model_id: "models/lora_intent_classifier_bert-base-uncased_model"
    threshold: 0.6 # Default: 0.6
    use_cpu: true
    category_mapping_path: "models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json"
```

> See: [config.yaml#classifier.category_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L59-L64).

| Threshold | Behavior                       |
| --------- | ------------------------------ |
| 0.5 - 0.6 | More permissive, higher recall |
| 0.7 - 0.8 | Balanced precision/recall      |
| 0.9+      | Very strict, fewer matches     |

## PII Detection Threshold

Configure PII detector sensitivity:

```yaml
classifier:
  pii_model:
    model_id: "models/lora_pii_detector_bert-base-uncased_model"
    threshold: 0.9 # Default: 0.9 (strict)
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

> See: [config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70).

:::tip
Use higher thresholds (0.9+) for PII to minimize false positives in production.
:::

## Jailbreak Detection Threshold

Tune prompt guard sensitivity:

```yaml
prompt_guard:
  enabled: true
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7 # Default: 0.7
  use_cpu: true
  jailbreak_mapping_path: "models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json"
```

> See: [config.yaml#prompt_guard](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L35-L41).

| Threshold | Trade-off                                 |
| --------- | ----------------------------------------- |
| 0.5 - 0.6 | Aggressive blocking, more false positives |
| 0.7       | Balanced (recommended)                    |
| 0.8 - 0.9 | Permissive, fewer blocks                  |

## Router Confidence Thresholds

Fine-tune the intelligent path selection:

```yaml
router:
  # High confidence threshold for automatic LoRA selection
  high_confidence_threshold: 0.99

  # Baseline scores for path evaluation
  lora_baseline_score: 0.8
  traditional_baseline_score: 0.7
  embedding_baseline_score: 0.75

  # Success calculation threshold
  success_confidence_threshold: 0.8

  # Default confidence threshold
  default_confidence_threshold: 0.95
```

> See: [config.yaml#router](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L414-L458).

## Semantic Cache Similarity Threshold

Adjust cache matching strictness per decision:

```yaml
decisions:
  - name: "health_decision"
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95 # Very strict for health

  - name: "general_decision"
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.75 # Relaxed for general
```

> See: [config.yaml#decisions](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L120-L411).

## BERT Model Threshold

Configure embedding model threshold for semantic matching:

```yaml
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6 # Semantic similarity threshold
  use_cpu: true
```

> See: [config.yaml#bert_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L1-L4).

## Tuning Guidelines

### When to Lower Thresholds

- Missing valid classifications (low recall)
- Cache hit ratio is too low
- Users report queries not being routed correctly

### When to Raise Thresholds

- Too many false positive matches
- Wrong categories being triggered
- PII/jailbreak false alarms

### Debugging Classification

Enable detailed logging to diagnose threshold issues:

```yaml
observability:
  metrics:
    enabled: true
  tracing:
    enabled: true
    sampling:
      type: "always_on"
```

Then check logs for classification confidence scores:

```
Classified query with confidence 0.72 to category 'math'
```
