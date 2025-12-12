---
title: PII Policy Configuration
sidebar_label: PII Policy
---

# PII Policy Configuration

This guide provides quick configuration recipes for PII (Personally Identifiable Information) detection and policy enforcement. Use these patterns to protect sensitive data based on your compliance requirements.

## Enable PII Detection per Decision

Add PII plugin to specific decision rules:

```yaml
decisions:
  - name: "health_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: "qwen3"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed: [] # Block all PII
```

> See: [config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139).

## Allow Specific PII Types

Permit certain PII types while blocking others:

```yaml
plugins:
  - type: "pii"
    configuration:
      enabled: true
      pii_types_allowed:
        - "LOCATION" # Allow location mentions
        - "DATE_TIME" # Allow dates and times
        - "ORGANIZATION" # Allow company names
      # All other types (PERSON, EMAIL, PHONE, etc.) will be blocked
```

> See: [config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139) AND [config.go pii_types_allowed](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go#L742).

## Supported PII Types

| PII Type       | Description             | Example               |
| -------------- | ----------------------- | --------------------- |
| `PERSON`       | Names of people         | "John Smith"          |
| `EMAIL`        | Email addresses         | "user@example.com"    |
| `PHONE`        | Phone numbers           | "+1-555-0123"         |
| `LOCATION`     | Geographic locations    | "New York"            |
| `DATE_TIME`    | Dates and times         | "January 15, 2024"    |
| `ORGANIZATION` | Company/org names       | "Acme Corp"           |
| `CREDIT_CARD`  | Credit card numbers     | "4111-1111-1111-1111" |
| `SSN`          | Social security numbers | "123-45-6789"         |
| `IP_ADDRESS`   | IP addresses            | "192.168.1.1"         |

## Strict PII Policy (Block All)

For maximum privacy protection:

```yaml
plugins:
  - type: "pii"
    configuration:
      enabled: true
      pii_types_allowed: [] # Empty list = block all PII
```

> See: [config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139).

## Permissive PII Policy (Warn Only)

Log PII without blocking:

```yaml
classifier:
  pii_model:
    threshold: 0.95 # Very high threshold
    # ...

decisions:
  - name: "internal_decision"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "PERSON"
            - "EMAIL"
            - "PHONE"
            - "LOCATION"
            - "DATE_TIME"
            - "ORGANIZATION"
```

> See: [config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70) AND [config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139).

## PII Model Configuration

Configure the underlying PII detection model:

```yaml
classifier:
  pii_model:
    model_id: "models/lora_pii_detector_bert-base-uncased_model"
    use_modernbert: false
    threshold: 0.9 # High threshold for fewer false positives
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

> See: [config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70) AND [pkg/utils/pii](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/utils/pii).

## Domain-Specific PII Policies

Different domains may require different PII handling:

```yaml
decisions:
  # Health: Very strict PII handling
  - name: "health_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed: [] # No PII allowed

  # Business: Allow organization names
  - name: "business_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "business"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "ORGANIZATION"
            - "LOCATION"

  # General: More permissive
  - name: "general_decision"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "LOCATION"
            - "DATE_TIME"
            - "ORGANIZATION"
```

## Debugging PII Detection

When PII is incorrectly blocked, check logs for:

```
PII policy violation for decision health_decision: denied PII types [PERSON, EMAIL]
```

To fix:

1. Add the PII type to `pii_types_allowed` if it should be permitted
2. Raise `classifier.pii_model.threshold` if false positives are occurring

> See code: [pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go).
