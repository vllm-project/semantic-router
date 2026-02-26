---
title: PII Policy Configuration
sidebar_label: PII Policy
---

# PII Policy Configuration

This guide provides quick configuration recipes for PII (Personally Identifiable Information) detection and policy enforcement. PII detection is a **first-class signal** — define named `pii` rules under `signals.pii`, then reference them in `decisions` using `type: "pii"`.

## Enable PII Detection for a Decision

Define a PII signal and reference it in a decision rule:

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.9
      description: "Block all PII types"

decisions:
  - name: "block_pii"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Request blocked: personal information detected."
```

> See: [config/prompt-guard/hybrid.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/prompt-guard/hybrid.yaml).

## Allow Specific PII Types

Permit certain PII types while blocking all others using `pii_types_allowed`:

```yaml
signals:
  pii:
    - name: "pii_allow_location_org"
      threshold: 0.5
      pii_types_allowed:
        - "GPE"          # Allow geographic locations
        - "DATE_TIME"    # Allow dates and times
        - "ORGANIZATION" # Allow company names
      description: "Allow location, dates, org names — block all other PII"
```

The signal fires only when PII types **not** in `pii_types_allowed` are detected above the threshold.

> See: [pkg/config/config.go — PIIRule](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go).

## Supported PII Types

| PII Type       | Description             | Example               |
| -------------- | ----------------------- | --------------------- |
| `PERSON`       | Names of people         | "John Smith"          |
| `EMAIL_ADDRESS`| Email addresses         | "user@example.com"    |
| `PHONE_NUMBER` | Phone numbers           | "+1-555-0123"         |
| `GPE`          | Geographic locations    | "New York"            |
| `DATE_TIME`    | Dates and times         | "January 15, 2024"    |
| `ORGANIZATION` | Company/org names       | "Acme Corp"           |
| `CREDIT_CARD`  | Credit card numbers     | "4111-1111-1111-1111" |
| `US_SSN`       | Social security numbers | "123-45-6789"         |
| `IP_ADDRESS`   | IP addresses            | "192.168.1.1"         |
| `STREET_ADDRESS`| Physical addresses     | "123 Main St, NY"     |
| `US_DRIVER_LICENSE` | US Driver's License | "D123456789"        |
| `IBAN_CODE`    | Bank account numbers    | "GB82 WEST 1234..."   |
| `DOMAIN_NAME`  | Domain/website names    | "example.com"         |
| `ZIP_CODE`     | ZIP/postal codes        | "10001"               |

## Strict PII Policy (Block All)

For maximum privacy protection — block every detected PII type:

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.5
      # pii_types_allowed omitted → ALL detected PII types trigger the signal
      description: "Block all PII"

decisions:
  - name: "block_pii"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Request blocked: personal information detected."
```

## Permissive PII Policy (Allow Most Types)

Log PII without blocking by allowing all common types:

```yaml
signals:
  pii:
    - name: "pii_block_sensitive_only"
      threshold: 0.95   # Very high threshold
      pii_types_allowed:
        - "PERSON"
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
        - "GPE"
        - "DATE_TIME"
        - "ORGANIZATION"
      description: "Only block highly sensitive PII (SSN, credit card, etc.)"
```

## PII Model Configuration

Configure the underlying PII detection model:

```yaml
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9   # Global fallback threshold (overridden per signal rule)
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"
```

> See: [pkg/utils/pii](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/utils/pii).

## Domain-Specific PII Policies

Different domains may require different PII handling — combine `pii` + `domain` signals:

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.5
      description: "Block all PII"
    - name: "pii_allow_email_phone"
      threshold: 0.5
      pii_types_allowed:
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
      description: "Allow email/phone for appointment booking"
    - name: "pii_allow_org_location"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "Allow org/location names common in code"

  domains:
    - name: "health"
      mmlu_categories: ["health"]
    - name: "economics"
      mmlu_categories: ["economics"]
    - name: "computer_science"
      mmlu_categories: ["computer_science"]

decisions:
  # Health: block all PII except email/phone (for appointment booking)
  - name: "block_pii_health"
    priority: 998
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_allow_email_phone"
        - type: "domain"
          name: "health"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Please only share your email or phone number."

  # Finance: block all PII
  - name: "block_pii_finance"
    priority: 999
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
        - type: "domain"
          name: "economics"
    plugins:
      - type: "fast_response"
        configuration:
          message: "For your security, please do not share personal information in financial queries."

  # Code: allow org/location names common in code
  - name: "block_pii_code"
    priority: 997
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_allow_org_location"
        - type: "domain"
          name: "computer_science"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Request blocked: sensitive personal information detected in code."
```

## Debugging PII Detection

When PII is incorrectly blocked, check logs for:

```
PII signal fired: rule=pii_deny_all, detected_types=[PERSON, EMAIL_ADDRESS], threshold=0.5
```

To fix:

1. Add the PII type to `pii_types_allowed` if it should be permitted
2. Raise the signal `threshold` if false positives are occurring
3. Enable debug logging for detailed signal evaluation:

```yaml
logging:
  level: debug
  pii_detection: true
```

> See code: [pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go).
