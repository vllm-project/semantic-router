# PII Detection

Semantic Router provides built-in Personally Identifiable Information (PII) detection to protect sensitive data in user queries. The system uses fine-tuned BERT models to identify and handle various types of PII according to configurable policies.

## Overview

The PII detection system:

- **Identifies** common PII types in user queries
- **Enforces** configurable PII policies per decision
- **Blocks** requests containing sensitive information based on signal rules
- **Integrates** with signal-driven decisions for fine-grained control
- **Logs** policy violations for monitoring

## Supported PII Types

The system can detect the following PII types:

| PII Type | Description | Examples |
|----------|-------------|----------|
| `PERSON` | Person names | "John Smith", "Mary Johnson" |
| `EMAIL_ADDRESS` | Email addresses | "user@example.com" |
| `PHONE_NUMBER` | Phone numbers | "+1-555-123-4567", "(555) 123-4567" |
| `US_SSN` | US Social Security Numbers | "123-45-6789" |
| `STREET_ADDRESS` | Physical addresses | "123 Main St, New York, NY" |
| `GPE` | Geopolitical entities | Countries, states, cities |
| `ORGANIZATION` | Organization names | "Microsoft", "OpenAI" |
| `CREDIT_CARD` | Credit card numbers | "4111-1111-1111-1111" |
| `US_DRIVER_LICENSE` | US Driver's License | "D123456789" |
| `IBAN_CODE` | International Bank Account Number | "GB82 WEST 1234 5698 7654 32" |
| `IP_ADDRESS` | IP addresses | "192.168.1.1", "2001:db8::1" |
| `DOMAIN_NAME` | Domain/website names | "example.com", "google.com" |
| `DATE_TIME` | Date/time information | "2024-01-15", "January 15th" |
| `AGE` | Age information | "25 years old", "born in 1990" |
| `NRP` | Nationality/Religious/Political groups | "American", "Christian", "Democrat" |
| `ZIP_CODE` | ZIP/postal codes | "10001", "SW1A 1AA" |

## Configuration

PII detection is now a **first-class signal** in the signal layer. You define named `pii` rules under `signals.pii`, then reference them in `decisions` using `type: "pii"`.

### Basic PII Detection

```yaml
# router-config.yaml

# ── PII Classifier Model ──────────────────────────────────────────────────
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"

# ── Signals ───────────────────────────────────────────────────────────────
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.9
      description: "Block all PII types"

# ── Decisions ─────────────────────────────────────────────────────────────
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
          message: "Request blocked: personal information detected. Please remove sensitive data and try again."
```

### Allow-List: Permit Specific PII Types

Use `pii_types_allowed` to permit certain PII types while blocking all others:

```yaml
signals:
  pii:
    # Block all PII
    - name: "pii_deny_all"
      threshold: 0.5
      description: "Block all PII"

    # Allow email and phone (e.g., for appointment booking)
    - name: "pii_allow_email_phone"
      threshold: 0.5
      pii_types_allowed:
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
      description: "Allow email and phone, block SSN/credit card etc."

    # Allow org/location names (e.g., for code/technical content)
    - name: "pii_allow_org_location"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "Allow geo, org, dates — common in code and config files"
```

### Domain-Aware PII Policies

Combine PII signals with domain signals for context-aware data protection:

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
      description: "Allow email and phone for appointment booking"

  domains:
    - name: "economics"
      description: "Finance and economics"
      mmlu_categories: ["economics"]
    - name: "health"
      description: "Health and medical"
      mmlu_categories: ["health"]

decisions:
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

  # Health: allow email/phone for appointment booking, block other PII
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
          message: "Please only share your email or phone number. Do not include other personal details."
```

### Environment-Based PII Policies (Dev vs Prod)

Apply different PII thresholds and allow-lists per environment:

```yaml
signals:
  pii:
    # Dev: relaxed — code context has many org/location references
    - name: "pii_dev"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "Relaxed PII for dev environment"

    # Prod: strict — block all PII for data minimization
    - name: "pii_prod"
      threshold: 0.9
      description: "Strict PII for production environment"

decisions:
  - name: "block_pii_prod"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_prod"
    modelRefs:
      - model: "qwen14b-prod"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Request blocked: personal information detected."
```

## Signal Configuration Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Signal name referenced in decision conditions (`type: "pii"`) |
| `threshold` | float (0.0–1.0) | ✅ | Minimum confidence score for PII entity detection |
| `pii_types_allowed` | string[] | ❌ | PII types that are **permitted** (not blocked). When empty, ALL detected PII types trigger the signal |
| `include_history` | bool | ❌ | When `true`, all conversation messages are analysed (default: `false`) |
| `description` | string | ❌ | Human-readable description of this rule |

**Threshold Guidelines by Use Case:**

- **Critical (healthcare, finance, legal)**: `0.9–0.95` — strict detection, fewer false positives
- **Customer-facing (support, sales)**: `0.75–0.85` — balanced detection
- **Internal tools (code, testing)**: `0.5–0.65` — relaxed to reduce false positives on technical content
- **Public content (docs, marketing)**: `0.6–0.75` — broader detection before publication

## PII Classifier Model Configuration

The `classifier.pii_model` section configures the underlying ML model used by all PII signals:

```yaml
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9   # Global fallback threshold (overridden per signal rule)
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"
```

## How PII Detection Works

1. **Signal Evaluation**: All `pii` signal rules run **in parallel** with other signals (keyword, domain, jailbreak, etc.) — zero added latency to the routing pipeline
2. **Entity Detection**: The PII classifier identifies PII entities in the request text
3. **Allow-List Check**: Each rule checks whether detected PII types are in its `pii_types_allowed` list
4. **Signal Fires**: If denied PII types are detected above the threshold, the signal fires
5. **Decision Matching**: Decisions reference fired signals via `type: "pii"` conditions
6. **Action**: Matching decisions execute their plugins (e.g., `fast_response` to block the request)
7. **Logging**: All PII detections and policy decisions are logged for monitoring

## API Integration

PII detection is automatically integrated into the routing process. When a request is made to the router, the system:

1. Analyses the input text for PII using the configured classifier
2. Evaluates all `pii` signal rules in parallel
3. Fires signals for rules where denied PII types are detected above threshold
4. Routes to decisions that match the fired signals

### Classification Endpoint

You can also check PII detection directly using the classification API:

```bash
curl -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My email is john.doe@example.com and I live in New York"
  }'
```

The response includes PII information along with classification results.

## Monitoring and Metrics

The system exposes PII-related metrics:

```
# Prometheus metrics
pii_detections_total{type="EMAIL_ADDRESS"} 45
pii_detections_total{type="PERSON"} 23
pii_policy_violations_total 12
pii_requests_blocked_total 8
```

## Best Practices

### 1. Use Priority to Order Security Decisions

Set PII-blocking decisions at high priority (e.g., `999`) so they are evaluated before routing decisions:

```yaml
decisions:
  - name: "block_pii"
    priority: 999   # Evaluated before routing decisions (priority < 999)
    rules:
      ...
```

### 2. Combine with Jailbreak Detection

Use both `pii` and `jailbreak` signals together for comprehensive security:

```yaml
signals:
  jailbreak:
    - name: "jailbreak_standard"
      threshold: 0.65
  pii:
    - name: "pii_deny_all"
      threshold: 0.5

decisions:
  - name: "block_jailbreak"
    priority: 1000
    rules:
      operator: "OR"
      conditions:
        - type: "jailbreak"
          name: "jailbreak_standard"
    plugins:
      - type: "fast_response"
        configuration:
          message: "Request blocked: policy violation."

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

### 3. Use Domain Context for Smarter Policies

Different domains have different PII sensitivity requirements. Combine `pii` + `domain` signals for context-aware policies rather than applying a single global rule.

### 4. Enable History for Conversational Applications

For multi-turn conversations, enable `include_history: true` to detect PII shared across multiple messages:

```yaml
signals:
  pii:
    - name: "pii_full_history"
      threshold: 0.9
      include_history: true
      description: "Detect PII across full conversation history"
```

## Troubleshooting

### High False Positives

- Increase the signal `threshold` (e.g., from `0.5` to `0.8`)
- Add common false-positive types to `pii_types_allowed`
- For code/technical content, allow `GPE`, `ORGANIZATION`, `DATE_TIME`

### Missed PII Detection

- Lower the signal `threshold`
- Check if the PII type is in the [supported types table](#supported-pii-types)
- Verify the PII model is properly loaded

### Debug Mode

Enable detailed PII logging:

```yaml
logging:
  level: debug
  pii_detection: true
```

This will log all PII detection decisions and signal evaluations.
