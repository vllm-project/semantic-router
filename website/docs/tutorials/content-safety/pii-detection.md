# PII Detection

Semantic Router provides built-in Personally Identifiable Information (PII) detection to protect sensitive data in user queries. The system uses fine-tuned BERT models to identify and handle various types of PII according to configurable policies.

## Overview

The PII detection system:

- **Identifies** common PII types in user queries
- **Enforces** model-specific PII policies
- **Blocks or masks** sensitive information based on configuration
- **Filters** model candidates based on PII compliance
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

### Basic PII Detection

Enable PII detection in your configuration:

```yaml
# config/config.yaml
classifier:
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_model"
    threshold: 0.7                 # Global detection threshold (0.0-1.0)
    use_cpu: true                  # Run on CPU
    pii_mapping_path: "config/pii_type_mapping.json"  # Path to PII type mapping
```

### Model-Level PII Policies

**Current Implementation**: PII detection policies are configured at the **model level**, not the category level. Each model can specify which PII types it allows or blocks.

```yaml
# Global PII configuration - detection threshold applies to all categories
classifier:
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    threshold: 0.7  # Global detection threshold
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"

# Model-specific PII policies - controls what PII each model allows
model_config:
  "secure-healthcare-llm":
    reasoning_family: "qwen3"
    preferred_endpoints: ["endpoint1"]
    pii_policy:
      allow_by_default: false      # Block all PII by default for healthcare model
      pii_types_allowed:           # Only allow these specific types
        - "PERSON"                 # Patient names may be needed
        - "GPE"                    # Geographic locations

  "finance-llm":
    reasoning_family: "qwen3"
    preferred_endpoints: ["endpoint2"]
    pii_policy:
      allow_by_default: false      # Block all PII by default for finance
      pii_types_allowed: []        # Don't allow any PII types

  "general-llm":
    reasoning_family: "qwen3"
    preferred_endpoints: ["endpoint1"]
    pii_policy:
      allow_by_default: true       # Allow all PII for general model
      # pii_types_allowed not needed when allow_by_default is true

# Categories route to models based on model_scores
categories:
  - name: healthcare
    system_prompt: "You are a healthcare expert..."
    model_scores:
      - model: secure-healthcare-llm  # This model has strict PII policy
        score: 1.0
        use_reasoning: false

  - name: finance
    system_prompt: "You are a finance expert..."
    model_scores:
      - model: finance-llm  # This model blocks all PII
        score: 1.0
        use_reasoning: false

  - name: general
    system_prompt: "You are a helpful assistant..."
    model_scores:
      - model: general-llm  # This model allows PII
        score: 1.0
        use_reasoning: false
```

**How It Works:**

1. **Detection**: PII classifier detects PII in the input using the global threshold (0.7)
2. **Model Selection**: Router selects a model based on category classification
3. **Policy Check**: Router checks if the selected model's `pii_policy` allows the detected PII types
4. **Routing Decision**: If PII is detected and the model blocks it, the request is rejected

**Configuration Options:**

- `allow_by_default: true` - Model allows all PII types (default if not specified)
- `allow_by_default: false` with `pii_types_allowed: []` - Model blocks all PII
- `allow_by_default: false` with `pii_types_allowed: ["TYPE1", "TYPE2"]` - Model only allows specific PII types


## How PII Detection Works

The PII detection system works as follows:

1. **Detection**: The PII classifier model analyzes incoming text to identify PII types
2. **Policy Check**: The system checks if the detected PII types are allowed for the target model
3. **Routing Decision**: Models that don't allow the detected PII types are filtered out
4. **Logging**: All PII detections and policy decisions are logged for monitoring

## API Integration

PII detection is automatically integrated into the routing process. When a request is made to the router, the system:

1. Analyzes the input text for PII using the configured classifier
2. Checks PII policies for candidate models
3. Filters out models that don't allow the detected PII types
4. Routes to an appropriate model that can handle the PII

**Note**: PII detection uses a global threshold (`classifier.pii_model.threshold`) for detection. PII policies are enforced at the model level via `pii_policy` configuration, which controls what types of PII each model accepts.

### Classification Endpoint

You can also check PII detection directly using the classification API:

```bash
curl -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My email is john.doe@example.com and I live in New York"
  }'
```

The response includes PII information along with category classification results.

## Monitoring and Metrics

The system exposes PII-related metrics:

```
# Prometheus metrics
pii_detections_total{type="EMAIL_ADDRESS"} 45
pii_detections_total{type="PERSON"} 23
pii_policy_violations_total{model="secure-model"} 12
pii_requests_blocked_total 8
pii_requests_masked_total 15
```

## Best Practices

### 1. Threshold Tuning

- Start with `threshold: 0.7` for balanced accuracy
- Increase to `0.8-0.9` for high-security environments
- Decrease to `0.5-0.6` for broader detection
- **Use model-level policies** to control which PII types each model can handle

#### PII Sensitivity Guidelines by Use Case

Different use cases have different PII sensitivity requirements. Configure the global detection threshold based on your most sensitive use case, then use model-level `pii_policy` to control access:

**High-Security Models (Healthcare, Finance, Legal):**

- Global threshold: `0.7` (standard detection)
- Model policy: `allow_by_default: false` with specific `pii_types_allowed`
- Rationale: Detect all PII, then selectively allow only necessary types
- Example: Healthcare model allows `PERSON` for patient names but blocks `SSN`, `CREDIT_CARD`
- Risk management: Model-level filtering prevents PII leakage

**General-Purpose Models:**

- Global threshold: `0.7` (standard detection)
- Model policy: `allow_by_default: true` (allows all PII)
- Rationale: General models can handle PII for broader use cases
- Example: Support chatbots need to process customer names, emails, etc.
- Risk management: Ensure logging and monitoring for PII usage

**Restricted Models (Code, Development):**

- Global threshold: `0.7` (keep standard to catch real PII)
- Model policy: `allow_by_default: true` or specific allowed types
- Rationale: Code artifacts may look like PII (UUIDs, test data)
- Example: Development tools need to process code with test SSNs, example emails
- Risk management: Use separate models for production vs development

### 2. Policy Design

- Use `allow_by_default: false` for sensitive models
- Explicitly list allowed PII types for clarity
- Consider different policies for different use cases
- **Use strict global thresholds combined with model-level policies** for defense in depth

### 3. Action Selection

- Use `block` for high-security scenarios
- Use `mask` when processing is still needed
- Use `allow` with logging for audit requirements

### 4. Model Filtering

- Configure PII policies to automatically filter model candidates
- Ensure at least one model can handle each PII scenario
- Test policy combinations thoroughly

## Troubleshooting

### Common Issues

**High False Positives**

- Lower the detection threshold
- Review training data for edge cases
- Consider custom model fine-tuning

**Missed PII Detection**

- Increase detection sensitivity
- Check if PII type is supported
- Verify model is properly loaded

**Policy Conflicts**

- Ensure at least one model allows detected PII types
- Check `allow_by_default` settings
- Review `pii_types_allowed` lists

### Debug Mode

Enable detailed PII logging:

```yaml
logging:
  level: debug
  pii_detection: true
```

This will log all PII detection decisions and policy evaluations.
