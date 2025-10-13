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
    threshold: 0.7                 # Detection sensitivity (0.0-1.0)
    use_cpu: true                  # Run on CPU
    pii_mapping_path: "config/pii_type_mapping.json"  # Path to PII type mapping
```

### Model-Specific PII Policies

Configure different PII policies for different models:

```yaml
# vLLM endpoints configuration
vllm_endpoints:
  - name: secure-model
    address: "127.0.0.1"
    port: 8080
  - name: general-model
    address: "127.0.0.1"
    port: 8081

# Model-specific configurations
model_config:
  secure-llm:
    pii_policy:
      allow_by_default: false      # Block all PII by default
      pii_types:                   # Only allow these specific types
        - "EMAIL_ADDRESS"
        - "GPE"
        - "ORGANIZATION"

  general-llm:
    pii_policy:
      allow_by_default: true       # Allow all PII by default
      pii_types: []                # Not used when allow_by_default is true
```

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

### 2. Policy Design

- Use `allow_by_default: false` for sensitive models
- Explicitly list allowed PII types for clarity
- Consider different policies for different use cases

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
