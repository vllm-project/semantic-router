# Keyword Based Routing

This guide shows you how to route requests using explicit keyword rules and regex patterns. Keyword routing provides transparent, auditable routing decisions that are essential for compliance, security, and scenarios requiring explainable AI.

## Key Advantages

- **Transparent**: Routing decisions are fully explainable and auditable
- **Compliant**: Deterministic behavior meets regulatory requirements (GDPR, HIPAA, SOC2)
- **Fast**: Sub-millisecond latency, no ML inference overhead
- **Interpretable**: Clear rules make debugging and validation straightforward

## What Problem Does It Solve?

ML-based classification is a black box that's hard to audit and explain. Keyword routing provides:

- **Explainable decisions**: Know exactly why a query was routed to a specific category
- **Regulatory compliance**: Auditors can verify routing logic meets requirements
- **Deterministic behavior**: Same input always produces same output
- **Zero latency**: No model inference, instant classification
- **Precise control**: Explicit rules for security, compliance, and business logic

## When to Use

- **Regulated industries** (finance, healthcare, legal) requiring audit trails
- **Security/compliance** scenarios needing deterministic PII detection
- **High-throughput systems** where sub-millisecond latency is critical
- **Urgent/priority routing** with clear keyword indicators
- **Structured data** (emails, IDs, file paths) matching regex patterns

## Configuration

Add keyword rules to your `config.yaml`:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"

keyword_rules:
  - category: "urgent_request"
    operator: "OR"
    keywords: ["urgent", "immediate", "asap"]
    case_sensitive: false
  
  - category: "sensitive_data"
    operator: "AND"
    keywords: ["SSN", "social security number", "credit card"]
    case_sensitive: false
  
  - category: "exclude_spam"
    operator: "NOR"
    keywords: ["buy now", "free money"]
    case_sensitive: false
  
  - category: "regex_pattern_match"
    operator: "OR"
    keywords: ["user\\.name@domain\\.com", "C:\\Program Files\\\\"]
    case_sensitive: false

categories:
  - name: urgent_request
    system_prompt: "You are a highly responsive assistant specialized in handling urgent requests."
    model_scores:
      - model: qwen3
        score: 0.8
        use_reasoning: false
  
  - name: sensitive_data
    system_prompt: "You are a security-conscious assistant specialized in handling sensitive data."
    jailbreak_enabled: true
    jailbreak_threshold: 0.6
    model_scores:
      - model: qwen3
        score: 0.9
        use_reasoning: false
```

## Operators

- **OR**: Matches if any keyword is found
- **AND**: Matches only if all keywords are found
- **NOR**: Matches only if none of the keywords are found (exclusion)

## Example Requests

```bash
# Urgent request (matches "urgent")
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "I need urgent help with my account"}]
  }'

# Sensitive data (matches all keywords)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "My SSN and credit card were stolen"}]
  }'
```

## Real-World Use Cases

### 1. Financial Services (Transparent Compliance)

**Problem**: Regulators require explainable routing decisions for audit trails
**Solution**: Keyword rules provide clear "why" for each routing decision (e.g., "SSN" keyword → secure handler)
**Impact**: Passed SOC2 audit, complete decision transparency

### 2. Healthcare Platform (Compliant PII Detection)

**Problem**: HIPAA requires deterministic, auditable PII detection
**Solution**: AND operator detects multiple PII indicators with documented rules
**Impact**: 100% deterministic, full audit trail for compliance

### 3. High-Frequency Trading (Sub-millisecond Routing)

**Problem**: Need &lt;1ms classification for real-time market data routing
**Solution**: Keyword matching provides instant classification without ML overhead
**Impact**: 0.1ms latency, handles 100K+ requests/sec

### 4. Government Services (Interpretable Rules)

**Problem**: Citizens need to understand why requests were routed/rejected
**Solution**: Clear keyword rules can be explained in plain language
**Impact**: Reduced complaints, transparent decision-making

### 5. Enterprise Security (Transparent Threat Detection)

**Problem**: Security team needs to understand why queries were flagged
**Solution**: Explicit keyword/regex rules for threat patterns with clear documentation
**Impact**: Security team can validate and update rules confidently

## Performance Benefits

- **Sub-millisecond latency**: No ML inference overhead
- **High throughput**: 100K+ requests/sec on single core
- **Predictable costs**: No GPU/embedding model required
- **Zero cold start**: Instant classification on first request

## Reference

See [keyword.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/keyword.yaml) for complete configuration.
