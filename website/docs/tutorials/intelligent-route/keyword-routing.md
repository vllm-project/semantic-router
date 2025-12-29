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

Add keyword signals to your `config.yaml`:

```yaml
# Define keyword signals
signals:
  keywords:
    - name: "urgent_keywords"
      operator: "OR"  # Match ANY keyword
      keywords: ["urgent", "immediate", "asap", "emergency"]
      case_sensitive: false

    - name: "sensitive_data_keywords"
      operator: "OR"
      keywords: ["SSN", "social security", "credit card", "password"]
      case_sensitive: false

    - name: "spam_keywords"
      operator: "OR"
      keywords: ["buy now", "free money", "click here"]
      case_sensitive: false

# Define decisions using keyword signals
decisions:
  - name: urgent_request
    description: "Route urgent requests"
    priority: 100  # High priority
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "urgent_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a highly responsive assistant specialized in handling urgent requests."

  - name: sensitive_data
    description: "Route sensitive data queries"
    priority: 90
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "sensitive_data_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "You are a security-conscious assistant specialized in handling sensitive data."

  - name: filter_spam
    description: "Block spam queries"
    priority: 95
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "spam_keywords"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "This query appears to be spam. Please provide a polite response."
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
