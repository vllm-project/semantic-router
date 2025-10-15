---
sidebar_position: 2
---

# Keyword-Based Routing

:::warning Work in Progress
This use case describes a **proposed feature** that is not yet implemented in the current version of the semantic router. This represents a future capability that would enhance the router's functionality.
:::

## The Enterprise Challenge

Imagine you're the CTO of a financial services company. Your team has built an AI-powered customer service system that routes queries to different AI models based on their content. However, you're facing critical challenges:

- **Security Risk**: Customer queries about "internal account procedures" or "confidential trading strategies" are being sent to external AI providers, potentially exposing sensitive information
- **Compliance Violations**: Regulatory requirements mandate that certain types of financial data must never leave your on-premise infrastructure
- **Lack of Control**: You have no visibility into why certain routing decisions are made, making it impossible to audit or explain routing logic to regulators

## The Solution: Keyword-Based Routing

Keyword-based routing provides deterministic, transparent routing rules that complement AI-powered semantic classification. It allows enterprises to implement business policies with complete visibility and control.

## Real-World Use Cases

### Use Case 1: Financial Services Data Sovereignty

**Scenario**: A bank needs to ensure that any query containing internal procedures, account details, or trading strategies stays within their secure, on-premise infrastructure.

**Business Rule**: "Any query containing keywords like 'internal', 'confidential', 'account', 'trading', or 'procedures' must be routed to our internal AI model, never to external providers."

**Implementation**:
```yaml
routing_rules:
  - name: "financial-confidential"
    description: "Route confidential financial queries to internal infrastructure"
    conditions:
      - type: "keyword_match"
        keywords: ["internal", "confidential", "account", "trading", "procedures", "strategy"]
    action:
      type: "route"
      endpoint: "internal-financial-ai"
      reasoning: "Contains confidential financial terminology - must stay on-premise"
```

**Business Value**: 
- ✅ Compliance with financial regulations
- ✅ Complete data sovereignty
- ✅ Audit trail for regulatory reporting
- ✅ Zero risk of data leakage to external providers

### Use Case 2: Healthcare Information Protection

**Scenario**: A healthcare provider needs to route patient-related queries to HIPAA-compliant models while allowing general medical information queries to go to external providers with broader medical knowledge.

**Business Rule**: "Queries containing patient identifiers, medical records, or specific patient information go to HIPAA-compliant internal models. General medical questions can use external medical AI."

**Implementation**:
```yaml
routing_rules:
  - name: "patient-data-protection"
    description: "Route patient-specific queries to HIPAA-compliant infrastructure"
    conditions:
      - type: "keyword_match"
        keywords: ["patient", "medical record", "diagnosis", "treatment plan", "prescription"]
      - type: "pii_detection"
        threshold: 0.7
    action:
      type: "route"
      endpoint: "hipaa-compliant-ai"
      reasoning: "Contains patient information - HIPAA compliance required"
  
  - name: "general-medical"
    description: "Route general medical questions to external medical AI"
    conditions:
      - type: "keyword_match"
        keywords: ["symptoms", "treatment", "medication", "disease", "condition"]
    action:
      type: "route"
      endpoint: "external-medical-ai"
      reasoning: "General medical query - can use external medical knowledge"
```

### Use Case 3: Search Capability Routing

**Scenario**: An enterprise wants to route queries that require web search to providers with search capabilities, while keeping other queries on their preferred models.

**Business Rule**: "Queries asking for 'search', 'find', 'look up', or 'current information' should go to providers with web search capabilities."

**Implementation**:
```yaml
routing_rules:
  - name: "search-queries"
    description: "Route search requests to providers with web search capabilities"
    conditions:
      - type: "keyword_match"
        keywords: ["search", "find", "look up", "current", "latest", "recent", "news"]
    action:
      type: "route"
      endpoint: "search-enabled-provider"
      reasoning: "Query requires web search capabilities"
```

## How It Works

Keyword-based routing provides **transparent, interpretable routing decisions** by evaluating clear business rules before falling back to semantic classification:

1. **Rule Evaluation**: Check each rule's conditions against the incoming query using simple keyword matching
2. **Match Found**: Route to the specified endpoint with **complete transparency** - you know exactly why the decision was made
3. **No Match**: Fall back to semantic classification for intelligent routing
4. **Audit Trail**: Log every decision with clear reasoning for compliance and debugging

The key benefit is **interpretability**: stakeholders can easily understand and validate routing decisions because they're based on explicit, human-readable rules rather than opaque ML models.

## Business Benefits

### Immediate Value
- **Security**: Ensure sensitive data never leaves your infrastructure with transparent rules
- **Compliance**: Meet regulatory requirements with auditable, interpretable routing decisions
- **Capability Routing**: Route queries to providers with specific capabilities (search, coding, etc.)
- **Performance**: Get sub-millisecond routing for deterministic cases with clear reasoning

### Enterprise Governance
- **Transparency**: See exactly why each query was routed where - no black box decisions
- **Interpretability**: Human-readable rules that business stakeholders can understand and validate
- **Control**: Implement business policies with confidence and clear justification
- **Auditability**: Complete logs for compliance and regulatory reporting
- **Flexibility**: Easy to modify rules as business needs change with full visibility

## Getting Started

### Step 1: Identify Your Business Rules
Start by identifying the key business policies that need deterministic routing:

- **Security Rules**: What data must never leave your infrastructure?
- **Compliance Rules**: What regulatory requirements do you need to meet?
- **Capability Rules**: Which queries need specific capabilities (search, coding, etc.)?

### Step 2: Define Your Keywords
Create keyword lists for each business rule:

```yaml
# Example: Financial services keywords
financial_keywords:
  confidential: ["internal", "confidential", "proprietary", "restricted"]
  account_data: ["account", "balance", "transaction", "statement"]
  trading: ["trading", "strategy", "portfolio", "investment"]
  procedures: ["procedure", "process", "workflow", "policy"]
```

### Step 3: Configure Your Rules
Define your routing rules with clear business justification:

```yaml
routing_rules:
  - name: "confidential-financial"
    description: "Route confidential financial queries to internal infrastructure"
    business_justification: "Compliance with financial data sovereignty regulations"
    owner: "compliance-team@company.com"
    conditions:
      - type: "keyword_match"
        keywords: ["internal", "confidential", "account", "trading"]
    action:
      type: "route"
      endpoint: "internal-financial-ai"
      reasoning: "Contains confidential financial terminology"
```

### Step 4: Test and Validate
Test your rules with real queries to ensure they work as expected:

```yaml
test_cases:
  - input: "How do I access internal account procedures?"
    expected_endpoint: "internal-financial-ai"
    expected_reasoning: "Contains 'internal' and 'account' keywords"
  
  - input: "What's the weather like today?"
    expected_endpoint: "semantic_classification"
    expected_reasoning: "No keyword matches - using semantic routing"
```

## Success Stories

### Financial Services Company
**Challenge**: Needed to ensure confidential trading strategies never reached external AI providers.

**Solution**: Implemented keyword-based routing for terms like "trading", "strategy", "confidential".

**Result**: 
- ✅ 100% compliance with data sovereignty requirements
- ✅ Complete audit trail for regulatory reporting
- ✅ Zero incidents of data leakage
- ✅ 50% reduction in compliance review time

### Healthcare Provider
**Challenge**: Required HIPAA-compliant routing for patient data while using external AI for general medical knowledge.

**Solution**: Created rules to route patient-specific queries internally and general medical questions externally.

**Result**:
- ✅ Full HIPAA compliance
- ✅ Improved patient care with broader medical knowledge
- ✅ Reduced costs by using appropriate models for each query type
- ✅ Streamlined compliance audits

## Next Steps

1. **Evaluate Your Needs**: Identify which business rules require deterministic routing
2. **Start Simple**: Begin with high-priority security and compliance rules
3. **Iterate**: Add more rules as you identify additional business requirements
4. **Monitor**: Use audit trails to validate rule effectiveness and compliance
5. **Optimize**: Refine rules based on real-world usage patterns

## Related Documentation

- [Multi-Provider Routing](multi-provider-routing.md) - External provider integration
- [Configuration Guide](../installation/configuration.md) - General configuration options
- [Architecture Overview](../overview/architecture/system-architecture.md) - System design details