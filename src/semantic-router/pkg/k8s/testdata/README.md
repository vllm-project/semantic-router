# Test Data for CRD Converter

This directory contains test data for the Kubernetes CRD to RouterConfig converter.

## Directory Structure

```
testdata/
├── base-config.yaml          # Static base configuration (shared across all tests)
├── input/                    # Input CRD YAML files (IntelligentPool + IntelligentRoute)
│   ├── 01-basic.yaml
│   ├── 02-keyword-only.yaml
│   ├── ...
│   └── 15-keyword-embedding-domain-no-plugin.yaml
└── output/                   # Generated RouterConfig YAML files
    ├── 01-basic.yaml
    ├── 02-keyword-only.yaml
    ├── ...
    └── 15-keyword-embedding-domain-no-plugin.yaml
```

## Base Configuration

`base-config.yaml` contains static configuration that doesn't come from CRDs:

- Reasoning families (deepseek, qwen3, gpt)
- Default reasoning effort level
- BERT model configuration
- Semantic cache settings
- Tools configuration
- Prompt guard settings
- Classifier configuration
- Router options
- Embedding models paths
- API configuration
- Observability settings

## Test Scenarios Overview

| # | File | Keyword | Embedding | Domain | Plugin | Use Case |
|---|------|---------|-----------|--------|--------|----------|
| 1 | 01-basic.yaml | ✓ | ✓ | ✓ | ✓ | Basic comprehensive example |
| 2 | 02-keyword-only.yaml | ✓ | ✗ | ✗ | ✗ | FAQ detection, greetings |
| 3 | 03-embedding-only.yaml | ✗ | ✓ | ✗ | ✗ | Customer support, technical issues |
| 4 | 04-domain-only.yaml | ✗ | ✗ | ✓ | ✗ | STEM queries, subject routing |
| 5 | 05-keyword-embedding.yaml | ✓ | ✓ | ✗ | ✗ | Urgent support with semantics |
| 6 | 06-keyword-domain.yaml | ✓ | ✗ | ✓ | ✗ | Academic homework assistance |
| 7 | 07-domain-embedding.yaml | ✗ | ✓ | ✓ | ✗ | Research queries by domain |
| 8 | 08-keyword-embedding-domain.yaml | ✓ | ✓ | ✓ | ✗ | Comprehensive tech support |
| 9 | 09-keyword-plugin.yaml | ✓ | ✗ | ✗ | ✓ | FAQ with caching |
| 10 | 10-embedding-plugin.yaml | ✗ | ✓ | ✗ | ✓ | PII-protected queries |
| 11 | 11-domain-plugin.yaml | ✗ | ✗ | ✓ | ✓ | Legal advice with disclaimers |
| 12 | 12-keyword-embedding-plugin.yaml | ✓ | ✓ | ✗ | ✓ | Security queries with protection |
| 13 | 13-keyword-domain-plugin.yaml | ✓ | ✗ | ✓ | ✓ | Medical queries with PII |
| 14 | 14-domain-embedding-plugin.yaml | ✗ | ✓ | ✓ | ✓ | Financial advice with protection |
| 15 | 15-keyword-embedding-domain-plugin.yaml | ✓ | ✓ | ✓ | ✓ | Enterprise compliance (full) |
| 16 | 16-keyword-embedding-domain-no-plugin.yaml | ✓ | ✓ | ✓ | ✗ | Educational tutorials |

## Test Scenarios Details

### Signal Type Combinations (No Plugins)

1. **02-keyword-only.yaml** - Only keyword signals
   - Use case: FAQ detection, greeting responses
   - Signals: urgent, greeting keywords

2. **03-embedding-only.yaml** - Only embedding signals
   - Use case: Customer support, technical issue detection
   - Signals: customer_support, technical_issue embeddings

3. **04-domain-only.yaml** - Only domain signals
   - Use case: STEM queries, subject-specific routing
   - Signals: math, physics, computer_science, chemistry domains

4. **05-keyword-embedding.yaml** - Keyword + Embedding
   - Use case: Urgent support requests with semantic matching
   - Signals: urgent keywords + support_request embeddings

5. **06-keyword-domain.yaml** - Keyword + Domain
   - Use case: Academic homework assistance
   - Signals: homework keywords + math/physics/chemistry domains

6. **07-domain-embedding.yaml** - Domain + Embedding
   - Use case: Research queries in specific domains
   - Signals: research_question embeddings + biology/chemistry/physics domains

7. **08-keyword-embedding-domain.yaml** - All three signal types
   - Use case: Comprehensive technical support routing
   - Signals: urgent keywords + technical_help embeddings + CS/engineering/math domains

### Signal Type Combinations (With Plugins)

8. **09-keyword-plugin.yaml** - Keyword + Plugins
   - Use case: FAQ with aggressive caching
   - Plugins: semantic-cache, header_mutation

9. **10-embedding-plugin.yaml** - Embedding + Plugins
   - Use case: PII-protected sensitive data handling
   - Plugins: pii (redaction), jailbreak protection

10. **11-domain-plugin.yaml** - Domain + Plugins
    - Use case: Legal advice with disclaimers
    - Plugins: system_prompt, semantic-cache

11. **12-keyword-embedding-plugin.yaml** - Keyword + Embedding + Plugins
    - Use case: Security queries with protection
    - Plugins: jailbreak, system_prompt, header_mutation

12. **13-keyword-domain-plugin.yaml** - Keyword + Domain + Plugins
    - Use case: Medical queries with PII protection
    - Plugins: pii (hash mode), system_prompt, semantic-cache

13. **14-domain-embedding-plugin.yaml** - Domain + Embedding + Plugins
    - Use case: Financial advice with comprehensive protection
    - Plugins: pii, system_prompt, jailbreak, semantic-cache

14. **15-keyword-embedding-domain-plugin.yaml** - Keyword + Embedding + Domain + Plugins
    - Use case: Enterprise compliance and legal queries with full protection
    - Signals: compliance/confidential keywords + business_analysis/legal_review embeddings + business/law/economics domains
    - Plugins: pii (hash/mask modes), jailbreak, system_prompt, semantic-cache, header_mutation
    - Multiple decisions with different plugin configurations

15. **16-keyword-embedding-domain-no-plugin.yaml** - All signals, no plugins
    - Use case: Educational tutorials across multiple domains
    - Signals: tutorial keywords + learning_intent embeddings + CS/math/engineering domains
    - Multiple decisions with different priorities

## Plugin Types Used

- **semantic-cache**: Cache responses for similar queries
- **pii**: Detect and redact/mask/hash PII entities
- **jailbreak**: Detect and block jailbreak attempts
- **system_prompt**: Inject custom system prompts
- **header_mutation**: Add custom headers to requests

## Running Tests

```bash
cd src/semantic-router
go test ./pkg/k8s -v -run TestConverterWithTestData
```

This will:

1. Load `base-config.yaml` as the static configuration base
2. Parse each input YAML file (IntelligentPool + IntelligentRoute)
3. Convert CRDs to RouterConfig format
4. Merge static base config with dynamic CRD-derived config
5. Generate output YAML files in `testdata/output/`
6. Validate that output can be unmarshaled correctly

## Output Structure

Each generated output file contains:

- **Static parts** (from base-config.yaml):
  - embedding_models, bert_model, classifier, prompt_guard
  - semantic_cache, observability, api, tools
  - reasoning_families, default_reasoning_effort
  
- **Dynamic parts** (from CRDs):
  - keyword_rules (from signals.keywords)
  - embedding_rules (from signals.embeddings)
  - categories (from signals.domains)
  - decisions (from decisions)
  - model_config (from IntelligentPool.models)
  - default_model (from IntelligentPool.defaultModel)
