# Hallucination Detection

Semantic Router provides advanced hallucination detection to verify that LLM responses are grounded in the provided context. The system uses fine-tuned ModernBERT token classifiers to identify claims that are not supported by retrieval results or tool outputs.

## Overview

The hallucination detection system:

- **Verifies** LLM responses against provided context (RAG results, tool outputs)
- **Identifies** unsupported claims at the token level
- **Provides** detailed explanations using NLI (Natural Language Inference)
- **Warns or blocks** when hallucinations are detected
- **Integrates** seamlessly with RAG and tool-calling workflows

## How It Works

Hallucination detection operates in a three-stage pipeline:

1. **Fact-Check Classification**: Determines if a query requires fact verification (factual questions vs. creative/opinion-based)
2. **Token-Level Detection**: Analyzes LLM responses to identify unsupported claims
3. **NLI Explanation** (optional): Provides detailed reasoning for hallucinated spans

## Configuration

### Global Model Configuration

First, configure the hallucination detection models in `router-defaults.yaml`:

```yaml
# router-defaults.yaml
# Hallucination mitigation configuration
# Disabled by default - enable in decisions via hallucination plugin
hallucination_mitigation:
  enabled: false

  # Fact-check classifier: determines if a prompt needs fact verification
  fact_check_model:
    model_id: "models/mom-halugate-sentinel"
    threshold: 0.6
    use_cpu: true

  # Hallucination detector: verifies if LLM response is grounded in context
  hallucination_model:
    model_id: "models/mom-halugate-detector"
    threshold: 0.8
    use_cpu: true

  # NLI model: provides explanations for hallucinated spans
  nli_model:
    model_id: "models/mom-halugate-explainer"
    threshold: 0.9
    use_cpu: true
```

### Enable Hallucination Detection in Decisions

Enable hallucination detection per decision using the `hallucination` plugin:

```yaml
# config.yaml
decisions:
  - name: "general_decision"
    description: "General questions with fact-checking"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general"
        - type: "fact_check"
          name: "needs_fact_check"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true  # Enable NLI for detailed explanations
          # Action when hallucination detected: "header", "body", "block", or "none"
          hallucination_action: "header"
          # Action when fact-check needed but no tool context: "header", "body", or "none"
          unverified_factual_action: "header"
          # Include detailed info (confidence, spans) in body warning
          include_hallucination_details: true
```

### Plugin Configuration Options

| Option                          | Values                              | Description                                              |
|---------------------------------|-------------------------------------|----------------------------------------------------------|
| `enabled`                       | `true`, `false`                     | Enable/disable hallucination detection for this decision |
| `use_nli`                       | `true`, `false`                     | Use NLI model for detailed explanations                 |
| `hallucination_action`          | `header`, `body`, `block`, `none`   | Action when hallucination detected                       |
| `unverified_factual_action`     | `header`, `body`, `none`            | Action when fact-check needed but no context available   |
| `include_hallucination_details` | `true`, `false`                     | Include confidence and spans in response body            |

### Action Modes

| Action   | Behavior                                      | Use Case                              |
|----------|-----------------------------------------------|---------------------------------------|
| `header` | Adds warning headers, allows response         | Development, monitoring               |
| `body`   | Adds warning in response body, allows response| User-facing warnings                  |
| `block`  | Returns error, blocks response                | Production, high-stakes applications  |
| `none`   | No action, only logs                          | Silent monitoring                     |

## How Hallucination Detection Works

When a request is processed:

1. **Fact-Check Classification**: The sentinel model determines if the query needs fact verification
2. **Context Extraction**: Tool results or RAG context are captured from the LLM response
3. **Hallucination Detection**: If context is available, the detector analyzes the response
4. **Action**: Based on configuration, the system adds headers, modifies body, or blocks the response

Response headers when hallucination detected:

```http
X-Hallucination-Detected: true
X-Hallucination-Confidence: 0.85
X-Unsupported-Spans: "Paris was founded in 1492"
```

## Use Cases

### RAG (Retrieval Augmented Generation)

Verify that LLM responses are grounded in retrieved documents:

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: false
      hallucination_action: "header"
      unverified_factual_action: "header"
```

**Example**: A customer support bot retrieves documentation and generates answers. Hallucination detection ensures responses don't include information not present in the docs.

### Tool-Calling Workflows

Validate that LLM responses accurately reflect tool outputs:

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: true
      hallucination_action: "block"
      unverified_factual_action: "header"
      include_hallucination_details: true
```

**Example**: An AI agent calls a database query tool. Hallucination detection prevents the agent from fabricating data not returned by the query.
