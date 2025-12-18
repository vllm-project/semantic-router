# MCP Based Routing

This guide shows you how to implement custom classification logic using the Model Context Protocol (MCP). MCP routing lets you integrate external services, LLMs, or custom business logic for classification decisions while keeping your data private and your routing logic extensible.

## Key Advantages

- **Baseline/High Accuracy**: Use powerful LLMs (GPT-4, Claude) for classification with in-context learning
- **Extensible**: Easily integrate custom classification logic without modifying router code
- **Private**: Keep classification logic and data in your own infrastructure
- **Flexible**: Combine LLM reasoning with business rules, user context, and external data

## What Problem Does It Solve?

Built-in classifiers are limited to predefined models and logic. MCP routing enables:

- **LLM-powered classification**: Use GPT-4/Claude for complex, nuanced categorization
- **In-context learning**: Provide examples and context to improve classification accuracy
- **Custom business logic**: Implement routing rules based on user tier, time, location, history
- **External data integration**: Query databases, APIs, feature flags during classification
- **Rapid experimentation**: Update classification logic without redeploying router

## When to Use

- **High-accuracy requirements** where LLM-based classification outperforms BERT/embeddings
- **Complex domains** needing nuanced understanding beyond keyword/embedding matching
- **Custom business rules** (user tiers, A/B tests, time-based routing)
- **Private/sensitive data** where classification must stay in your infrastructure
- **Rapid iteration** on classification logic without code changes

## Configuration

Configure MCP classifier in your `config.yaml`:

```yaml
classifier:
  # Disable in-tree classifier
  category_model:
    model_id: ""
  
  # Enable MCP classifier
  mcp_category_model:
    enabled: true
    transport_type: "http"
    url: "http://localhost:8090/mcp"
    threshold: 0.6
    timeout_seconds: 30
    # tool_name: "classify_text"  # Optional: auto-discovers if not specified

categories: []  # Categories loaded from MCP server

default_model: openai/gpt-oss-20b

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

model_config:
  openai/gpt-oss-20b:
    reasoning_family: gpt-oss
    preferred_endpoints: [endpoint1]
```

## How It Works

1. **Startup**: Router connects to MCP server and calls `list_categories` tool
2. **Category Loading**: MCP returns categories, system prompts, and descriptions
3. **Classification**: For each request, router calls `classify_text` tool
4. **Routing**: MCP response includes category, model, and reasoning settings

### MCP Response Format

**list_categories**:

```json
{
  "categories": ["math", "science", "technology"],
  "category_system_prompts": {
    "math": "You are a mathematics expert...",
    "science": "You are a science expert..."
  },
  "category_descriptions": {
    "math": "Mathematical and computational queries",
    "science": "Scientific concepts and queries"
  }
}
```

**classify_text**:

```json
{
  "class": 3,
  "confidence": 0.85,
  "model": "openai/gpt-oss-20b",
  "use_reasoning": true
}
```

## Example MCP Server

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ClassifyRequest(BaseModel):
    text: str

@app.post("/mcp/list_categories")
def list_categories():
    return {
        "categories": ["math", "science", "general"],
        "category_system_prompts": {
            "math": "You are a mathematics expert.",
            "science": "You are a science expert.",
            "general": "You are a helpful assistant."
        }
    }

@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    # Custom classification logic
    if "equation" in request.text or "solve" in request.text:
        return {
            "class": 0,  # math
            "confidence": 0.9,
            "model": "openai/gpt-oss-20b",
            "use_reasoning": True
        }
    return {
        "class": 2,  # general
        "confidence": 0.7,
        "model": "openai/gpt-oss-20b",
        "use_reasoning": False
    }
```

## Example Requests

```bash
# Math query (MCP decides routing)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Solve the equation: 2x + 5 = 15"}]
  }'
```

## Benefits

- **Custom Logic**: Implement domain-specific classification rules
- **Dynamic Routing**: MCP decides model and reasoning per query
- **Centralized Control**: Manage routing logic in external service
- **Scalability**: Scale classification independently from router
- **Integration**: Connect to existing ML infrastructure

## Real-World Use Cases

### 1. Complex Domain Classification (High Accuracy)

**Problem**: Nuanced legal/medical queries need better accuracy than BERT/embeddings
**Solution**: MCP uses GPT-4 with in-context examples for classification
**Impact**: 98% accuracy vs 85% with BERT, baseline for quality comparison

### 2. Proprietary Classification Logic (Private)

**Problem**: Classification logic contains trade secrets, can't use external services
**Solution**: MCP server runs in private VPC, keeps all logic and data internal
**Impact**: Full data privacy, no external API calls

### 3. Custom Business Rules (Extensible)

**Problem**: Need to route based on user tier, location, time, A/B tests
**Solution**: MCP combines LLM classification with database queries and business logic
**Impact**: Flexible routing without modifying router code

### 4. Rapid Experimentation (Extensible)

**Problem**: Data science team needs to test new classification approaches daily
**Solution**: MCP server updated independently, router unchanged
**Impact**: Deploy new classification logic in minutes vs days

### 5. Multi-Tenant Platform (Extensible + Private)

**Problem**: Each customer needs custom classification, data must stay isolated
**Solution**: MCP loads tenant-specific models/rules, enforces data isolation
**Impact**: 1000+ tenants with custom logic, full data privacy

### 6. Hybrid Approach (High Accuracy + Extensible)

**Problem**: Need LLM accuracy for edge cases, fast routing for common queries
**Solution**: MCP uses cached responses for common patterns, LLM for novel queries
**Impact**: 95% cache hit rate, LLM accuracy on long tail

## Advanced MCP Server Examples

### Context-Aware Classification

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest, user_id: str = Header(None)):
    # Check user history
    user_history = get_user_history(user_id)

    # Adjust classification based on context
    if user_history.is_premium:
        return {
            "class": 0,
            "confidence": 0.95,
            "model": "openai/gpt-4",  # Premium model
            "use_reasoning": True
        }

    # Free tier gets fast model
    return {
        "class": 0,
        "confidence": 0.85,
        "model": "openai/gpt-oss-20b",
        "use_reasoning": False
    }
```

### Time-Based Routing

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    current_hour = datetime.now().hour

    # Peak hours: use cached responses
    if 9 <= current_hour <= 17:
        return {
            "class": get_cached_category(request.text),
            "confidence": 0.9,
            "model": "fast-model",
            "use_reasoning": False
        }

    # Off-peak: enable reasoning
    return {
        "class": classify_with_ml(request.text),
        "confidence": 0.95,
        "model": "reasoning-model",
        "use_reasoning": True
    }
```

### Risk-Based Routing

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    # Calculate risk score
    risk_score = calculate_risk(request.text)

    if risk_score > 0.8:
        # High risk: human review
        return {
            "class": 999,  # Special category
            "confidence": 1.0,
            "model": "human-review-queue",
            "use_reasoning": False
        }

    # Normal routing
    return standard_classification(request.text)
```

## Benefits vs Built-in Classifiers

| Feature | Built-in | MCP |
|---------|----------|-----|
| Custom Models | ❌ | ✅ |
| Business Logic | ❌ | ✅ |
| Dynamic Updates | ❌ | ✅ |
| User Context | ❌ | ✅ |
| A/B Testing | ❌ | ✅ |
| External APIs | ❌ | ✅ |
| Latency | 5-50ms | 50-200ms |
| Complexity | Low | High |

## Performance Considerations

- **Latency**: MCP adds 50-200ms per request (network + classification)
- **Caching**: Cache MCP responses for repeated queries
- **Timeout**: Set appropriate timeout (30s default)
- **Fallback**: Configure default model when MCP unavailable
- **Monitoring**: Track MCP latency and error rates

## Reference

See [config-mcp-classifier.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/out-tree/config-mcp-classifier.yaml) for complete configuration.
