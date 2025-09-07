# Router API Reference

The Semantic Router provides a gRPC-based API that integrates seamlessly with Envoy's External Processing (ExtProc) protocol. This document covers the API endpoints, request/response formats, and integration patterns.

## API Overview

The Semantic Router operates as an ExtProc server that processes HTTP requests through Envoy Proxy. It doesn't expose direct REST endpoints but rather processes OpenAI-compatible API requests routed through Envoy.

### Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Envoy
    participant Router
    participant Backend
    
    Client->>Envoy: POST /v1/chat/completions
    Envoy->>Router: ExtProc Request
    Router->>Router: Classify & Route
    Router->>Envoy: Routing Headers
    Envoy->>Backend: Forward to Selected Model
    Backend->>Envoy: Model Response
    Envoy->>Router: Response Processing
    Router->>Envoy: Enhanced Response
    Envoy->>Client: Final Response
```

## OpenAI API Compatibility

The router processes standard OpenAI API requests:

### Chat Completions Endpoint

**Endpoint:** `POST /v1/chat/completions`

#### Request Format

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user", 
      "content": "What is the derivative of x^2?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations"
      }
    }
  ]
}
```

#### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion", 
  "created": 1677858242,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The derivative of x^2 is 2x."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  },
  "routing_metadata": {
    "selected_model": "mathematics",
    "confidence": 0.96,
    "processing_time_ms": 15,
    "cache_hit": false,
    "security_checks": {
      "pii_detected": false,
      "jailbreak_detected": false
    }
  }
}
```

## Routing Headers

The router adds metadata headers to both requests and responses:

### Request Headers (Added by Router)

| Header | Description | Example |
|--------|-------------|---------|
| `x-semantic-destination-endpoint` | Backend endpoint selected | `endpoint1` |
| `x-selected-model` | Model category determined | `mathematics` |
| `x-routing-confidence` | Classification confidence | `0.956` |
| `x-request-id` | Unique request identifier | `req-abc123` |
| `x-cache-status` | Cache hit/miss status | `miss` |

### Response Headers (Added by Router)

| Header | Description | Example |
|--------|-------------|---------|
| `x-processing-time` | Total processing time (ms) | `45` |
| `x-classification-time` | Classification time (ms) | `12` |
| `x-security-checks` | Security check results | `pii:false,jailbreak:false` |
| `x-tools-selected` | Number of tools selected | `2` |

## Health Check API

The router provides health check endpoints for monitoring:

### Router Health

**Endpoint:** `GET http://localhost:8080/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "models": {
    "category_classifier": "loaded",
    "pii_detector": "loaded", 
    "jailbreak_guard": "loaded"
  },
  "cache": {
    "status": "healthy",
    "entries": 1247,
    "hit_rate": 0.73
  },
  "endpoints": {
    "endpoint1": "healthy",
    "endpoint2": "healthy", 
    "endpoint3": "degraded"
  }
}
```

## Metrics API

Prometheus-compatible metrics are available:

**Endpoint:** `GET http://localhost:9090/metrics`

### Key Metrics

```prometheus
# Request metrics
semantic_router_requests_total{endpoint="endpoint1",category="mathematics",status="success"} 1247
semantic_router_request_duration_seconds{endpoint="endpoint1"} 0.045

# Classification metrics
semantic_router_classification_accuracy{category="mathematics"} 0.94
semantic_router_classification_duration_seconds 0.012

# Cache metrics  
semantic_router_cache_hit_ratio 0.73
semantic_router_cache_size 1247

# Security metrics
semantic_router_pii_detections_total{action="block"} 23
semantic_router_jailbreak_attempts_total{action="block"} 5
```

### Reasoning Mode Metrics

The router exposes dedicated Prometheus counters to monitor reasoning mode decisions and template usage across model families. These metrics are emitted by the router and can be scraped by your Prometheus server.

- `llm_reasoning_decisions_total{category, model, enabled, effort}`
  - Description: Count of reasoning decisions made per category and selected model, with whether reasoning was enabled and the applied effort level.
  - Labels:
    - category: category name determined during routing
    - model: final selected model for the request
    - enabled: "true" or "false" depending on the decision
    - effort: effort level used when enabled (e.g., low|medium|high)

- `llm_reasoning_template_usage_total{family, param}`
  - Description: Count of times a model-family-specific template parameter was applied to requests.
  - Labels:
    - family: normalized model family (e.g., qwen3, deepseek, gpt-oss, gpt)
    - param: name of the template knob applied (e.g., enable_thinking, thinking, reasoning_effort)

- `llm_reasoning_effort_usage_total{family, effort}`
  - Description: Count of times a reasoning effort level was set for a given model family.
  - Labels:
    - family: normalized model family
    - effort: effort level (e.g., low|medium|high)

Example PromQL:

```prometheus
# Reasoning decisions by category and model (last 5m)
sum by (category, model, enabled, effort) (
  rate(llm_reasoning_decisions_total[5m])
)

# Template usage by model family and parameter (last 5m)
sum by (family, param) (
  rate(llm_reasoning_template_usage_total[5m])
)

# Effort distribution by model family (last 5m)
sum by (family, effort) (
  rate(llm_reasoning_effort_usage_total[5m])
)
```

### Cost and Routing Metrics

The router exposes additional metrics for cost accounting and routing decisions.

- `llm_model_cost_total{model, currency}`
  - Description: Total accumulated cost attributed to each model (computed from token usage and per-1M pricing), labeled by currency.
  - Labels:
    - model: model name used for the request
    - currency: currency code (e.g., "USD")

- `llm_routing_reason_codes_total{reason_code, model}`
  - Description: Count of routing decisions by reason code and selected model.
  - Labels:
    - reason_code: why a routing decision happened (e.g., auto_routing, model_specified, pii_policy_alternative_selected)
    - model: final selected model

Example PromQL:

```prometheus
# Cost by model and currency over the last hour
sum by (model, currency) (increase(llm_model_cost_total[1h]))

# Or, if you only use USD, a common query is:
sum by (model) (increase(llm_model_cost_total{currency="USD"}[1h]))

# Routing decisions by reason code over the last 15 minutes
sum by (reason_code) (increase(llm_routing_reason_codes_total[15m]))
```

### Pricing Configuration

Provide per-1M pricing for your models so the router can compute request cost and emit metrics/logs.

```yaml
model_config:
  phi4:
    pricing:
      currency: USD
      prompt_per_1m: 0.07
      completion_per_1m: 0.35
  "mistral-small3.1":
    pricing:
      currency: USD
      prompt_per_1m: 0.1
      completion_per_1m: 0.3
  gemma3:27b:
    pricing:
      currency: USD
      prompt_per_1m: 0.067
      completion_per_1m: 0.267
```

Notes:
- Pricing is optional; if omitted, cost is treated as 0 and only token metrics are emitted.
- Cost is computed as: (prompt_tokens * prompt_per_1m + completion_tokens * completion_per_1m) / 1_000_000 (in the configured currency).

## gRPC ExtProc API

For direct integration with the ExtProc protocol:

### Service Definition

```protobuf
syntax = "proto3";

package envoy.service.ext_proc.v3;

service ExternalProcessor {
  rpc Process(stream ProcessingRequest) returns (stream ProcessingResponse);
}

message ProcessingRequest {
  oneof request {
    RequestHeaders request_headers = 1;
    RequestBody request_body = 2;
    ResponseHeaders response_headers = 3;
    ResponseBody response_body = 4;
  }
}

message ProcessingResponse {
  oneof response {
    RequestHeadersResponse request_headers = 1;
    RequestBodyResponse request_body = 2;
    ResponseHeadersResponse response_headers = 3;
    ResponseBodyResponse response_body = 4;
  }
}
```

### Processing Methods

#### Request Headers Processing

```go
func (r *Router) handleRequestHeaders(headers *ProcessingRequest_RequestHeaders) *ProcessingResponse {
    // Extract request metadata
    // Set up request context
    // Return continue response
}
```

#### Request Body Processing

```go
func (r *Router) handleRequestBody(body *ProcessingRequest_RequestBody) *ProcessingResponse {
    // Parse OpenAI request
    // Classify query intent
    // Run security checks
    // Select optimal model
    // Return routing headers
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "PII detected in request",
    "type": "security_violation",
    "code": "pii_detected",
    "details": {
      "entities_found": ["EMAIL", "PERSON"],
      "action_taken": "block"
    }
  }
}
```

### HTTP Status Codes

| Status | Description |
|--------|-------------|
| 200 | Success |
| 400 | Bad Request (malformed input) |
| 403 | Forbidden (security violation) |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (backend down) |

## Configuration API

### Runtime Configuration Updates

**Endpoint:** `POST /admin/config/update`

```json
{
  "classification": {
    "confidence_threshold": 0.8
  },
  "security": {
    "enable_pii_detection": true
  },
  "cache": {
    "ttl_seconds": 7200
  }
}
```

## WebSocket API (Optional)

For real-time streaming responses:

**Endpoint:** `ws://localhost:8801/v1/chat/stream`

```javascript
const ws = new WebSocket('ws://localhost:8801/v1/chat/stream');

ws.send(JSON.stringify({
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}));

ws.onmessage = function(event) {
  const chunk = JSON.parse(event.data);
  console.log(chunk.choices[0].delta.content);
};
```

## Client Libraries

### Python Client

```python
import requests

class SemanticRouterClient:
    def __init__(self, base_url="http://localhost:8801"):
        self.base_url = base_url
        
    def chat_completion(self, messages, model="gpt-3.5-turbo", **kwargs):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                **kwargs
            }
        )
        return response.json()
        
    def get_health(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = SemanticRouterClient()
result = client.chat_completion([
    {"role": "user", "content": "What is 2 + 2?"}
])
```

### JavaScript Client

```javascript
class SemanticRouterClient {
    constructor(baseUrl = 'http://localhost:8801') {
        this.baseUrl = baseUrl;
    }
    
    async chatCompletion(messages, model = 'gpt-3.5-turbo', options = {}) {
        const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model,
                messages,
                ...options
            })
        });
        
        return response.json();
    }
    
    async getHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
}

// Usage
const client = new SemanticRouterClient();
const result = await client.chatCompletion([
    { role: 'user', content: 'Solve x^2 + 5x + 6 = 0' }
]);
```

## Rate Limiting

The router implements rate limiting with the following headers:

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

### Rate Limit Response

```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "code": "too_many_requests",
    "details": {
      "limit": 1000,
      "window": "1h",
      "retry_after": 60
    }
  }
}
```

## Best Practices

### 1. Request Optimization

```python
# Include relevant context
messages = [
    {
        "role": "system", 
        "content": "You are a mathematics tutor."
    },
    {
        "role": "user",
        "content": "Explain derivatives in simple terms"
    }
]

# Use appropriate tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "For mathematical calculations"
        }
    }
]
```

### 2. Error Handling

```python
try:
    response = client.chat_completion(messages)
    if 'error' in response:
        handle_router_error(response['error'])
    else:
        process_response(response)
        
except requests.exceptions.Timeout:
    handle_timeout_error()
except requests.exceptions.ConnectionError:
    handle_connection_error()
```

### 3. Monitoring Integration

```python
import time

start_time = time.time()
response = client.chat_completion(messages)
duration = time.time() - start_time

# Log routing metadata
routing_info = response.get('routing_metadata', {})
logger.info(f"Request routed to {routing_info.get('selected_model')} "
           f"with confidence {routing_info.get('confidence')} "
           f"in {duration:.2f}s")
```

## Next Steps

- **[Classification API](classification.md)**: Detailed classification endpoints
- **[System Architecture](../architecture/system-architecture.md)**: System monitoring and observability
- **[Quick Start Guide](../getting-started/installation.md)**: Real-world integration examples
- **[Configuration Guide](../getting-started/configuration.md)**: Production configuration

For more advanced API usage and custom integrations, refer to the examples directory or join our community discussions.
