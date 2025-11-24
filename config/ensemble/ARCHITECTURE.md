# Ensemble Service Architecture

## Overview

The ensemble orchestration feature is implemented as an independent OpenAI-compatible API server that runs alongside the semantic router. This design provides clean separation of concerns and allows the ensemble service to scale independently.

## Architecture Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request
       │
       ▼
┌─────────────────────────────┐
│ Semantic Router (Port 8080) │
│  ┌─────────────────────┐    │
│  │ ExtProc (Port 50051)│    │
│  └─────────────────────┘    │
│  ┌─────────────────────┐    │
│  │  API Server         │    │
│  └─────────────────────┘    │
└─────────────┬───────────────┘
              │
              │ (Optional: Route to Ensemble)
              │
              ▼
┌──────────────────────────────┐
│ Ensemble Service (Port 8081) │
│  ┌──────────────────────┐    │
│  │ /v1/chat/completions │    │
│  │ /health              │    │
│  └──────────────────────┘    │
└──────────┬───────────────────┘
           │
           │ Parallel Queries
           │
    ┌──────┴──────┬──────────┬──────────┐
    ▼             ▼          ▼          ▼
┌────────┐   ┌────────┐  ┌────────┐  ┌────────┐
│Model A │   │Model B │  │Model C │  │Model N │
│:8001   │   │:8002   │  │:8003   │  │:800N   │
└────────┘   └────────┘  └────────┘  └────────┘
           │
           │ Responses
           │
           ▼
    Aggregation Engine
    (Voting, Weighted, etc.)
           │
           ▼
    Aggregated Response
```

## Components

### 1. Semantic Router (Existing)

- **ExtProc Server** (Port 50051): Envoy external processor for request/response filtering
- **API Server** (Port 8080): Classification and system prompt APIs
- **Metrics Server** (Port 9190): Prometheus metrics

### 2. Ensemble Service (New)

- **Independent HTTP Server** (Port 8081, configurable)
- **OpenAI-Compatible API**: `/v1/chat/completions` endpoint
- **Health Check**: `/health` endpoint
- **Started Automatically**: When `ensemble.enabled: true` in config

### 3. Model Endpoints

- **Multiple Backends**: Each with OpenAI-compatible API
- **Configured in YAML**: Via `endpoint_mappings`
- **Parallel Queries**: Executed concurrently with semaphore control

## Request Flow

### 1. Direct Ensemble Request

Client directly queries the ensemble service:

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "x-ensemble-enable: true" \
  -H "x-ensemble-models: model-a,model-b,model-c" \
  -H "x-ensemble-strategy: voting" \
  -d '{"messages":[...]}'
```

**Flow:**
1. Client → Ensemble Service (Port 8081)
2. Ensemble Service → Model Endpoints (Parallel)
3. Model Endpoints → Ensemble Service (Responses)
4. Ensemble Service → Aggregation → Client (Final Response)

### 2. Via Semantic Router (Future Enhancement)

Semantic router could route to ensemble service based on headers/config:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "x-ensemble-enable: true" \
  -H "x-ensemble-models: model-a,model-b,model-c" \
  -d '{"messages":[...]}'
```

**Flow:**
1. Client → Semantic Router (Port 8080)
2. Router detects ensemble header → Routes to Ensemble Service
3. Ensemble Service → Model Endpoints (Parallel)
4. Ensemble Service → Router → Client

## Key Design Decisions

### Why Independent Service?

1. **Clean Separation**: ExtProc is designed for single downstream endpoint
2. **Scalability**: Ensemble service can be scaled independently
3. **Flexibility**: Can be used standalone or with semantic router
4. **Simplicity**: Each component has a single, clear responsibility
5. **Maintainability**: Clear boundaries between components

### Port Allocation

| Service | Default Port | Configurable | Purpose |
|---------|--------------|--------------|---------|
| ExtProc | 50051 | `-port` | gRPC ExtProc server |
| API Server | 8080 | `-api-port` | Classification APIs |
| Ensemble | 8081 | `-ensemble-port` | Ensemble orchestration |
| Metrics | 9190 | `-metrics-port` | Prometheus metrics |

### Configuration

Ensemble service reads configuration from the same `config.yaml`:

```yaml
ensemble:
  enabled: true  # Start ensemble service
  default_strategy: "voting"
  default_min_responses: 2
  timeout_seconds: 30
  max_concurrent_requests: 10
  endpoint_mappings:
    model-a: "http://localhost:8001/v1/chat/completions"
    model-b: "http://localhost:8002/v1/chat/completions"
    model-c: "http://localhost:8003/v1/chat/completions"
```

## Deployment Scenarios

### Scenario 1: Standalone Ensemble

Deploy only the ensemble service:

```bash
./bin/router -config=config/ensemble-only.yaml
```

Config with all other features disabled, only ensemble enabled.

### Scenario 2: Integrated with Semantic Router

Deploy all services together (default):

```bash
./bin/router -config=config/config.yaml
```

All services start based on their enabled flags.

### Scenario 3: Scaled Ensemble

Run multiple ensemble service instances:

```bash
# Instance 1
./bin/router -config=config1.yaml -ensemble-port=8081

# Instance 2  
./bin/router -config=config2.yaml -ensemble-port=8082
```

Load balancer distributes requests across instances.

## API Specification

### POST /v1/chat/completions

OpenAI-compatible endpoint with ensemble extensions.

#### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `x-ensemble-enable` | Yes | Must be "true" |
| `x-ensemble-models` | Yes | Comma-separated model names |
| `x-ensemble-strategy` | No | Aggregation strategy (default from config) |
| `x-ensemble-min-responses` | No | Minimum responses required (default from config) |
| `Authorization` | No | Forwarded to model endpoints |

#### Request Body

Standard OpenAI chat completion request:

```json
{
  "model": "ensemble",
  "messages": [
    {"role": "user", "content": "Your question"}
  ]
}
```

#### Response Headers

| Header | Description |
|--------|-------------|
| `x-vsr-ensemble-used` | "true" if ensemble was used |
| `x-vsr-ensemble-models-queried` | Number of models queried |
| `x-vsr-ensemble-responses-received` | Number of successful responses |

#### Response Body

Standard OpenAI chat completion response with aggregated content.

### GET /health

Health check endpoint.

#### Response

```json
{
  "status": "healthy",
  "service": "ensemble"
}
```

## Aggregation Strategies

### Voting

Parses responses and selects most common answer:

```yaml
x-ensemble-strategy: voting
```

Best for: Classification, multiple choice questions

### Weighted

Selects response with highest confidence:

```yaml
x-ensemble-strategy: weighted
```

Best for: Models with different reliability profiles

### First Success

Returns first valid response:

```yaml
x-ensemble-strategy: first_success
```

Best for: Latency-sensitive applications

### Score Averaging

Balances confidence and latency:

```yaml
x-ensemble-strategy: score_averaging
```

Best for: Balanced quality and speed

## Error Handling

### Insufficient Responses

If fewer than `min_responses` succeed:

```json
{
  "error": "Ensemble orchestration failed: insufficient responses: got 1, required 2"
}
```

### Invalid Configuration

If model not in endpoint_mappings:

```json
{
  "error": "endpoint not found for model: model-x"
}
```

### Timeout

If requests exceed timeout:

```json
{
  "error": "HTTP request failed: context deadline exceeded"
}
```

## Monitoring

### Logs

Ensemble service logs:
- Request details (models, strategy, min responses)
- Execution results (queried, received, strategy used)
- Errors and failures

### Metrics

Future enhancement: Prometheus metrics for:
- Request count per strategy
- Response latency per model
- Success/failure rates
- Aggregation time

## Security Considerations

1. **Authentication**: Headers forwarded to model endpoints
2. **Network**: Use HTTPS in production
3. **Rate Limiting**: Apply at load balancer level
4. **Endpoint Validation**: Only configured endpoints are queried
5. **Timeout Protection**: Prevents resource exhaustion

## Future Enhancements

1. **Semantic Router Integration**: Automatic routing to ensemble service
2. **Streaming Support**: SSE for streaming responses
3. **Advanced Reranking**: Separate model for ranking responses
4. **Caching**: Cache ensemble results
5. **Metrics**: Comprehensive Prometheus metrics
6. **Circuit Breaker**: Automatic endpoint failure detection
7. **Load Balancing**: Intelligent distribution across model endpoints
