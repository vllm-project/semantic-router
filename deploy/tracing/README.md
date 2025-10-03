# Distributed Tracing Deployment Example

This directory contains an example deployment configuration for testing distributed tracing with Jaeger.

## Quick Start

1. **Start the services**:

```bash
docker-compose -f ../docker-compose.tracing.yaml up -d
```

2. **Access the UIs**:

- Jaeger UI: http://localhost:16686
- Grafana: http://localhost:3000
- Router API: http://localhost:8080

3. **Send test requests**:

```bash
# Example request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

4. **View traces in Jaeger**:

- Navigate to http://localhost:16686
- Select service: `vllm-semantic-router`
- Click "Find Traces"

## Configuration

The router is configured with:

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "otlp"
      endpoint: "jaeger:4317"
      insecure: true
    sampling:
      type: "always_on"
    resource:
      service_name: "vllm-semantic-router"
```

## Services

### Jaeger

- **OTLP gRPC**: Port 4317
- **OTLP HTTP**: Port 4318
- **Jaeger UI**: Port 16686
- **Collector**: Port 14268

### Semantic Router

- **gRPC ExtProc**: Port 50051
- **Classification API**: Port 8080
- **Metrics**: Port 9190

### Grafana

- **Web UI**: Port 3000
- Default credentials: admin/admin
- Pre-configured with Jaeger data source

## Trace Examples

### Request Flow

```
semantic_router.request.received [2ms]
├─ semantic_router.classification [45ms]
│  └─ category: math, confidence: 0.95
├─ semantic_router.security.jailbreak_detection [12ms]
│  └─ jailbreak.detected: false
├─ semantic_router.cache.lookup [3ms]
│  └─ cache.hit: false
├─ semantic_router.routing.decision [5ms]
│  └─ selected_model: gpt-4, reasoning: true
└─ semantic_router.backend.selection [2ms]
   └─ endpoint: endpoint1
```

### Key Attributes

- `request.id`: Unique request identifier
- `category.name`: Classified category
- `routing.selected_model`: Selected model
- `reasoning.enabled`: Reasoning mode
- `cache.hit`: Cache hit status

## Stopping Services

```bash
docker-compose -f ../docker-compose.tracing.yaml down
```

To remove volumes:

```bash
docker-compose -f ../docker-compose.tracing.yaml down -v
```

## Troubleshooting

### Traces not appearing

1. Check Jaeger is running:

```bash
curl http://localhost:16686
```

2. Verify router can connect to Jaeger:

```bash
docker logs semantic-router | grep -i tracing
```

3. Check for initialization message:

```
Distributed tracing initialized (provider: opentelemetry, exporter: otlp)
```

### Router fails to start

1. Check configuration:

```bash
docker logs semantic-router
```

2. Verify Jaeger is ready:

```bash
docker logs jaeger
```

## Next Steps

- [Full Tracing Documentation](../../website/docs/tutorials/observability/distributed-tracing.md)
- [Quick Start Guide](../../website/docs/tutorials/observability/tracing-quickstart.md)
- [Configuration Reference](../../config/config.production.yaml)
