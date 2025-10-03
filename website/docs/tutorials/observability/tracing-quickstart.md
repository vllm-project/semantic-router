# Quick Start: Distributed Tracing

Get started with distributed tracing in 5 minutes.

## Step 1: Enable Tracing

Edit your `config.yaml`:

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "stdout"
    sampling:
      type: "always_on"
    resource:
      service_name: "vllm-semantic-router"
      deployment_environment: "development"
```

## Step 2: Start the Router

```bash
./semantic-router --config config.yaml
```

## Step 3: Send a Test Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## Step 4: View Traces

Check your console output for JSON trace spans:

```json
{
  "Name": "semantic_router.request.received",
  "Attributes": [
    {"Key": "request.id", "Value": "req-123"},
    {"Key": "http.method", "Value": "POST"}
  ]
}
```

## What's Next?

### Production Deployment with Jaeger

1. **Start Jaeger**:

   ```bash
   docker run -d -p 4317:4317 -p 16686:16686 \
     jaegertracing/all-in-one:latest
   ```

2. **Update config.yaml**:

   ```yaml
   observability:
     tracing:
       enabled: true
       exporter:
         type: "otlp"
         endpoint: "localhost:4317"
         insecure: true
       sampling:
         type: "probabilistic"
         rate: 0.1
   ```

3. **View traces**: http://localhost:16686

### Key Metrics to Monitor

- **Classification Time**: `classification.time_ms` attribute
- **Cache Hit Rate**: Filter by `cache.hit = true`
- **Security Blocks**: Filter by `security.action = blocked`
- **Routing Decisions**: `routing.strategy` and `routing.reason` attributes

### Common Use Cases

**Find slow requests:**

```
Min Duration: 1s
Service: vllm-semantic-router
```

**Debug specific request:**

```
Tags: request.id = req-abc-123
```

**Analyze classification performance:**

```
Operation: semantic_router.classification
Sort by: Duration (desc)
```

## Learn More

- [Full Distributed Tracing Guide](./distributed-tracing.md)
- [Configuration Reference](../../installation/configuration.md)
- [Observability Overview](./observability.md)
