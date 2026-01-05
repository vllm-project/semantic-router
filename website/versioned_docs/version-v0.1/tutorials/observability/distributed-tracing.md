# Distributed Tracing with OpenTelemetry

This guide explains how to configure and use distributed tracing in vLLM Semantic Router for enhanced observability and debugging capabilities.

## Overview

vLLM Semantic Router implements comprehensive distributed tracing using OpenTelemetry, providing fine-grained visibility into the request processing pipeline. Tracing helps you:

- **Debug Production Issues**: Trace individual requests through the entire routing pipeline
- **Optimize Performance**: Identify bottlenecks in classification, caching, and routing
- **Monitor Security**: Track PII detection and jailbreak prevention operations
- **Analyze Decisions**: Understand routing logic and reasoning mode selection
- **Correlate Services**: Connect traces across the router and vLLM backends

## Architecture

### Trace Hierarchy

A typical request trace follows this structure:

```
semantic_router.request.received [root span]
├─ semantic_router.classification
├─ semantic_router.security.pii_detection
├─ semantic_router.security.jailbreak_detection
├─ semantic_router.cache.lookup
├─ semantic_router.routing.decision
├─ semantic_router.backend.selection
├─ semantic_router.system_prompt.injection
└─ semantic_router.upstream.request
```

### Span Attributes

Each span includes rich attributes following OpenInference conventions for LLM observability:

**Request Metadata:**

- `request.id` - Unique request identifier
- `user.id` - User identifier (if available)
- `http.method` - HTTP method
- `http.path` - Request path

**Model Information:**

- `model.name` - Selected model name
- `routing.original_model` - Original requested model
- `routing.selected_model` - Model selected by router

**Classification:**

- `category.name` - Classified category
- `classifier.type` - Classifier implementation
- `classification.time_ms` - Classification duration

**Security:**

- `pii.detected` - Whether PII was found
- `pii.types` - Types of PII detected
- `jailbreak.detected` - Whether jailbreak attempt detected
- `security.action` - Action taken (blocked, allowed)

**Routing:**

- `routing.strategy` - Routing strategy (auto, specified)
- `routing.reason` - Reason for routing decision
- `reasoning.enabled` - Whether reasoning mode enabled
- `reasoning.effort` - Reasoning effort level

**Performance:**

- `cache.hit` - Cache hit/miss status
- `cache.lookup_time_ms` - Cache lookup duration
- `processing.time_ms` - Total processing time

## Configuration

### Basic Configuration

Add the `observability.tracing` section to your `config.yaml`:

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "stdout"  # or "otlp"
      endpoint: "localhost:4317"
      insecure: true
    sampling:
      type: "always_on"  # or "probabilistic"
      rate: 1.0
    resource:
      service_name: "vllm-semantic-router"
      service_version: "v0.1.0"
      deployment_environment: "production"
```

### Configuration Options

#### Exporter Types

**stdout** - Print traces to console (development)

```yaml
exporter:
  type: "stdout"
```

**otlp** - Export to OTLP-compatible backend (production)

```yaml
exporter:
  type: "otlp"
  endpoint: "jaeger:4317"  # Jaeger, Tempo, Datadog, etc.
  insecure: true  # Use false with TLS in production
```

#### Sampling Strategies

**always_on** - Sample all requests (development/debugging)

```yaml
sampling:
  type: "always_on"
```

**always_off** - Disable sampling (emergency performance)

```yaml
sampling:
  type: "always_off"
```

**probabilistic** - Sample a percentage of requests (production)

```yaml
sampling:
  type: "probabilistic"
  rate: 0.1  # Sample 10% of requests
```

### Environment-Specific Configurations

#### Development

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
      service_name: "vllm-semantic-router-dev"
      deployment_environment: "development"
```

#### Production

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "otlp"
      endpoint: "tempo:4317"
      insecure: false  # Use TLS
    sampling:
      type: "probabilistic"
      rate: 0.1  # 10% sampling
    resource:
      service_name: "vllm-semantic-router"
      service_version: "v0.1.0"
      deployment_environment: "production"
```

## Deployment

### With Jaeger

1. **Start Jaeger** (all-in-one for testing):

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

2. **Configure Router**:

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

3. **Access Jaeger UI**: http://localhost:16686

### With Grafana Tempo

1. **Configure Tempo** (tempo.yaml):

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces
```

2. **Start Tempo**:

```bash
docker run -d --name tempo \
  -p 4317:4317 \
  -p 3200:3200 \
  -v $(pwd)/tempo.yaml:/etc/tempo.yaml \
  grafana/tempo:latest \
  -config.file=/etc/tempo.yaml
```

3. **Configure Router**:

```yaml
observability:
  tracing:
    enabled: true
    exporter:
      type: "otlp"
      endpoint: "tempo:4317"
      insecure: true
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: router-config
data:
  config.yaml: |
    observability:
      tracing:
        enabled: true
        exporter:
          type: "otlp"
          endpoint: "jaeger-collector.observability.svc:4317"
          insecure: false
        sampling:
          type: "probabilistic"
          rate: 0.1
        resource:
          service_name: "vllm-semantic-router"
          deployment_environment: "production"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-router
spec:
  template:
    spec:
      containers:
      - name: router
        image: vllm-semantic-router:latest
        env:
        - name: CONFIG_PATH
          value: /config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: router-config
```

## Usage Examples

### Viewing Traces

#### Console Output (stdout exporter)

```json
{
  "Name": "semantic_router.classification",
  "SpanContext": {
    "TraceID": "abc123...",
    "SpanID": "def456..."
  },
  "Attributes": [
    {
      "Key": "category.name",
      "Value": "math"
    },
    {
      "Key": "classification.time_ms",
      "Value": 45
    }
  ],
  "Duration": 45000000
}
```

#### Jaeger UI

1. Navigate to http://localhost:16686
2. Select service: `vllm-semantic-router`
3. Click "Find Traces"
4. View trace details and timeline

### Analyzing Performance

**Find slow requests:**

```
Service: vllm-semantic-router
Min Duration: 1s
Limit: 20
```

**Analyze classification bottlenecks:**
Filter by operation: `semantic_router.classification`
Sort by duration (descending)

**Track cache effectiveness:**
Filter by tag: `cache.hit = true`
Compare durations with cache misses

### Debugging Issues

**Find failed requests:**
Filter by tag: `error = true`

**Trace specific request:**
Filter by tag: `request.id = req-abc-123`

**Find PII violations:**
Filter by tag: `security.action = blocked`

## Trace Context Propagation

The router automatically propagates trace context using W3C Trace Context headers:

**Request headers** (extracted by router):

```
traceparent: 00-abc123-def456-01
tracestate: vendor=value
```

**Upstream headers** (injected by router):

```
traceparent: 00-abc123-ghi789-01
x-vsr-destination-endpoint: endpoint1
x-selected-model: gpt-4
```

This enables end-to-end tracing from client → router → vLLM backend.

## Performance Considerations

### Overhead

Tracing adds minimal overhead when properly configured:

- **Always-on sampling**: ~1-2% latency increase
- **10% probabilistic**: ~0.1-0.2% latency increase
- **Async export**: No blocking on span export

### Optimization Tips

1. **Use probabilistic sampling in production**

   ```yaml
   sampling:
     type: "probabilistic"
     rate: 0.1  # Adjust based on traffic
   ```

2. **Adjust sampling rate dynamically**
   - High traffic: 0.01-0.1 (1-10%)
   - Medium traffic: 0.1-0.5 (10-50%)
   - Low traffic: 0.5-1.0 (50-100%)

3. **Use batch exporters** (default)
   - Spans are batched before export
   - Reduces network overhead

4. **Monitor exporter health**
   - Watch for export failures in logs
   - Configure retry policies

## Troubleshooting

### Traces Not Appearing

1. **Check tracing is enabled**:

```yaml
observability:
  tracing:
    enabled: true
```

2. **Verify exporter endpoint**:

```bash
# Test OTLP endpoint connectivity
telnet jaeger 4317
```

3. **Check logs for errors**:

```
Failed to export spans: connection refused
```

### Missing Spans

1. **Check sampling rate**:

```yaml
sampling:
  type: "probabilistic"
  rate: 1.0  # Increase to see more traces
```

2. **Verify span creation in code**:

- Spans are created at key processing points
- Check for nil context

### High Memory Usage

1. **Reduce sampling rate**:

```yaml
sampling:
  rate: 0.01  # 1% sampling
```

2. **Verify batch exporter is working**:

- Check export interval
- Monitor queue length

## Best Practices

1. **Start with stdout in development**
   - Easy to verify tracing works
   - No external dependencies

2. **Use probabilistic sampling in production**
   - Balances visibility and performance
   - Start with 10% and adjust

3. **Set meaningful service names**
   - Use environment-specific names
   - Include version information

4. **Add custom attributes for your use case**
   - Customer IDs
   - Deployment region
   - Feature flags

5. **Monitor exporter health**
   - Track export success rate
   - Alert on high failure rates

6. **Correlate with metrics**
   - Use same service name
   - Cross-reference trace IDs in logs

## Integration with vLLM Stack

### Future Enhancements

The tracing implementation is designed to support future integration with vLLM backends:

1. **Trace context propagation** to vLLM
2. **Correlated spans** across router and engine
3. **End-to-end latency** analysis
4. **Token-level timing** from vLLM

Stay tuned for updates on vLLM integration!

## References

- [OpenTelemetry Go SDK](https://github.com/open-telemetry/opentelemetry-go)
- [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo](https://grafana.com/oss/tempo/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
