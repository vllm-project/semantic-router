# OpenTelemetry Distributed Tracing Examples

This directory contains examples demonstrating end-to-end distributed tracing with vLLM Semantic Router using OpenTelemetry.

## Overview

These examples show how to:

- **Auto-instrument OpenAI Python client** for automatic trace creation
- **Propagate trace context** from client → router → vLLM backends
- **Visualize request flows** in Jaeger UI
- **Debug performance issues** with detailed span timings
- **Correlate errors** across service boundaries

## Architecture

```
┌─────────────────┐     traceparent     ┌──────────────────┐     traceparent     ┌────────────────┐
│  OpenAI Client  │ ──────────────────> │ Semantic Router  │ ──────────────────> │  vLLM Backend  │
│  (Python App)   │     HTTP Headers    │   (ExtProc)      │    HTTP Headers     │   (Optional)   │
└─────────────────┘                     └──────────────────┘                     └────────────────┘
        │                                        │                                        │
        │                                        │                                        │
        └────────────────────────────────────────┴────────────────────────────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  Jaeger/Tempo    │
                                        │  (OTLP Collector)│
                                        └──────────────────┘
```

## Files

- **`openai_client_tracing.py`** - Python example with OpenAI client auto-instrumentation
- **`docker-compose.yml`** - Complete tracing stack with Jaeger
- **`router-config.yaml`** - Router configuration with tracing enabled
- **`requirements.txt`** - Python dependencies for the example
- **`README.md`** - This file

## Quick Start

### Prerequisites

1. **Docker and Docker Compose** installed
2. **Python 3.8+** installed
3. **Semantic Router image** built (or use from registry)

### Step 1: Start the Tracing Stack

Start Jaeger and the semantic router with tracing enabled:

```bash
# From this directory
docker-compose up -d
```

This starts:
- **Jaeger** on ports 16686 (UI) and 4317 (OTLP gRPC)
- **Semantic Router** on port 8000 with tracing enabled

Verify services are running:

```bash
docker-compose ps
```

### Step 2: Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Example

Run the Python example that makes requests to the router:

```bash
python openai_client_tracing.py
```

The example will:
1. Initialize OpenTelemetry tracing
2. Auto-instrument the OpenAI client
3. Make several example requests with different query types
4. Send traces to Jaeger

### Step 4: View Traces in Jaeger

Open the Jaeger UI in your browser:

```
http://localhost:16686
```

1. Select **Service**: `openai-client-example`
2. Click **Find Traces**
3. Click on a trace to see the detailed timeline

You should see traces with spans from:
- `openai-client-example` (the Python client)
- `vllm-semantic-router` (the router processing)
- Individual operations (classification, routing, etc.)

## Example Trace Visualization

A typical trace will show:

```
Trace: example_1_auto_routing (2.3s total)
├── openai.chat.completions.create (2.3s)
│   └── HTTP POST /v1/chat/completions (2.2s)
│       ├── semantic_router.request.received (2.2s)
│       │   ├── semantic_router.classification (45ms) [category=science]
│       │   ├── semantic_router.cache.lookup (3ms) [cache_miss=true]
│       │   ├── semantic_router.routing.decision (23ms) [selected=llama-3.1-70b]
│       │   └── semantic_router.upstream.request (2.1s)
│       │       └── vllm.generate (2.0s) [tokens=156]
```

## Configuration Options

### Environment Variables

The example supports these environment variables:

- **`SEMANTIC_ROUTER_URL`** - Router URL (default: `http://localhost:8000`)
- **`OTLP_ENDPOINT`** - OTLP collector endpoint (default: `http://localhost:4317`)
- **`OPENAI_API_KEY`** - API key (default: `dummy-key-for-local-testing`)

Example with custom configuration:

```bash
export SEMANTIC_ROUTER_URL="http://my-router:8000"
export OTLP_ENDPOINT="http://tempo:4317"
python openai_client_tracing.py
```

### Router Configuration

Edit `router-config.yaml` to customize tracing behavior:

```yaml
observability:
  tracing:
    enabled: true
    exporter:
      type: "otlp"
      endpoint: "jaeger:4317"
    sampling:
      type: "always_on"  # or "probabilistic"
      rate: 1.0          # sample rate for probabilistic
```

**Sampling strategies:**

- **`always_on`** - Sample 100% of requests (development/debugging)
- **`always_off`** - Disable sampling (emergency)
- **`probabilistic`** - Sample a percentage (production)

**Production recommendation:**
```yaml
sampling:
  type: "probabilistic"
  rate: 0.1  # Sample 10% of requests
```

## Advanced Usage

### Custom Spans

Add custom spans to track specific operations:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("custom.attribute", "value")
    # Your code here
```

### Multiple Backends

Configure the router to route to different vLLM backends and see how traces flow:

```yaml
models:
  - name: "llama-3.1-8b"
    category: "general"
    endpoints:
      - name: "backend-1"
        address: "http://vllm-1:8000"
        
  - name: "llama-3.1-70b"
    category: "reasoning"
    endpoints:
      - name: "backend-2"
        address: "http://vllm-2:8000"
```

### Alternative Tracing Backends

#### Grafana Tempo

Replace Jaeger with Tempo in `docker-compose.yml`:

```yaml
services:
  tempo:
    image: grafana/tempo:latest
    ports:
      - "4317:4317"
      - "3200:3200"
    command: ["-config.file=/etc/tempo.yaml"]
```

Update router config:

```yaml
exporter:
  endpoint: "tempo:4317"
```

#### Datadog

Use Datadog OTLP endpoint:

```yaml
exporter:
  type: "otlp"
  endpoint: "https://otlp.datadoghq.com"
  insecure: false
```

## Troubleshooting

### Traces Not Appearing

1. **Check services are running:**
   ```bash
   docker-compose ps
   ```

2. **Check router logs for tracing initialization:**
   ```bash
   docker-compose logs semantic-router | grep -i tracing
   ```

3. **Verify OTLP endpoint connectivity:**
   ```bash
   telnet localhost 4317
   ```

4. **Check sampling rate:**
   - Ensure `always_on` for development
   - Increase `rate` if using probabilistic sampling

### Connection Refused Errors

If the Python example can't connect to the router:

```bash
# Check router is accessible
curl http://localhost:8000/health

# Check Docker network
docker network inspect distributed-tracing_tracing-network
```

### High Memory Usage

If you see high memory usage:

1. **Reduce sampling rate:**
   ```yaml
   sampling:
     type: "probabilistic"
     rate: 0.01  # 1% sampling
   ```

2. **Check batch export settings** in the application code

## Best Practices

1. **Start with stdout exporter** to verify tracing works before using OTLP
2. **Use probabilistic sampling** in production (10% is a good starting point)
3. **Add meaningful attributes** to spans for debugging
4. **Monitor exporter health** and track export failures
5. **Correlate traces with logs** using the same service name
6. **Set appropriate timeout values** for span export

## Production Deployment

For production, consider:

1. **Use TLS** for OTLP endpoint:
   ```yaml
   exporter:
     endpoint: "otlp-collector.prod.svc:4317"
     insecure: false
   ```

2. **Tune sampling rate** based on traffic:
   - High traffic: 0.01-0.1 (1-10%)
   - Medium traffic: 0.1-0.5 (10-50%)
   - Low traffic: 0.5-1.0 (50-100%)

3. **Use a dedicated OTLP collector** (not Jaeger directly)

4. **Set resource limits** in Kubernetes:
   ```yaml
   resources:
     limits:
       memory: "2Gi"
       cpu: "1000m"
   ```

## Additional Resources

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenAI Instrumentation](https://github.com/traceloop/openllmetry)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [vLLM Semantic Router Tracing Guide](../../website/docs/tutorials/observability/distributed-tracing.md)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)

## Support

For questions or issues:
- Create an issue in the repository
- Join the `#semantic-router` channel in vLLM Slack
- Check the [troubleshooting guide](../../website/docs/troubleshooting/)

## License

Apache 2.0 - See LICENSE file in the repository root.
