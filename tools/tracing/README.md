# Tracing (Local Development Stack)

This directory provides a local Jaeger + tracing-enabled semantic-router stack for development, debugging, and demonstration.

## Why here?

`tools/tracing` groups this with other local-only utilities (see `tools/observability` for metrics stack). Production deployments should rely on manifests in `deploy/kubernetes` / `openshift` instead of this all-in-one compose.

## Quick Start

```bash
docker compose -f tools/tracing/docker-compose.tracing.yaml up -d
```

## Access

- Jaeger UI: http://localhost:16686
- Tracing Router API: http://localhost:8081
- Metrics (tracing instance): http://localhost:9191/metrics

## Send a Test Request

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## View Traces

1. Open Jaeger UI
2. Select service: `vllm-semantic-router` (set via resource.service_name in config/config.tracing.yaml)
3. Click "Find Traces"

## Stopping

```bash
docker compose -f tools/tracing/docker-compose.tracing.yaml down
```

With volumes removal:

```bash
docker compose -f tools/tracing/docker-compose.tracing.yaml down -v
```

## Environment Variables

| Variable                    | Purpose                         |
| --------------------------- | ------------------------------- |
| OTEL_EXPORTER_OTLP_ENDPOINT | Where spans are exported (gRPC) |
| OTEL_SERVICE_NAME           | Logical service name in traces  |

Note: the router reads tracing settings from the YAML config (observability.tracing._). The OTEL\__ env vars here are only informational and do not override the YAML. To change exporter endpoint or service name, edit `config/config.tracing.yaml`.

## Relationship with Metrics Stack

If you also want Prometheus/Grafana metrics:

```bash
make o11y-local  # or o11y-compose for full docker mode
```

## Next Steps

- Distributed tracing docs: `website/docs/tutorials/observability/distributed-tracing.md`
- Tracing quickstart: `website/docs/tutorials/observability/tracing-quickstart.md`
- Router config examples: `config/`
