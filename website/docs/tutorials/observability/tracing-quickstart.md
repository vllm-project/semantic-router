# Tracing Quickstart

This guide helps you spin up a local tracing stack and see your first traces in a minute.

## Prerequisites

- Docker and Docker Compose

## Start the local tracing stack

The repo includes a compose file that starts Jaeger and a tracing-enabled Semantic Router instance.

- The router uses `config/config.tracing.yaml` which has tracing enabled and the exporter pointed at Jaeger.

Run:

```bash
# from repo root
docker compose -f tools/tracing/docker-compose.tracing.yaml up -d
```

## Send a test request

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## View traces

1. Open Jaeger UI: http://localhost:16686
2. Choose service: `vllm-semantic-router`
3. Find traces → click one to inspect spans

You should see spans like:

- `semantic_router.request.received`
- `semantic_router.classification`
- `semantic_router.cache.lookup`
- `semantic_router.routing.decision`
- `semantic_router.backend.selection`

## Customize

- Change service name or sampling in `config/config.tracing.yaml` under `observability.tracing`.
- To export to another backend (e.g., Tempo), set `exporter.endpoint` and `insecure` accordingly.

## Troubleshooting

- No traces? Confirm tracing is enabled in the YAML and Jaeger is reachable at `jaeger:4317` inside the compose network.
- Empty service list in Jaeger? Make one request to generate spans, then refresh.
