# Adapter Configuration Guide

vLLM Semantic Router supports multiple protocol adapters, allowing you to use the same routing engine with different front-end protocols. This guide covers how to configure and use the **Envoy ExtProc** and **HTTP REST** adapters.

## Architecture Overview

The adapter system consists of:

- **RouterEngine**: Protocol-agnostic core routing logic (classification, caching, tool selection, replay recording)
- **Adapter Manager**: Orchestrates multiple protocol adapters from config
- **Protocol Adapters**: Handle protocol-specific request/response formats

```
┌─────────────────────────────────────────────────────┐
│                  Adapter Manager                    │
│  ┌──────────────┐           ┌──────────────┐        │
│  │ ExtProc      │           │ HTTP REST    │        │
│  │ Adapter      │           │ Adapter      │        │
│  │ (Envoy)      │           │ (Direct)     │        │
│  └──────┬───────┘           └──────┬───────┘        │
│         │                          │                │
│         └────────┬──────────────┬──┘                │
│                  │              │                   │
│         ┌────────▼──────────────▼────────┐          │
│         │     Router Engine              │          │
│         │ - Classification               │          │
│         │ - Caching                      │          │
│         │ - Tool Selection               │          │
│         │ - Router Replay                │          │
│         │ - PII/Jailbreak Detection      │          │
│         └────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

## Configuration

### Adapter Configuration

Configure adapters in `config.yaml`:

```yaml
# Adapter Configuration - Choose which protocols to enable
# Multiple adapters can run simultaneously on different ports
adapters:
  - type: "envoy" # ExtProc gRPC adapter for Envoy proxy
    enabled: true
    port: 50051 # Internal port for ExtProc (Envoy connects here)

  - type: "http" # Direct HTTP REST API (OpenAI-compatible)
    enabled: true
    port: 9000 # External port for direct HTTP access
```

### Envoy ExtProc Adapter

The ExtProc adapter integrates with Envoy proxy using the External Processing protocol.

**Configuration:**

```yaml
adapters:
  - type: "envoy"
    enabled: true
    port: 50051
```

**Envoy Configuration:**

```yaml
# envoy.yaml
ext_proc:
  grpc_service:
    envoy_grpc:
      cluster_name: ext_proc_cluster
    timeout: 300s

  processing_mode:
    request_header_mode: SKIP
    response_header_mode: SKIP
    request_body_mode: BUFFERED
    response_body_mode: BUFFERED
    request_trailer_mode: SKIP
    response_trailer_mode: SKIP

clusters:
  - name: ext_proc_cluster
    type: STRICT_DNS
    connect_timeout: 1s
    load_assignment:
      cluster_name: ext_proc_cluster
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: semantic-router
                    port_value: 50051
```

**Access:**

```bash
# Requests go through Envoy (port 8801)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### HTTP REST Adapter

The HTTP adapter provides direct OpenAI-compatible REST API access without requiring Envoy.

**Configuration:**

```yaml
adapters:
  - type: "http"
    enabled: true
    port: 9000
```

**Access:**

```bash
# Direct HTTP access (no Envoy needed)
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

## Shared RouterEngine

Both adapters use the **same RouterEngine instance**, meaning:

✅ **Shared State:**

- Classification decisions
- Cache entries (semantic cache)
- Router replay records
- Tool selection
- Model selection logic

✅ **Consistent Behavior:**

- Same routing decisions for identical queries
- Replay records visible from both adapters
- Cache hits work across adapters

✅ **Independent Protocol Handling:**

- ExtProc uses Envoy ExtProc gRPC protocol
- HTTP uses OpenAI-compatible REST
- Each handles its own request/response format

## Example: Running Both Adapters

**1. Configure both adapters:**

```yaml
adapters:
  - type: "envoy"
    enabled: true
    port: 50051
  - type: "http"
    enabled: true
    port: 9000
```

**4. Send requests via both adapters:**

```bash
# Via Envoy (ExtProc)
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 5+5?"}]}'

# Via HTTP adapter
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Calculate 10*10"}]}'
```

**5. View all replay records (from either adapter):**

```bash
curl http://localhost:9000/v1/router_replay | jq
# Shows records from BOTH adapters!
```

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Envoy ExtProc Documentation](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
- [vLLM Semantic Router Documentation](https://vllm-semantic-router.com)
