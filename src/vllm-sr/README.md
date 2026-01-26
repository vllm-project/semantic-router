# vLLM Semantic Router

Intelligent Router for Mixture-of-Models (MoM).

GitHub: https://github.com/vllm-project/semantic-router

## Quick Start

### Installation

```bash
# Install from PyPI
pip install vllm-sr

# Or install from source (development)
cd src/vllm-sr
pip install -e .
```

### Usage

```bash
# Initialize vLLM Semantic Router Configuration
vllm-sr init

# Start the router (includes dashboard)
vllm-sr serve

# Open dashboard in browser
vllm-sr dashboard

# View logs
vllm-sr logs router
vllm-sr logs envoy
vllm-sr logs dashboard

# Check status
vllm-sr status

# Stop
vllm-sr stop
```

## Features

- **Router**: Intelligent request routing based on intent classification
- **Envoy Proxy**: High-performance proxy with ext_proc integration
- **Dashboard**: Web UI for monitoring and testing (http://localhost:8700)
- **Metrics**: Prometheus metrics endpoint (http://localhost:9190/metrics)

## Endpoints

After running `vllm-sr serve`, the following endpoints are available:

| Endpoint | Port | Description |
|----------|------|-------------|
| Dashboard | 8700 | Web UI for monitoring and Playground |
| API | 8888* | Chat completions API (configurable in config.yaml) |
| Metrics | 9190 | Prometheus metrics |
| gRPC | 50051 | Router gRPC (internal) |
| Jaeger UI | 16686 | Distributed tracing UI |
| Grafana (embedded) | 8700 | Dashboards at /embedded/grafana |
| Prometheus UI | 9090 | Metrics storage and querying |

*Default port, configurable via `listeners` in config.yaml

### Observability

`vllm-sr serve` automatically starts the observability stack:

- **Jaeger**: Distributed tracing embedded at http://localhost:8700/embedded/jaeger (also available directly at http://localhost:16686)
- **Grafana**: Pre-configured dashboards embedded at http://localhost:8700/embedded/grafana
- **Prometheus**: Metrics collection at http://localhost:9090

**Note**: Grafana is optimized for embedded access through the dashboard. For the best experience, use http://localhost:8700/embedded/grafana where anonymous authentication is pre-configured.

Tracing is enabled by default. Traces are visible in Jaeger under the `vllm-sr` service name.

## Configuration

### Plugin Configuration

The CLI supports configuring plugins in your routing decisions. Plugins are per-decision behaviors that customize request handling (security, caching, customization, debugging).

**Supported Plugin Types:**

- `semantic-cache` - Cache similar requests for performance
- `jailbreak` - Detect and block adversarial prompts
- `pii` - Detect and enforce PII policies
- `system_prompt` - Inject custom system prompts
- `header_mutation` - Add/modify HTTP headers
- `hallucination` - Detect hallucinations in responses
- `router_replay` - Record routing decisions for debugging

**Plugin Examples:**

1. **semantic-cache** - Cache similar requests:

```yaml
plugins:
  - type: "semantic-cache"
    configuration:
      enabled: true
      similarity_threshold: 0.92  # 0.0-1.0, higher = more strict
      ttl_seconds: 3600  # Optional: cache TTL in seconds
```

2. **jailbreak** - Block adversarial prompts:

```yaml
plugins:
  - type: "jailbreak"
    configuration:
      enabled: true
      threshold: 0.8  # Optional: detection sensitivity 0.0-1.0
```

3. **pii** - Enforce PII policies:

```yaml
plugins:
  - type: "pii"
    configuration:
      enabled: true
      threshold: 0.7  # Optional: detection sensitivity 0.0-1.0
      pii_types_allowed: ["EMAIL_ADDRESS"]  # Optional: list of allowed PII types
```

4. **system_prompt** - Inject custom instructions:

```yaml
plugins:
  - type: "system_prompt"
    configuration:
      enabled: true
      system_prompt: "You are a helpful assistant."
      mode: "replace"  # "replace" (default) or "insert" (prepend)
```

5. **header_mutation** - Modify HTTP headers:

```yaml
plugins:
  - type: "header_mutation"
    configuration:
      add:
        - name: "X-Custom-Header"
          value: "custom-value"
      update:
        - name: "User-Agent"
          value: "SemanticRouter/1.0"
      delete:
        - "X-Old-Header"
```

6. **hallucination** - Detect hallucinations:

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: false  # Optional: use NLI for detailed analysis
      hallucination_action: "header"  # "header", "body", or "none"
```

7. **router_replay** - Record decisions for debugging:

```yaml
plugins:
  - type: "router_replay"
    configuration:
      enabled: true
      max_records: 200  # Optional: max records in memory (default: 200)
      capture_request_body: false  # Optional: capture request payloads (default: false)
      capture_response_body: false  # Optional: capture response payloads (default: false)
      max_body_bytes: 4096  # Optional: max bytes to capture (default: 4096)
```

**Validation Rules:**

- **Plugin Type**: Must be one of: `semantic-cache`, `jailbreak`, `pii`, `system_prompt`, `header_mutation`, `hallucination`, `router_replay`
- **enabled**: Must be a boolean (required for most plugins)
- **threshold/similarity_threshold**: Must be a float between 0.0 and 1.0
- **max_records/max_body_bytes**: Must be a positive integer
- **ttl_seconds**: Must be a non-negative integer
- **pii_types_allowed**: Must be a list of strings (if provided)
- **system_prompt**: Must be a string (if provided)
- **mode**: Must be "replace" or "insert" (if provided)

**CLI Commands:**

```bash
# Initialize config with plugin examples
vllm-sr init

# Validate configuration (including plugins)
vllm-sr validate config.yaml

# Generate router config with plugins
vllm-sr config router --config config.yaml
```

### File Descriptor Limits

The CLI automatically sets file descriptor limits to 65,536 for Envoy proxy. To customize:

```bash
export VLLM_SR_NOFILE_LIMIT=100000  # Optional (min: 8192)
vllm-sr serve
```

## License

Apache 2.0
