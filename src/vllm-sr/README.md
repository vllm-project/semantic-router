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
# Start the router (includes dashboard, simulator sidecar, and first-run setup)
HF_TOKEN=hf_xxx vllm-sr serve

# Start an isolated second local stack on offset host ports
VLLM_SR_STACK_NAME=lane-b VLLM_SR_PORT_OFFSET=200 HF_TOKEN=hf_xxx vllm-sr serve

# Open the dashboard
# http://localhost:8700
# second stack example: http://localhost:8900

# Optional: open the dashboard in your browser
vllm-sr dashboard

# View logs
vllm-sr logs router
vllm-sr logs envoy
vllm-sr logs dashboard
vllm-sr logs simulator

# Check status
vllm-sr status

# Stop
vllm-sr stop
```

### Kubernetes Deployment

The same CLI deploys to Kubernetes via Helm:

```bash
# Deploy to Kubernetes (uses your existing config.yaml)
HF_TOKEN=hf_xxx vllm-sr serve --target k8s --profile dev --config config.yaml

# Deploy to a specific namespace and context
HF_TOKEN=hf_xxx vllm-sr serve --target k8s --namespace production --context prod-cluster

# Check status / logs / stop
vllm-sr status --target k8s
vllm-sr logs router --target k8s -f
vllm-sr stop --target k8s
```

**Credential handling:** Sensitive environment variables (`HF_TOKEN`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) are automatically stored in a Kubernetes Secret
(`vllm-sr-env-secrets`) and mounted via `envFrom`. They never appear as
plain-text values in Helm overrides or the Deployment spec. Non-sensitive
variables (`HF_ENDPOINT`, `HF_HOME`, etc.) are passed as standard `env`
entries.

The secret is created before `helm upgrade --install` and cleaned up by
`vllm-sr stop --target k8s`.

If you start in an empty directory, `vllm-sr serve` bootstraps a minimal workspace and opens the dashboard in setup mode. Configure your first model there, then activate routing.

Local dashboard state is persisted under `.vllm-sr/dashboard-data/` and bind-mounted into the container at `/app/data`. User accounts, evaluation history, and ML pipeline artifacts survive `vllm-sr stop` followed by a new `vllm-sr serve` as long as that workspace directory is kept.

The fleet simulator sidecar is started on the same runtime network by default. The dashboard backend proxies it at `/api/fleet-sim/*`, and the dashboard exposes its workflows under the `Fleet Sim` top-bar dropdown.

To run parallel local stacks from the same machine or multiple worktrees, set `VLLM_SR_STACK_NAME` and `VLLM_SR_PORT_OFFSET` before `vllm-sr serve`, `vllm-sr status`, `vllm-sr dashboard`, and `vllm-sr stop`. The stack name isolates container and network names, and the port offset shifts the published host ports while keeping internal container ports unchanged.

### Advanced YAML-first setup

```bash
# Validate a hand-authored canonical config before serving
vllm-sr validate config.yaml
```

`vllm-sr init` was removed in v0.3. Author `config.yaml` directly using the canonical `version/listeners/providers/routing/global` layout, migrate an older file with `vllm-sr config migrate --config old-config.yaml`, or import supported OpenClaw model providers with `vllm-sr config import --from openclaw`. Router-wide defaults come from the router itself and can be overridden under `global:`.

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

Each example shows the plugin list inside a canonical `routing.decisions[]` entry.

1. **semantic-cache** - Cache similar requests:

```yaml
routing:
  decisions:
    - name: "cached-route"
      plugins:
        - type: "semantic-cache"
          configuration:
            enabled: true
            similarity_threshold: 0.92  # 0.0-1.0, higher = more strict
            ttl_seconds: 3600  # Optional: cache TTL in seconds
```

2. **jailbreak** - Block adversarial prompts:

```yaml
routing:
  decisions:
    - name: "guarded-route"
      plugins:
        - type: "jailbreak"
          configuration:
            enabled: true
            threshold: 0.8  # Optional: detection sensitivity 0.0-1.0
```

3. **pii** - Enforce PII policies:

```yaml
routing:
  decisions:
    - name: "pii-route"
      plugins:
        - type: "pii"
          configuration:
            enabled: true
            threshold: 0.7  # Optional: detection sensitivity 0.0-1.0
            pii_types_allowed: ["EMAIL_ADDRESS"]  # Optional: list of allowed PII types
```

4. **system_prompt** - Inject custom instructions:

```yaml
routing:
  decisions:
    - name: "persona-route"
      plugins:
        - type: "system_prompt"
          configuration:
            enabled: true
            system_prompt: "You are a helpful assistant."
            mode: "replace"  # "replace" (default) or "insert" (prepend)
```

5. **header_mutation** - Modify HTTP headers:

```yaml
routing:
  decisions:
    - name: "header-route"
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
routing:
  decisions:
    - name: "fact-check-route"
      plugins:
        - type: "hallucination"
          configuration:
            enabled: true
            use_nli: false  # Optional: use NLI for detailed analysis
            hallucination_action: "header"  # "header", "body", or "none"
```

7. **router_replay** - Record decisions for debugging:

```yaml
routing:
  decisions:
    - name: "debug-route"
      plugins:
        - type: "router_replay"
          configuration:
            enabled: true
            max_records: 200  # Optional: max records in memory (default: 200)
            capture_request_body: false  # Optional: capture request payloads (default: false)
            capture_response_body: false  # Optional: capture response payloads (default: false)
            max_body_bytes: 4096  # Optional: max bytes to capture (default: 4096)
```

Router replay records are exposed through:

- `GET /v1/router_replay?limit=20&offset=0&search=req-123&decision=foo&model=bar&cache_status=cached` - List recent records with pagination metadata. Default page size is `20`; larger `limit` values are capped at `100`.
- `GET /v1/router_replay/aggregate?search=req-123&decision=foo&model=bar&cache_status=cached` - Return summary and chart aggregates for the filtered replay set.
- `GET /v1/router_replay/{id}` - Fetch a single replay record.

If a replay page would exceed the ext-proc gRPC message budget, the router returns `413 Payload Too Large` instead of failing the stream.

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
# Validate configuration (including plugins)
vllm-sr validate

# Migrate older configs to the canonical contract
vllm-sr config migrate --config old-config.yaml

# Import supported OpenClaw model providers into canonical config.yaml
vllm-sr config import --from openclaw --source openclaw.json --target config.yaml
```

### File Descriptor Limits

The CLI automatically sets file descriptor limits to 65,536 for Envoy proxy. To customize:

```bash
export VLLM_SR_NOFILE_LIMIT=100000  # Optional (min: 8192)
vllm-sr serve
```

## License

Apache 2.0
