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

# Evaluate how signals fire for a prompt (requires: vllm-sr serve)
# Single prompt — readable summary (default)
vllm-sr eval --prompt "Explain inflation vs recession in plain English."
# decision: economics
# used signals: 3
#   - domain:economics
#   - keyword:inflation
#   - embedding:price_movement
# matched signals: 3
#   - domains:economics
#   - keywords:inflation
#   - embeddings:price_movement
# unmatched signals: 3
# signal confidences:
#   - domain:economics: 0.95
#   - keyword:inflation: 0.87
#   - embedding:price_movement: 0.82
# routing: economics

# Single prompt — full JSON payload
vllm-sr eval --prompt "Explain inflation vs recession in plain English." --json

# Multi-turn messages array (OpenAI chat format) — readable summary
vllm-sr eval --messages '[{"role":"system","content":"You are a careful tutor."},{"role":"user","content":"Explain inflation vs recession in plain English."}]'

# Multi-turn messages array — full JSON payload
vllm-sr eval --messages '[{"role":"system","content":"You are a careful tutor."},{"role":"user","content":"Explain inflation vs recession in plain English."}]' --json

# Override endpoint (e.g. remote stack or non-default port)
vllm-sr eval --prompt "hello" --endpoint http://localhost:8080

# Common errors:
#   Router not started:
#     ERROR - Router is not running at http://localhost:8080/api/v1/eval. Start the router with 'vllm-sr serve' and retry.
#   Wrong port (hitting a proxy instead of the router API):
#     ERROR - Router returned 403 from http://localhost:8080/api/v1/eval. This looks like a proxy or gateway —
#             check that --endpoint points directly to the router API port (default: 8080), not to Envoy or another proxy.
#   Invalid request body (400):
#     ERROR - Router returned 400 INVALID_INPUT: text cannot be empty
#   Service unavailable (503):
#     ERROR - Router returned 503 SERVICE_UNAVAILABLE: classifier not ready

# Check status
vllm-sr status

# Send a one-shot chat completion through Envoy (default model: MoM)
vllm-sr chat "hello"
vllm-sr chat --json "hello"
# Remote or port-forwarded stack
vllm-sr chat --base-url http://localhost:8080 "hello"

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

# Chat completion against a port-forwarded or ingress URL (requires --base-url)
vllm-sr chat --base-url http://localhost:8080 "hello"
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
- `memory` - Retrieve and store route-local conversation memory
- `system_prompt` - Inject custom system prompts
- `header_mutation` - Add/modify HTTP headers
- `hallucination` - Detect hallucinations in responses
- `router_replay` - Record routing decisions for debugging
- `rag` - Inject retrieved knowledge into prompts
- `image_gen` - Hand a matched route off to an image generation backend
- `fast_response` - Return a route-local response immediately
- `request_params` - Sanitize or cap request body parameters before forwarding
- `response_jailbreak` - Screen model output before returning it
- `tools` - Restrict or curate tool access per route

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

2. **fast_response** - Return a route-local response:

```yaml
routing:
  decisions:
    - name: "guarded-route"
      plugins:
        - type: "fast_response"
          configuration:
            message: "This request was blocked by the matched route policy."
```

3. **system_prompt** - Inject custom instructions:

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

4. **header_mutation** - Modify HTTP headers:

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

5. **hallucination** - Detect hallucinations:

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

6. **router_replay** - Record decisions for debugging:

```yaml
routing:
  decisions:
    - name: "debug-route"
      plugins:
        - type: "router_replay"
          configuration:
            enabled: true
            max_records: 10000  # Optional: max records in memory (default: 10000)
            capture_request_body: true  # Optional: capture request payloads (default: true)
            capture_response_body: true  # Optional: capture response payloads (default: true)
            max_body_bytes: 4096  # Optional: max bytes to capture (default: 4096)
```

7. **memory** - Retrieve route-local memory:

```yaml
routing:
  decisions:
    - name: "memory-route"
      plugins:
        - type: "memory"
          configuration:
            enabled: true
            retrieval_limit: 5
            similarity_threshold: 0.75
            auto_store: true
```

8. **rag** - Inject retrieved context:

```yaml
routing:
  decisions:
    - name: "knowledge-route"
      plugins:
        - type: "rag"
          configuration:
            enabled: true
            backend: "milvus"
            top_k: 5
            similarity_threshold: 0.8
```

9. **tools** - Restrict available tools:

```yaml
routing:
  decisions:
    - name: "tool-route"
      plugins:
        - type: "tools"
          configuration:
            enabled: true
            mode: "filtered"
            allow_tools: ["search_web"]
            block_tools: ["exec_cmd"]
```

10. **image_gen** - Route to an image backend:

```yaml
routing:
  decisions:
    - name: "image-route"
      plugins:
        - type: "image_gen"
          configuration:
            enabled: true
            backend: "vllm_omni"
            backend_config:
              base_url: "http://image-router:8005"
```

11. **request_params** - Cap or strip request parameters:

```yaml
routing:
  decisions:
    - name: "budget-route"
      plugins:
        - type: "request_params"
          configuration:
            blocked_params: ["logprobs", "top_logprobs"]
            max_tokens_limit: 512
            max_n: 1
            strip_unknown: true
```

12. **response_jailbreak** - Screen generated output:

```yaml
routing:
  decisions:
    - name: "safety-route"
      plugins:
        - type: "response_jailbreak"
          configuration:
            enabled: true
            threshold: 0.8
            action: "header"
```

Router replay records are exposed through:

- `GET /v1/router_replay?limit=20&offset=0&search=req-123&decision=foo&model=bar&cache_status=cached` - List recent records with pagination metadata. Default page size is `20`; larger `limit` values are capped at `100`.
- `GET /v1/router_replay/aggregate?search=req-123&decision=foo&model=bar&cache_status=cached` - Return summary and chart aggregates for the filtered replay set.
- `GET /v1/router_replay/{id}` - Fetch a single replay record.

If a replay page would exceed the ext-proc gRPC message budget, the router returns `413 Payload Too Large` instead of failing the stream.

**Validation Rules:**

- **Plugin Type**: Must be one of: `semantic-cache`, `memory`, `system_prompt`, `header_mutation`, `hallucination`, `router_replay`, `rag`, `image_gen`, `fast_response`, `request_params`, `response_jailbreak`, `tools`
- **enabled**: Must be a boolean (required for most plugins)
- **similarity_threshold/min_confidence_threshold**: Must be a float between 0.0 and 1.0
- **max_records/max_body_bytes**: Must be a positive integer
- **ttl_seconds**: Must be a non-negative integer
- **system_prompt**: Must be a string (if provided)
- **mode**: Must be "replace" or "insert" (if provided)
- **injection_mode**: Must be `tool_role` or `system_prompt` (if provided)
- **on_failure**: Must be `skip`, `block`, or `warn` (if provided)
- **action**: Must be `block`, `header`, or `none` (if provided)

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
