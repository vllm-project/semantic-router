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

*Default port, configurable via `listeners` in config.yaml

## Configuration

### File Descriptor Limits

The CLI automatically sets file descriptor limits to 65,536 for Envoy proxy. To customize:

```bash
export VLLM_SR_NOFILE_LIMIT=100000  # Optional (min: 8192)
vllm-sr serve
```

## License

Apache 2.0
