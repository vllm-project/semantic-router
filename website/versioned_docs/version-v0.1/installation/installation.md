---
sidebar_position: 2
---

# Installation

This guide will help you install and run the vLLM Semantic Router. The router runs entirely on CPU and does not require GPU for inference.

## System Requirements

:::note
No GPU required - the router runs efficiently on CPU using optimized BERT models.
:::

**Requirements:**

- **Python**: 3.10 or higher
- **Docker**: Required for running the router container

## Quick Start

### 1. Install vLLM Semantic Router

```bash
# Create a virtual environment (recommended)
python -m venv vsr
source vsr/bin/activate  # On Windows: vsr\Scripts\activate

# Install from PyPI
pip install vllm-sr
```

Verify installation:

```bash
vllm-sr --version
```

### 2. Start `vllm-sr`

```bash
vllm-sr serve
```

If `config.yaml` does not exist yet, `vllm-sr serve` bootstraps a minimal workspace and starts the dashboard in setup mode.

The router will:

- Automatically download required ML models (~1.5GB, one-time)
- Start the dashboard on port 8700
- Start Envoy proxy on port 8888 after activation
- Start the semantic router service after activation
- Enable metrics on port 9190

### 3. Open the Dashboard

Open [http://localhost:8700](http://localhost:8700) in your browser.

For first-run setup:

1. Configure one or more models.
2. Choose a routing preset or keep the single-model baseline.
3. Activate the generated config.

After activation, `config.yaml` is written to the current directory and the router exits setup mode.

### 4. Optional: open the dashboard from the CLI

```bash
vllm-sr dashboard
```

### 5. Test the Router

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Common Commands

```bash
# View logs
vllm-sr logs router        # Router logs
vllm-sr logs envoy         # Envoy logs
vllm-sr logs router -f     # Follow logs

# Check status
vllm-sr status

# Stop the router
vllm-sr stop
```

## Advanced Configuration

### YAML-first workflow

If you prefer to edit YAML directly instead of using the dashboard setup flow:

```bash
# Generate a lean advanced sample in the current directory
vllm-sr init

# Validate it before serving
vllm-sr validate config.yaml
```

`vllm-sr init` is optional. It generates an advanced sample and `.vllm-sr/router-defaults.yaml` for YAML-first users. `router-defaults.yaml` contains advanced runtime defaults and is not required for first-run dashboard setup.

### HuggingFace Settings

Set environment variables before starting:

```bash
export HF_ENDPOINT=https://huggingface.co  # Or mirror: https://hf-mirror.com
export HF_TOKEN=your_token_here            # Only for gated models
export HF_HOME=/path/to/cache              # Custom cache directory

vllm-sr serve
```

### Custom Options

```bash
# Use custom config file
vllm-sr serve --config my-config.yaml

# Use custom Docker image
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# Control image pull policy
vllm-sr serve --image-pull-policy always
```

## Next Steps

- **[Configuration Guide](configuration.md)** - Advanced routing and signal configuration
- **[API Documentation](../api/router.md)** - Complete API reference
- **[Tutorials](../tutorials/intelligent-route/keyword-routing.md)** - Learn by example

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **Community**: Join `#semantic-router` channel in vLLM Slack
- **Documentation**: [vllm-semantic-router.com](https://vllm-semantic-router.com/)
