---
sidebar_position: 2
---

# Quickstart

This guide will help you install and run the vLLM Semantic Router. The router runs entirely on CPU and does not require GPU for inference.

## System Requirements

:::note
No GPU required - the router runs efficiently on CPU using optimized BERT models.
:::

**Requirements:**

- **Python**: 3.10 or higher
- **Container Runtime**: Docker (required for running the router container)

## Quick Start

### 1. Use the one-line installer (macOS/Linux)

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

The installer:

- Detects Python 3.10 or newer
- Installs `vllm-sr` into `~/.local/share/vllm-sr`
- Writes a launcher to `~/.local/bin/vllm-sr`
- Prepares Docker for `vllm-sr serve` unless you opt out
- Starts `vllm-sr serve` automatically and opens the dashboard when possible
- Prints dashboard access and remote-server hints if a browser cannot be opened

If `~/.local/bin` is not already on your `PATH`, the installer prints the export line to add it.

Windows users should use the manual PyPI flow below.

### 2. Manual PyPI install

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

### 3. Restart `vllm-sr` later

```bash
vllm-sr serve
```

If you skipped `--no-launch`, the installer already ran one `vllm-sr serve` for you.

If `config.yaml` does not exist yet in the current directory, `vllm-sr serve` bootstraps a minimal setup config and starts the dashboard in setup mode.

The router will:

- Automatically download required ML models (~1.5GB, one-time)
- Start the dashboard on port 8700
- Start the `vllm-sr-sim` sidecar on port 8810
- Start Envoy proxy on port 8888 after activation
- Start the semantic router service after activation
- Enable metrics on port 9190

### 4. Open the Dashboard

Open [http://localhost:8700](http://localhost:8700) in your browser.

If you ran the installer on a remote server and the browser did not open automatically, use the URL and SSH tunnel hint printed by the installer.

For first-run setup:

1. Configure one or more models.
2. Choose a routing preset or keep the single-model baseline.
3. Activate the generated config.

After activation, `config.yaml` is written to the current directory and the router exits setup mode.

### 5. Test the Router

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 6. Optional: open the dashboard from the CLI

```bash
vllm-sr dashboard
```

## Common Commands

```bash
# View logs
vllm-sr logs router        # Router logs
vllm-sr logs envoy         # Envoy logs
vllm-sr logs simulator     # Fleet simulator sidecar logs
vllm-sr logs router -f     # Follow logs

# Check status
vllm-sr status             # Includes simulator sidecar state

# Stop the router
vllm-sr stop
```

## Advanced Configuration

### YAML-first workflow

If you prefer to edit YAML directly instead of using the dashboard setup flow:

```bash
# Validate your canonical config before serving
vllm-sr validate config.yaml
```

`vllm-sr init` was removed in v0.3. Create `config.yaml` directly with the canonical `version/listeners/providers/routing/global` layout, migrate an older file with `vllm-sr config migrate --config old-config.yaml`, or import supported OpenClaw model providers with `vllm-sr config import --from openclaw`.

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

# Set the router log level
vllm-sr serve --log-level debug

# Use custom Docker image
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# Control image pull policy
vllm-sr serve --image-pull-policy always
```

## Docker Compose

For local development and testing:

- **[Docker Compose](docker-compose)** - Quick local deployment

## Next Steps

- **[Install with Operator](k8s/operator)** - Deploy on Kubernetes or OpenShift with the operator
- **[Configuration Guide](configuration)** - Advanced routing and signal configuration
- **[API Documentation](../api/router)** - Complete API reference
- **[Tutorials](../tutorials/signal/overview)** - Learn by example

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **Community**: Join `#semantic-router` channel in vLLM Slack
- **Documentation**: [vllm-semantic-router.com](https://vllm-semantic-router.com/)
