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
- **Container Runtime**: Docker or Podman (required for running the router container)

## Quick Start

### 1. Use the one-line installer (macOS/Linux)

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

The installer:

- Detects Python 3.10 or newer
- Installs `vllm-sr` into `~/.local/share/vllm-sr`
- Writes a launcher to `~/.local/bin/vllm-sr`
- Prepares Docker or Podman for `vllm-sr serve` unless you opt out
- Starts `vllm-sr serve` automatically and opens the dashboard when possible
- Prints dashboard access and remote-server hints if a browser cannot be opened

Useful variants:

```bash
# Install only the CLI
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --mode cli

# Pin local serve mode to Podman
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime podman

# Force the first launch onto the AMD/ROCm path
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --platform amd

# Install without auto-starting serve + dashboard
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --no-launch

# Skip runtime bootstrap and keep only userland install steps
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime skip
```

If `~/.local/bin` is not already on your `PATH`, the installer prints the export line to add it.

### Agent install (OpenClaw and similar agents)

Use the agent flow when another local agent should perform the install on your behalf. Keep the same supported installer, but tell the agent to use the no-launch path so it does not auto-start `vllm-sr serve` or open a browser during the handoff.

Preferred long-term path: publish or install the repo-managed `openclaw-vsr-bridge` skill from this repository (`skills/openclaw-vsr-bridge/SKILL.md`). Until that registry entry exists, use the hosted prompt files below so homepage and agent copy stay short.

Hosted prompt files:

- CLI-only prompt: [https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md](https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md)
- Install-and-bridge prompt: [https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md](https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md)

CLI-only prompt:

```text
Fetch and follow https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md.
```

Install-and-bridge prompt:

```text
Fetch and follow https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md.
```

The hosted Markdown files still point back to the same supported installer. OpenClaw handoff still happens later through the bridge skill or `vllm-sr config import --from openclaw`, not through a second installer.

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

# Use custom Docker image
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# Control image pull policy
vllm-sr serve --image-pull-policy always
```

## Kubernetes Deployment

For production deployments on Kubernetes or OpenShift, use the **Kubernetes Operator**:

### Quick Start with Operator

```bash
# Clone repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/deploy/operator

# Install CRDs and operator
make install
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest

# Deploy a semantic router instance
kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

**Benefits:**

- ✅ Declarative configuration using Kubernetes CRDs
- ✅ Automatic platform detection (OpenShift/Kubernetes)
- ✅ Built-in high availability and scaling
- ✅ Integrated monitoring and observability
- ✅ Lifecycle management and upgrades

See the **[Kubernetes Operator Guide](k8s/operator)** for complete documentation.

### Other Kubernetes Deployment Options

- **[Istio Integration](k8s/istio)** - Service mesh deployment
- **[AI Gateway](k8s/ai-gateway)** - Gateway API integration
- **[Production Stack](k8s/production-stack)** - Complete production setup
- **[Dynamo](k8s/dynamo)** - Dynamic configuration management

## Docker Compose

For local development and testing:

- **[Docker Compose](docker-compose)** - Quick local deployment

## Next Steps

- **[Configuration Guide](configuration)** - Advanced routing and signal configuration
- **[Kubernetes Operator](k8s/operator)** - Production Kubernetes deployment
- **[API Documentation](../api/router)** - Complete API reference
- **[Tutorials](../tutorials/signal/overview)** - Learn by example

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **Community**: Join `#semantic-router` channel in vLLM Slack
- **Documentation**: [vllm-semantic-router.com](https://vllm-semantic-router.com/)
