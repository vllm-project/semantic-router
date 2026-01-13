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

### 2. Initialize Configuration

```bash
# Create config.yaml in current directory
vllm-sr init
```

This creates a `config.yaml` file with default settings.

### 3. Configure Your Backend

Edit the generated `config.yaml` to configure your model and backend endpoint:

```yaml
providers:
  # Model configuration
  models:
    - name: "qwen/qwen3-1.8b"           # Model name
      endpoints:
        - name: "my_vllm"
          weight: 1
          endpoint: "localhost:8000"    # Domain or IP:port
          protocol: "http"              # http or https
      access_key: "your-token-here"     # Optional: for authentication

  # Default model for fallback
  default_model: "qwen/qwen3-1.8b"
```

**Configuration Options:**

- **endpoint**: Domain name or IP address with port (e.g., `localhost:8000`, `api.openai.com`)
- **protocol**: `http` or `https`
- **access_key**: Optional authentication token (Bearer token)
- **weight**: Load balancing weight (default: 1)

**Example: Local vLLM**

```yaml
providers:
  models:
    - name: "qwen/qwen3-1.8b"
      endpoints:
        - name: "local_vllm"
          weight: 1
          endpoint: "localhost:8000"
          protocol: "http"
  default_model: "qwen/qwen3-1.8b"
```

**Example: External API with HTTPS**

```yaml
providers:
  models:
    - name: "openai/gpt-4"
      endpoints:
        - name: "openai_api"
          weight: 1
          endpoint: "api.openai.com"
          protocol: "https"
      access_key: "sk-xxxxxx"
  default_model: "openai/gpt-4"
```

### 4. Start the Router

```bash
vllm-sr serve
```

The router will:

- Automatically download required ML models (~1.5GB, one-time)
- Start Envoy proxy on port 8888
- Start the semantic router service
- Enable metrics on port 9190

### 5. Launch Dashboard

```bash
vllm-sr dashboard
```

### 6. Test the Router

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
