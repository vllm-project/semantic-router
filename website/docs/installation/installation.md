---
sidebar_position: 2
---

# Install in Local

This guide will help you set up and install the Semantic Router on your system. The router runs entirely on CPU and does not require GPU for inference.

## System Requirements

:::note
No GPU required - the router runs efficiently on CPU using optimized BERT models.
:::

Semantic Router depends on the following software:

- **Go**: v1.24.1 or higher (matches the module requirements)
- **Rust**: v1.90.0 or higher (for Candle bindings)
- **Python**: v3.8 or higher (for model downloads)
- **HuggingFace CLI**: Required for fetching models

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

### 2. Install Dependencies

#### Install Go (if not already installed)

```bash
# Check if Go is installed
go version

# If not installed, download from https://golang.org/dl/
# Or use package manager:
# macOS: brew install go
# Ubuntu: sudo apt install golang-go
```

#### Install Rust (if not already installed)

```bash
# Check if Rust is installed
rustc --version

# If not installed:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Install Python (if not already installed)

```bash
# Check if Python is installed
python --version

# If not installed:
# macOS: brew install python
# Ubuntu: sudo apt install python3 python3-pip (Tips: need python3.8+)
```

#### Install HuggingFace CLI

```bash
pip install huggingface_hub hf_transfer
```

### 3. Build the Project

```bash
# Build everything (Rust + Go)
make build
```

This command will:

- Build the Rust candle-binding library
- Build the Go router binary
- Place the executable in `bin/router`

### 4. Download Pre-trained Models

```bash
# Download all required models (about 1.5GB total)
make download-models
```

This downloads the CPU-optimized BERT models for:

- Category classification
- PII detection
- Jailbreak detection

:::tip
`make test` invokes `make download-models` automatically, so you only need to run this step manually the first time or when refreshing the cache.
:::

### 5. Configure Backend Endpoints

Edit `config/config.yaml` to point to your vLLM or OpenAI-compatible backend:

```yaml
# Example: Configure your vLLM or Ollama endpoints
vllm_endpoints:
  - name: "your-endpoint"
    address: "127.0.0.1"        # MUST be IP address (IPv4 or IPv6)
    port: 11434                 # Replace with your port
    weight: 1

model_config:
  "your-model-name":            # Replace with your model name
    pii_policy:
      allow_by_default: false  # Deny all PII by default
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON", "GPE", "PHONE_NUMBER"]  # Only allow these specific PII types
    preferred_endpoints: ["your-endpoint"]

default_model: "your-model-name"
```

:::note[**Important: Address Format Requirements**]
The `address` field **must** contain a valid IP address (IPv4 or IPv6). Domain names are not supported.

**✅ Correct formats:**

- `"127.0.0.1"` (IPv4)
- `"192.168.1.100"` (IPv4)

**❌ Incorrect formats:**

- `"localhost"` → Use `"127.0.0.1"` instead
- `"your-server.com"` → Use the server's IP address
- `"http://127.0.0.1"` → Remove protocol prefix
- `"127.0.0.1:8080"` → Use separate `port` field

:::

:::note[**Important: Model Name Consistency**]
The model name in `model_config` must **exactly match** the `--served-model-name` used when starting vLLM. If they don't match, the router won't route requests to your model.

If `--served-model-name` is not set, you can also use the default `id` returned by `/v1/models` (e.g., `Qwen/Qwen3-1.8B`) as the key in `model_config` and for `default_model`.
:::

#### Example: Llama Model

```bash
# Start vLLM with Llama
vllm serve meta-llama/Llama-2-7b-hf --port 8000 --served-model-name llama2-7b
```

```yaml
# config.yaml
vllm_endpoints:
  - name: "llama-endpoint"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "llama2-7b":                    # Must match --served-model-name
    preferred_endpoints: ["llama-endpoint"]

default_model: "llama2-7b"
```

#### Example: Qwen Model

```bash
# Start vLLM with Qwen
vllm serve Qwen/Qwen3-1.8B --port 8000 --served-model-name qwen3
```

```yaml
# config.yaml
vllm_endpoints:
  - name: "qwen-endpoint"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  "qwen3":                        # Must match --served-model-name
    reasoning_family: "qwen3"     # Enable Qwen3 reasoning syntax
    preferred_endpoints: ["qwen-endpoint"]

default_model: "qwen3"
```

For more configuration options, see the [Configuration Guide](configuration.md).

## Running the Router

### 1. Start the Services

Open two terminals and run:

**Terminal 1: Start Envoy Proxy**

```bash
make run-envoy
```

**Terminal 2: Start Semantic Router**

```bash
make run-router
```

### Step 2: Manual Testing

You can also send custom requests:

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is the derivative of x^2?"}
    ]
  }'
```

Using `"model": "MoM"` (Mixture of Models) lets the router automatically select the best model based on the query category.

:::tip[VSR Decision Headers]
Use `curl -i` to see routing decision headers (`x-vsr-selected-category`, `x-vsr-selected-model`). See [VSR Headers](../troubleshooting/vsr-headers.md) for details.
:::

### 3. Monitoring (Optional)

By default, the router exposes Prometheus metrics at `:9190/metrics`. To disable monitoring:

**Option A: CLI flag**

```bash
./bin/router -metrics-port=0
```

**Option B: Configuration**

```yaml
observability:
  metrics:
    enabled: false
```

When disabled, the `/metrics` endpoint won't start, but all other functionality remains unaffected.

## Next Steps

After successful installation:

1. **[Configuration Guide](configuration.md)** - Customize your setup and add your own endpoints
2. **[API Documentation](../api/router.md)** - Detailed API reference
3. **[VSR Headers](../troubleshooting/vsr-headers.md)** - Understanding router decision tracking headers

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **Documentation**: Full documentation at [Read the Docs](https://vllm-semantic-router.com/)

You now have a working Semantic Router that runs entirely on CPU and intelligently routes requests to specialized models!
