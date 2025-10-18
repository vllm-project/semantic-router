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

- **Go**: V1.24.1 or higher (matches the module requirements)
- **Rust**: V1.90.0 or higher (for Candle bindings)
- **Python**: V3.8 or higher (for model downloads)
- **HuggingFace CLI**: Required for fetching models (`pip install huggingface_hub`)

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

Edit `config/config.yaml` to point to your LLM endpoints:

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
The model name in your configuration **must exactly match** the `--served-model-name` parameter used when starting your vLLM server:

```bash
# When starting vLLM server:
vllm serve microsoft/phi-4 --port 11434 --served-model-name your-model-name

# The config.yaml must reference the model in model_config:
model_config:
  "your-model-name":  # ✅ Must match --served-model-name
    preferred_endpoints: ["your-endpoint"]

vllm_endpoints:
  "your-model-name":             # ✅ Must match --served-model-name
    # ... configuration
```

If these names don't match, the router won't be able to route requests to your model.

The default configuration includes example endpoints that you should update for your setup.
:::

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
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of x^2?"}
    ]
  }'
```

:::tip[VSR Decision Tracking]
The router automatically adds response headers (`x-vsr-selected-category`, `x-vsr-selected-reasoning`, `x-vsr-selected-model`) to help you understand how requests are being processed. Use `curl -i` to see these headers in action. See [VSR Headers Documentation](../troubleshooting/vsr-headers.md) for details.
:::

## Next Steps

After successful installation:

1. **[Configuration Guide](configuration.md)** - Customize your setup and add your own endpoints
2. **[API Documentation](../api/router.md)** - Detailed API reference
3. **[VSR Headers](../troubleshooting/vsr-headers.md)** - Understanding router decision tracking headers

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-org/semantic-router/issues)
- **Documentation**: Full documentation at [Read the Docs](https://vllm-semantic-router.com/)

You now have a working Semantic Router that runs entirely on CPU and intelligently routes requests to specialized models!
