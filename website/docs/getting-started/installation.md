# Install in Local

This guide will help you set up and install the Semantic Router on your system. The router runs entirely on CPU and does not require GPU for inference.

## System Requirements

**Note**: No GPU required - the router runs efficiently on CPU using optimized BERT models.

### Software Dependencies

- **Go**: Version 1.24.1 or higher (matches the module requirements)
- **Rust**: Version 1.70 or higher (for Candle bindings)
- **Python**: Version 3.8 or higher (for model downloads)
- **HuggingFace CLI**: For model downloads (`pip install huggingface_hub`)

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
pip install huggingface_hub
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

> **Tip:** `make test` invokes `make download-models` automatically, so you only need to run this step manually the first time or when refreshing the cache.

### 5. Configure Backend Endpoints

Edit `config/config.yaml` to point to your LLM endpoints:

```yaml
# Example: Configure your vLLM or Ollama endpoints
vllm_endpoints:
  - name: "your-endpoint"
    address: "your-llm-server.com"  # Replace with your server
    port: 11434                     # Replace with your port
    models:
      - "your-model-name"           # Replace with your model
    weight: 1

model_config:
  "your-model-name":
    pii_policy:
      allow_by_default: false  # Deny all PII by default
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON", "GPE", "PHONE_NUMBER"]  # Only allow these specific PII types
    preferred_endpoints: ["your-endpoint"]
```

The default configuration includes example endpoints that you should update for your setup.

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

## Next Steps

After successful installation:

1. **[Configuration Guide](configuration.md)** - Customize your setup and add your own endpoints
2. **[API Documentation](../api/router.md)** - Detailed API reference

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-org/semantic-router/issues)
- **Documentation**: Full documentation at [Read the Docs](https://vllm-semantic-router.com/)

You now have a working Semantic Router that runs entirely on CPU and intelligently routes requests to specialized models!
