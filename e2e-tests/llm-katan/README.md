# LLM Katan - Lightweight LLM Server for Testing

A lightweight LLM serving package using FastAPI and HuggingFace transformers, designed for testing and development with real tiny models.

## Features

- 🚀 **FastAPI-based**: High-performance async web server
- 🤗 **HuggingFace Integration**: Real model inference with transformers
- ⚡ **Tiny Models**: Ultra-lightweight models for fast testing (Qwen3-0.6B, etc.)
- 🔄 **Multi-Instance**: Run same model on different ports with different names
- 🎯 **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- 📦 **PyPI Ready**: Easy installation and distribution
- 🛠️ **vLLM Support**: Optional vLLM backend for production-like performance

## Quick Start

### Installation

```bash
pip install llm-katan
```

### Setup

#### HuggingFace Token (Required)

LLM Katan uses HuggingFace transformers to download models. You'll need a HuggingFace token for:

- Private models
- Avoiding rate limits
- Reliable model downloads

**Option 1: Environment Variable**

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

**Option 2: Login via CLI**

```bash
huggingface-cli login
```

**Option 3: Token file in home directory**

```bash
# Create ~/.cache/huggingface/token file with your token
echo "your_token_here" > ~/.cache/huggingface/token
```

**Get your token:** Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Basic Usage

```bash
# Start server with a tiny model
llm-katan --model Qwen/Qwen3-0.6B --port 8000

# Start with custom served model name
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# With vLLM backend (optional)
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --backend vllm
```

### Multi-Instance Testing

```bash
# Terminal 1: Qwen endpoint
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name "Qwen/Qwen2-0.5B-Instruct"

# Terminal 2: Same model, different name
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## API Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)

### Example API Usage

```bash
# Basic chat completion
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Creative writing example
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Write a short poem about coding"}
    ],
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Check available models
curl http://127.0.0.1:8000/v1/models

# Health check
curl http://127.0.0.1:8000/health
```

## Use Cases

- **Testing**: Lightweight alternative to full LLM deployments
- **Development**: Fast iteration with real model behavior
- **CI/CD**: Automated testing with actual inference
- **Prototyping**: Quick setup for AI application development

## Configuration

### Command Line Options

```bash
# All available options
llm-katan [OPTIONS]

Required:
  -m, --model TEXT              Model name to load (e.g., 'Qwen/Qwen3-0.6B') [required]

Optional:
  -n, --name, --served-model-name TEXT    Model name to serve via API (defaults to model name)
  -p, --port INTEGER            Port to serve on (default: 8000)
  -h, --host TEXT               Host to bind to (default: 0.0.0.0)
  -b, --backend [transformers|vllm]      Backend to use (default: transformers)
  --max, --max-tokens INTEGER   Maximum tokens to generate (default: 512)
  -t, --temperature FLOAT       Sampling temperature (default: 0.7)
  -d, --device [auto|cpu|cuda]  Device to use (default: auto)
  --log-level [debug|info|warning|error]  Log level (default: INFO)
  --version                     Show version and exit
  --help                        Show help and exit
```

#### Advanced Usage Examples

```bash
# Custom generation settings
llm-katan --model Qwen/Qwen3-0.6B --max-tokens 1024 --temperature 0.9

# Force specific device
llm-katan --model Qwen/Qwen3-0.6B --device cpu --log-level debug

# Custom host and port
llm-katan --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 9000

# Multiple servers with different settings
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --max-tokens 512 --temperature 0.1
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --max-tokens 256 --temperature 0.9
```

### Environment Variables

- `LLM_KATAN_MODEL`: Default model to load
- `LLM_KATAN_PORT`: Default port (8000)
- `LLM_KATAN_BACKEND`: Backend type (transformers|vllm)

## Development

```bash
# Clone and install in development mode
git clone <repo>
cd e2e-tests/llm-katan
pip install -e .

# Run with development dependencies
pip install -e ".[dev]"
```

## License

MIT License

## Contributing

Contributions welcome! Please see the main repository for guidelines.

---

*Part of the [semantic-router project ecosystem](https://vllm-semantic-router.com/)*
