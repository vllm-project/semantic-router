# LLM Katan - Lightweight LLM Server for Testing

A lightweight LLM serving package using FastAPI and HuggingFace transformers,
designed for testing and development with real tiny models.

> **🎬 [See Live Demo](https://vllm-project.github.io/semantic-router/e2e-tests/llm-katan/terminal-demo.html)**
> Interactive terminal showing multi-instance setup in action!

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

LLM Katan uses HuggingFace transformers to download models.
You'll need a HuggingFace token for:

- Private models
- Avoiding rate limits
- Reliable model downloads

#### Option 1: Environment Variable

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

#### Option 2: Login via CLI

```bash
huggingface-cli login
```

#### Option 3: Token file in home directory

```bash
# Create ~/.cache/huggingface/token file with your token
echo "your_token_here" > ~/.cache/huggingface/token
```

**Get your token:**
Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

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

**🎬 [Live Demo](https://vllm-project.github.io/semantic-router/e2e-tests/llm-katan/terminal-demo.html)**
See this in action with animated terminals!

> *Note: If GitHub Pages isn't enabled, you can also
> [download and open the demo locally](./terminal-demo.html)*

<!-- markdownlint-disable MD033 -->
<details>
<summary>📺 Preview (click to expand)</summary>
<!-- markdownlint-enable MD033 -->

```bash
# Terminal 1: Installing and starting GPT-3.5-Turbo mock
$ pip install llm-katan
Successfully installed llm-katan-0.1.8

$ llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name "gpt-3.5-turbo"
🚀 Starting LLM Katan server with model: Qwen/Qwen3-0.6B
📛 Served model name: gpt-3.5-turbo
✅ Server running on http://0.0.0.0:8000

# Terminal 2: Starting Claude-3-Haiku mock
$ llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "claude-3-haiku"
🚀 Starting LLM Katan server with model: Qwen/Qwen3-0.6B
📛 Served model name: claude-3-haiku
✅ Server running on http://0.0.0.0:8001

# Terminal 3: Testing both endpoints
$ curl localhost:8000/v1/models | jq '.data[0].id'
"gpt-3.5-turbo"

$ curl localhost:8001/v1/models | jq '.data[0].id'
"claude-3-haiku"

# Same tiny model, different API names! 🎯
```

</details>

```bash
# Terminal 1: Mock GPT-3.5-Turbo
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name "gpt-3.5-turbo"

# Terminal 2: Mock Claude-3-Haiku
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "claude-3-haiku"

# Terminal 3: Test both endpoints
curl http://localhost:8000/v1/models  # Returns "gpt-3.5-turbo"
curl http://localhost:8001/v1/models  # Returns "claude-3-haiku"
```

**Perfect for testing multi-provider scenarios with one tiny model!**

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

### Strengths

- **Fastest time-to-test**: 30 seconds from install to running
- **Minimal resource footprint**: Designed for tiny models and efficient testing
- **No GPU required**: Runs on laptops, Macs, and any CPU-only environment
- **CI/CD integration friendly**: Lightweight and automation-ready
- **Multiple instances**: Run same model with different names on different ports

### Ideal For

- **Automated testing pipelines**: Quick LLM endpoint setup for test suites
- **Development environment mocking**: Real inference without production overhead
- **Quick prototyping**: Fast iteration with actual model behavior
- **Educational/learning scenarios**: Easy setup for AI development learning

### Not Ideal For

- **Production workloads**: Use Ollama or vLLM for production deployments
- **Large model serving**: Designed for tiny models (< 1B parameters)
- **Complex multi-agent workflows**: Use Semantic Kernel or similar frameworks
- **High-performance inference**: Use vLLM or specialized serving solutions

## Configuration

### Command Line Options

```bash
# All available options
llm-katan [OPTIONS]

Required:
  -m, --model TEXT              Model name to load (e.g., 'Qwen/Qwen3-0.6B') [required]

Optional:
  -n, --name, --served-model-name TEXT
                                Model name to serve via API (defaults to model name)
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
llm-katan --model Qwen/Qwen3-0.6B --port 8001 \
  --name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --max-tokens 256 --temperature 0.9
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
