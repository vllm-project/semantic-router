# vLLM Semantic Router

Intelligent Router for Mixture-of-Models (MoM).

GitHub: https://github.com/vllm-project/semantic-router

## Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install vllm-sr

# Or install from source (development)
cd src/vllm-sr
pip install -e .
```

### Usage

```bash
# Initialize vLLM Semantic Router Configuration
vllm-sr init

# Start with config
vllm-sr serve config.yaml

# View logs
vllm-sr logs

# Check status
vllm-sr status

# Stop
vllm-sr stop
```

## License

Apache 2.0
