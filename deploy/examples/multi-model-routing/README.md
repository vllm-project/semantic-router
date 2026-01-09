# Multi-Model Routing Demo

Intelligent query routing between **NVIDIA Nemotron Nano 9B** (Math/Physics/Chemistry/CS) and **Qwen 2.5-3B** (History/Literature/Business/General) using [vLLM Semantic Router](https://vllm-semantic-router.com).

**Note:** This demo routes technical and analytical queries (math, physics, chemistry, computer science) to Nemotron and open-ended conversational queries to Qwen. This routing logic is fully configurable in `config.yaml`.

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- NVIDIA GPU with CUDA support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start (downloads models on first run)
bash start_demo.sh

# Test
python run_demo.py
```

## Stop

```bash
vllm-sr stop
docker compose -f docker-compose-models.yml down
```

## Configuration

**Note:** Configured for DGX Spark. Adjust GPU settings in `docker-compose-models.yml` for other hardware if needed.

Edit `config.yaml` to:

- Add/remove models
- Change routing rules
- Adjust cache thresholds

Edit `docker-compose-models.yml` to:

- Change GPU memory allocation
- Add new vLLM servers
