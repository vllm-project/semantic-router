# LLM Katan Server

A FastAPI wrapper around [llm-katan](https://pypi.org/project/llm-katan/) that provides the same API design as mock-vllm but uses real LLM functionality.

## Architecture

This server acts as a proxy that:

1. Receives OpenAI-compatible API requests
2. Forwards them to a running `llm-katan` instance
3. Returns the responses with proper model name mapping
4. Falls back to echo behavior if `llm-katan` is unavailable

## Features

- Same API design as mock-vllm (FastAPI-based)
- Proxies requests to real `llm-katan` backend
- OpenAI-compatible API endpoints:
  - GET /health
  - GET /v1/models  
  - POST /v1/chat/completions
- Fallback behavior when backend is unavailable
- Configurable via environment variables

## Environment Variables

- `MODEL`: HuggingFace model name for llm-katan (default: `Qwen/Qwen2-0.5B-Instruct`)
- `SERVED_MODEL_NAME`: Model name to expose in API (default: same as MODEL)
- `LLM_KATAN_URL`: URL of the llm-katan backend (default: `http://localhost:8001`)
- `HUGGINGFACE_HUB_TOKEN`: HuggingFace authentication token

## Setup

### 1. Start llm-katan backend

```bash
# Install llm-katan
pip install llm-katan

# Start llm-katan server on port 8001
llm-katan --model Qwen/Qwen2-0.5B-Instruct --port 8001
```

### 2. Start this FastAPI server

```bash
# Using Docker
docker run -p 8000:8000 llm-katan-server

# Or directly with Python
pip install -r requirements.txt
python app.py
```

## Usage

### Docker Compose (Recommended)

```yaml
services:
  llm-katan-backend:
    image: python:3.11-slim
    command: >
      sh -c "pip install llm-katan && 
             llm-katan --model Qwen/Qwen2-0.5B-Instruct --port 8001 --host 0.0.0.0"
    ports:
      - "8001:8001"
    environment:
      - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

  llm-katan-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL=Qwen/Qwen2-0.5B-Instruct
      - SERVED_MODEL_NAME=Qwen/Qwen2-0.5B-Instruct
      - LLM_KATAN_URL=http://llm-katan-backend:8001
    depends_on:
      - llm-katan-backend
```
