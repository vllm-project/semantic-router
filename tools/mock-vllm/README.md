# Mock vLLM (OpenAI-compatible) service

A tiny FastAPI server that emulates minimal endpoints used by the router:

- GET /health
- GET /v1/models
- POST /v1/chat/completions

Intended for local testing with Docker Compose profile `testing`.
