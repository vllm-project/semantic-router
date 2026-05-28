# anthropic-shim

A small FastAPI service that fronts [`llama.cpp`'s `llama-server`][llama-server]
and patches three gaps in its Anthropic Messages API support so the
vLLM Semantic Router e2e suite can exercise realistic upstream
Anthropic behaviour without forking llama.cpp.

## What it fixes

| Gap | llama-server behaviour | Shim behaviour |
| --- | --- | --- |
| `system` as `TextBlockParam[]` | Concatenates text fields with no separator (`"You are helpful.Be concise."`) | Joins text fields with `\n` before forwarding |
| `tool_result.content` as `TextBlockParam[]` | Same flattening issue inside tool results | Joins text fields with `\n` before forwarding |
| Prompt-cache token counters | `cache_creation_input_tokens` and `cache_read_input_tokens` are never populated | Tracks per-session request-prefix hashes; sets `cache_creation_input_tokens` on first request and `cache_read_input_tokens` on subsequent repeats |

Everything else (image blocks, `top_k`, `metadata.user_id`,
`tool_result.is_error`, headers like `anthropic-version` and
`anthropic-beta`, streaming SSE) is forwarded verbatim.

## Layout

```
e2e/testing/anthropic-shim/
├── anthropic_shim/
│   ├── __init__.py
│   ├── __main__.py     # python -m anthropic_shim entry point
│   ├── app.py          # FastAPI proxy
│   └── translate.py    # pure translation helpers
├── tests/
│   ├── test_app.py
│   └── test_translate.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running locally

```bash
# Start llama-server with a tiny GGUF model on port 8080
docker run --rm -p 8080:8080 \
  -v /path/to/models:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --jinja --host 0.0.0.0 --port 8080 -c 4096

# In another terminal, start the shim on port 9080
pip install -e .
ANTHROPIC_SHIM_UPSTREAM_URL=http://127.0.0.1:8080 \
  python -m anthropic_shim

# Hit the shim with an Anthropic Messages request
curl -s http://127.0.0.1:9080/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen-test",
    "system": [
      {"type": "text", "text": "You are a helpful assistant."},
      {"type": "text", "text": "Be very concise.", "cache_control": {"type": "ephemeral"}}
    ],
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 16
  }'
```

## Configuration

All configuration is environment-driven so the same image can be
reused across overlays:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_SHIM_UPSTREAM_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `ANTHROPIC_SHIM_HOST` | `0.0.0.0` | bind address |
| `ANTHROPIC_SHIM_PORT` | `9080` | bind port |
| `ANTHROPIC_SHIM_SESSION_HEADER` | `x-vsr-test-session-id` | request header used to scope prompt-cache state |
| `ANTHROPIC_SHIM_REQUEST_TIMEOUT` | `300` | upstream request timeout (seconds) |

Requests without the session header share a single global session,
which is fine for single-tenant test runs.

## Tests

```bash
pip install -e .[dev]
pytest
```

The unit tests cover every translation rule plus edge cases
(nil-safety, mixed content blocks, malformed payloads). The
end-to-end translation behaviour is also exercised against a stub
upstream that records every forwarded request.

[llama-server]: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
