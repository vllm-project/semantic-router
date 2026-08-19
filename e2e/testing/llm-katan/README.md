# LLM Katan

LLM Katan is a small OpenAI-shaped server used by Semantic Router tests. It can
load a tiny model or return deterministic fixture responses, which makes it
useful when a test needs a model endpoint without a production serving stack.

It implements the subset of the Chat Completions API used by this repository.
It is not a production server or a general drop-in replacement for an OpenAI
API.

## Choose a backend

| Backend | Use it for | Additional requirements |
| --- | --- | --- |
| `echo` | Inspecting messages sent by the router and deterministic memory tests | None |
| `transformers` | Exercising real generation with a small Hugging Face model | Model download and PyTorch |
| `vllm` | Exercising the same fixture API with a local vLLM engine | Install the `vllm` extra and provide suitable hardware |

All backends expose `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions`, and `GET /metrics`. Interactive FastAPI
documentation is available at `/docs`.

## Run locally

From the repository root, install the package in an isolated environment:

```bash
python3 -m venv .venv-llm-katan
. .venv-llm-katan/bin/activate
python -m pip install -e e2e/testing/llm-katan
```

Start the download-free echo backend. `--model` is required by the CLI even
when the echo backend is selected:

```bash
llm-katan \
  --model fixture-model \
  --served-model-name fixture-model \
  --backend echo \
  --host 127.0.0.1 \
  --port 8000
```

In another shell, verify the endpoint:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "fixture-model",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16
  }'
```

The echo response includes every inbound message. Tests use that behavior to
check prompt, memory, and system-message injection without log scraping.

### Use a real tiny model

```bash
llm-katan \
  --model Qwen/Qwen3-0.6B \
  --served-model-name qwen-test \
  --device cpu \
  --port 8000
```

The first run downloads the model. Authentication is only needed for gated or
private repositories; follow the model host's credential guidance rather than
putting a token in a command or document. Dynamic int8 quantization is enabled
by default on CPU and falls back to full precision when the platform does not
support it. Use `--no-quantize` when a test requires full-precision behavior.

The Transformers and vLLM backends run model-supplied code with
`trust_remote_code=True`. Only load repositories you trust.

## Run the container fixture

The nightly image is convenient for integration tests:

```bash
docker run --rm -p 8000:8000 \
  ghcr.io/vllm-project/semantic-router/llm-katan:nightly
```

`nightly` is a mutable development tag. Pin a dated or digest-qualified image
when repeatability matters. The default container command downloads
`Qwen/Qwen3-0.6B`; override the command to use the echo backend:

```bash
docker run --rm -p 8000:8000 \
  ghcr.io/vllm-project/semantic-router/llm-katan:nightly \
  llm-katan --model fixture-model --backend echo --host 0.0.0.0
```

For the Kubernetes fixtures, see the
[LLM Katan deployment README](../../../deploy/kubernetes/llm-katan/README.md).

## Simulate multiple provider model names

Several servers can load the same small model while advertising different
names. This tests provider and model selection; it does not simulate the
quality or protocol differences of those providers.

```bash
# Shell 1
llm-katan --model Qwen/Qwen3-0.6B --port 8000 \
  --served-model-name gpt-test

# Shell 2
llm-katan --model Qwen/Qwen3-0.6B --port 8001 \
  --served-model-name claude-test
```

## Configuration

Run `llm-katan --help` for the complete option list. The most relevant options
are `--model`, `--served-model-name`, `--backend`, `--device`, `--host`,
`--port`, `--max-tokens`, `--temperature`, and
`--quantize/--no-quantize`. Supported devices are `auto`, `cpu`, `cuda`, and
`xpu`.

The process also accepts these environment overrides:

| Variable | Overrides |
| --- | --- |
| `YLLM_MODEL` | `--model` |
| `YLLM_SERVED_MODEL_NAME` | `--served-model-name` |
| `YLLM_BACKEND` | `--backend` |
| `YLLM_HOST` | `--host` |
| `YLLM_PORT` | `--port` |

Command-line values are parsed first, then these variables are applied.

## Development

Install development dependencies from this directory:

```bash
python -m pip install -e '.[dev]'
```

This package currently has no standalone test directory. Its behavior is
covered through the repository's E2E profiles and container workflows; add
unit tests with any behavior change that can be exercised without the full
stack.
