# Provider mocker

A deterministic provider service for Router tests and examples. It runs without
model downloads, model weights, GPUs, or an upstream inference server. Its native
wire handlers are independent of the production Router codec.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m provider_mocker --host 127.0.0.1 --port 8000
.venv/bin/python -m pytest
```

Run these commands from `tools/test/services/provider-mocker`. From another
working directory, add this directory to `PYTHONPATH`, or install this package.

## Native endpoints

| Endpoint | Contract |
| --- | --- |
| `POST /v1/chat/completions` | Chat Completions, buffered and SSE |
| `POST /v1/responses` | Responses, buffered and SSE, including image-generation events |
| `POST /v1/messages` | Messages, buffered and SSE, tools, stop sequences and cache counters |
| `POST /v1/images/generations` | Buffered Images with deterministic valid 1×1 PNGs |
| `GET /health` | Readiness |
| `GET /v1/models` | Configured model metadata |
| `POST /classify` | Remote classifier and response guard fixture |
| `GET /debug/last-request` | Last native request per test session |

Images accept `response_format: b64_json` and `n` from 1 to 4. Images streaming,
URL results, image editing and unsupported fields fail explicitly. The fixed PNG
exercises response framing and image decoding; it does not simulate image quality.

`provider_mocker/schema_contract.json` keeps the pinned OpenAI/Anthropic field
inventories and provider extensions. Unknown fields return the native protocol's
error envelope. Request observation preserves the original JSON and only the
`x-vsr-test-session-id` and `x-vsr-e2e-*` headers. Sessions use that header or query
parameter and otherwise share `__global__`. Observation and cache state are bounded
and local to the single service worker.

## Scenarios

Set `PROVIDER_MOCKER_SCENARIO` before starting the process:

| Value | Behavior |
| --- | --- |
| `default` | Native protocol echoes, Router Flow planner/worker fixtures and protocol markers |
| `memory` | Full message echo, deterministic memory fact extraction and query rewriting |
| `looper` | Fixed ratings/confidence/fusion responses and token counts; `/test/calls` and `/test/reset` |
| `hallucination` | Keyword-selected grounded or deliberately inconsistent text |
| `toolcall` | Web-search tool round trip, deliberately inconsistent answer and creative bypass |
| `cli` | `ok` replies, provider base-path handling and authorization canary observations |

Scenarios select Chat Completions behavior. The other native endpoints retain
their protocol fixtures. `PROVIDER_MOCKER_MODEL` selects the model advertised by
`/v1/models`; responses preserve the requested model.

`PROVIDER_MOCKER_EXPECT_AUTHORIZATION` holds an expected bearer **token**. Requests
must send `Authorization: Bearer <token>`; a successful match logs
`authorization-canary-received` without exposing the token. The CLI scenario logs
the received path and accepts prefixed paths ending in `/chat/completions`.

`PROVIDER_MOCKER_RESPONSE_DELAY_MS` adds asynchronous controlled work before native
responses. `PROVIDER_MOCKER_SHADOW_CONTROL=true` enables `/debug/shadow/{scenario}`
with `timeout`, `malformed` and `queue` scenarios. POST `/reset` accepts `healthy`,
`hold` or `malformed`; POST `/release` frees held requests. A hold expires after
60 seconds, increments an observable expiration counter and returns HTTP 504.

Protocol marker inputs select tool calls (`__mock_tool_call__`), structured schema
echo (`__mock_structured_output__`), HTTP 429 (`__mock_provider_error__`), incomplete
SSE (`__mock_incomplete_stream__`) and partial-output errors
(`__mock_midstream_error__`). Messages also recognizes `__mock_protocol_matrix__`.

## Image inputs

The Docker build context is this directory. The image copies only the pinned
`requirements.txt` and `provider_mocker/` runtime package. Tests, docs and local
virtual environments are excluded. Dependency updates regenerate both lock files:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip compile requirements-dev.in -o requirements-dev.txt
```

Real small-model qualification is a separate optional workflow in
`tools/test/services/tiny-model`; no inference engine is bundled into this fixture.
