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
| `POST /v1/systemone` | System One decisions for the Router's `systemone` deployments |
| `GET /v1/systemone/models` | The one System One model it lists, `e2e/fake-systemone` |
| `GET /debug/last-request` | Last native request per test session |

Images accept `response_format: b64_json` and `n` from 1 to 4. Images streaming,
URL results, image editing and unsupported fields fail explicitly. The fixed PNG
exercises response framing and image decoding; it does not simulate image quality.

`provider_mocker/schema_contract.json` keeps the pinned OpenAI/Anthropic field
inventories and provider extensions. Unknown fields return the native protocol's
error envelope. Request observation preserves the original JSON, the request path,
a SHA-256 digest of the `Authorization` header (not the credential) and only the
`x-vsr-test-session-id` and `x-e2e-*` headers. Sessions use that header or query
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
(`__mock_midstream_error__`). Messages also recognizes `__mock_protocol_matrix__`
and `__mock_gateway_fields__`, which adds fields no Messages schema declares: a
`usage.total_cost` in exponent form and a top-level `gateway_trace_id`.

## System One

`/v1/systemone` answers the questions the Router compiles its application
signals into, for a `systemone` deployment with
`endpoint: http://<service>:8000/v1/systemone` and `model: e2e/fake-systemone`.
It takes a standard request (`state` and `questions`) or a batch (`states`
alone, answered entry by entry under the same names) and, like the model
runtime, refuses a body that mixes the two with `invalid_request`.
It runs no model and reads all of every text; markers in the text a question
reads pin its answer:

| Input | Answer |
| --- | --- |
| `__PII_SPAN__ <LABEL> <word>` | A span question offering `LABEL` (pii) finds it over the first copy of `<word>` |
| `__SYSTEMONE_CHOICE__ <key>` | A choice question offering `key` (a turn_relation label, say) chooses it |
| `Ignore all previous instructions` | A choice question offering `jailbreak` chooses it |
| `__SYSTEMONE_YES__` | A noul question answers true |
| `__SYSTEMONE_REJECT__` | A noul question about a candidate (a match judge's, such as `same_request`) answers false; any other noul question true |
| An answer sentence its context lacks | A span question offering `unsupported` (hallucination) finds the first one |
| `__SYSTEMONE_UNREAD__` | A `require_full_input` question fails with `max_length_exceeded` |
| `__SYSTEMONE_UNAVAILABLE__` | The call fails with 503 |

A state's images (PNG data URLs) carry markers in their `tEXt` chunks, which
every question of that state reads with its text. The Router sends a request's
images only to the questions whose rules read them (`input.image`), so a
marker in an image reaches exactly those questions.

Otherwise a choice question answers its first option, a noul false and a span
question nothing. Choice, noul and span are the question types it lists; any
other fails with `invalid_question`. A failed question carries its error in
its answer, as a model runtime answers it, and the call's other questions are
answered.

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
