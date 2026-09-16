# Mock vLLM (OpenAI-compatible) service

A FastAPI provider simulator for the Router E2E suite:

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/responses
- POST /classify

The request boundary is closed against the pinned OpenAI schema revision and the
pinned vLLM OpenAI-compatible provider revision in `schema_contract.json`.
Official protocol fields, Router protocol extensions, and provider-only fields are
tracked separately. Published fields are accepted without being discarded,
provider-only fields are type checked, unknown fields fail with an OpenAI error
envelope, and nested provider objects are kept intact. `GET /debug/last-request`
exposes the last native provider body for a test session so deployment tests can
verify what Envoy and ExtProc actually sent.

Install `requirements-dev.txt` and run `pytest` to exercise every published and
provider-native top-level request field, nested preservation, and strict
unknown-field behavior.

`provider_boundary.py` owns HTTP validation and bounded request observation;
`chat_request.py` contains the small typed view used by deterministic Chat
responses. The provider contract itself remains in `provider_contract.py`.
`workflow_chat.py` owns Router Flow planner/worker chat replies.
`classify.py` serves the `prompt_guard` `http_classify` stand-in on `/classify`;
it scores only the first window of the posted text, so the response-jailbreak E2E
can tell a whole-response scan from a first-chunk-only one.

## Shadow failure fixture

`MOCK_VLLM_SHADOW_CONTROL=true` enables the `router-replay` profile's controlled
Chat Completions fixture. Normal requests return `Hello from <model>.` through
the existing response builder. Other profiles keep the normal request echo.

`POST /debug/shadow/{scenario}/reset` accepts `{"mode":"healthy"}`,
`{"mode":"hold"}`, or `{"mode":"malformed"}`. Scenarios are `timeout`, `malformed`,
and `queue`, corresponding to `openai/shadow-<scenario>` models.
`GET /debug/shadow/{scenario}` reports received and active requests, barrier
expirations, and the last eight request IDs. A reset fails while a request is
active.

`hold` waits on an asynchronous release event, so the fixture can still serve
primary requests and status reads. `POST /debug/shadow/{scenario}/release`
releases the held requests and restores healthy responses. A forgotten barrier
expires after 60 seconds with HTTP 504 and an expiration count; E2E cases reject
that result. `malformed` returns HTTP 200 with a JSON object where the Chat
Completions response requires a `choices` array.

Run the fixture tests with the existing pytest suite, without setting
`MOCK_VLLM_SHADOW_CONTROL` in the test process. The tests create an isolated
controlled app and also exercise the default app with controls disabled.
