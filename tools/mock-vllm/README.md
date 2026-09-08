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
`classify.py` serves the `prompt_guard` `http_classify` stand-in on `/classify`;
it scores only the first window of the posted text, so the response-jailbreak E2E
can tell a whole-response scan from a first-chunk-only one.
