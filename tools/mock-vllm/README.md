# Mock vLLM (OpenAI-compatible) service

A FastAPI provider simulator for the Router E2E suite:

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/responses

The request boundary is closed against the pinned OpenAI schema revision in
`schema_contract.json`. Published fields are accepted without being discarded,
unknown fields fail with an OpenAI error envelope, and nested provider objects are
kept intact. `GET /debug/last-request` exposes the last native provider body for a
test session so deployment tests can verify what Envoy and ExtProc actually sent.

Install `requirements-dev.txt` and run `pytest` to exercise every published
top-level request field, nested preservation, and strict unknown-field behavior.

`provider_boundary.py` owns HTTP validation and bounded request observation;
`chat_request.py` contains the small typed view used by deterministic Chat
responses. The provider contract itself remains in `provider_contract.py`.
