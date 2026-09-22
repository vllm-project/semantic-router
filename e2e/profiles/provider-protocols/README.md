# Provider protocol compatibility

This profile routes client Chat Completions, Responses, and Anthropic Messages
through Envoy and the Router to provider-mocker's native Anthropic Messages
endpoint. Together with `response-api`, it exercises every client/backend
protocol pairing in buffered and streaming mode, plus tool lifecycles,
structured output, usage, cache counters, and provider errors.

```bash
make e2e-test E2E_PROFILE=provider-protocols
```

The E2E framework builds and loads the local fixture image, or consumes the
qualified image prepared by CI. The backend contains no model download, model
cache volume, llama.cpp server, or translation sidecar. The Router owns the
protocol translation under test. Fixture requests are observable at
`/debug/last-request`, scoped by `x-vsr-test-session-id`.
