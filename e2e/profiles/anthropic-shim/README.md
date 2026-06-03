# anthropic-shim profile

## What this profile is for

This profile exercises the outbound emitter's Anthropic-shaped response
path end-to-end by routing requests at a backend that natively speaks
the Anthropic Messages API: `llama.cpp` (`llama-server`) behind the
`anthropic-shim` Python proxy.

It is needed because the baseline `kubernetes` profile routes Anthropic
clients at an OpenAI-shaped backend (the mock-vLLM simulator), so the
emitter's cache-token propagation and stop-reason mapping paths are only
partially exercised there. Tests that assert on
`usage.cache_creation_input_tokens` or `stop_reason == "stop_sequence"`
must run against a backend that actually synthesises those fields.

## What it deploys

| Component | Description |
| --- | --- |
| Envoy Gateway + Envoy AI Gateway | Shared gateway stack (same as `kubernetes` / `multi-endpoint`) |
| Semantic Router (ExtProc) | Built locally (`e2e-test` image tag) |
| `anthropic-backend-qwen` | `llama-server` + `anthropic-shim` sidecar in `anthropic-backend-system` namespace |

The `anthropic-shim` sidecar translates system arrays, tool-result
content, and synthesises prompt-cache token counters from
`x-vsr-test-session-id`-scoped prefix hashes so tests can assert on
cache-creation vs cache-read cycles without a real Anthropic API key.

## When to use it

Run this profile when adding or validating tests for:

- `usage.cache_creation_input_tokens` / `usage.cache_read_input_tokens`
- `stop_reason == "stop_sequence"` vs `"end_turn"`
- Request-side header and field preservation (via `/debug/last-request`)
- Any test in `testmatrix.AnthropicShimContract`

## Running the profile

```bash
make e2e-test PROFILE=anthropic-shim
```
