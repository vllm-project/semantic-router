# Inference Protocol Contracts

Inventory of client-facing inference types, neutral Router types, and wire codecs.
Architecture rules live in
[architecture-guardrails.md](architecture-guardrails.md#api-type-contracts).

## Runtime boundary

The inference hot path has three type layers:

1. `pkg/protocolcodec` parses or renders a supported wire format.
2. `pkg/llmprotocol` carries protocol-neutral semantic state.
3. `pkg/extproc` evaluates routing and plugin policy against neutral state while
   Envoy owns upstream transport.

Router policy must not mutate provider SDK or wire structs. Provider-specific JSON,
SSE events, stop reasons, and error envelopes belong in codecs.

## Built-in wire formats

| Format | Buffered codec | Stream codec | Compatibility tests |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `pkg/protocolcodec/codec_openai_chat.go` | `pkg/protocolcodec/stream_chat.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |
| OpenAI Responses | `pkg/protocolcodec/codec_openai_responses.go` | `pkg/protocolcodec/stream_responses.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |
| Anthropic Messages | `pkg/protocolcodec/codec_anthropic_messages.go` | `pkg/protocolcodec/stream_anthropic.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |

Each codec declares capabilities. Unsupported tools, multimodal content, reasoning,
structured output, multiple candidates, or streaming behavior must fail explicitly;
translation must not silently discard semantic fields.

The compatibility inventory is pinned to the upstream OpenAI OpenAPI schemas for
Chat Completions and Responses and the generated Anthropic Messages API types. The
closed-field tests in `pkg/protocolcodec/official_schema_contract_test.go` fail when
an upstream top-level field is neither represented by the wire codec nor classified
as a Router extension. Semantically unsupported request fields fail with a typed
`unsupported_feature` error; provider response metadata that is not model output is
reported through bounded diagnostics.

## Neutral model

| Location | Contract | Notes |
| --- | --- | --- |
| `pkg/llmprotocol/types.go` | Requests, responses, messages, content, tools, sampling | No wire JSON or provider transport |
| `pkg/llmprotocol/stream.go` | Neutral stream events and request-scoped codec interfaces | Enforces terminal and usage semantics |
| `pkg/llmprotocol/errors.go` | Typed protocol and transport errors | HTTP errors remain distinct from model response resources |
| `pkg/llmprotocol/capabilities.go` | Codec feature declarations and requirements | Checked before target encoding |
| `pkg/llmprotocol/policy.go` | Bounds and unknown-field policy | Same-format preservation is never a cross-format fallback |

Trusted identity and correlation metadata is populated by the transport boundary,
not decoded from client-controlled fields.

## ExtProc integration

| Location | Responsibility |
| --- | --- |
| `pkg/extproc/processor_protocol_contract.go` | Detect source and target wire contracts and hold the request-scoped codec state |
| `pkg/extproc/processor_req_body.go` | Decode once, invoke neutral request policy, encode for the selected backend |
| `pkg/extproc/processor_res_body.go` | Decode provider output and encode the client response |
| `pkg/extproc/processor_res_semantic_stream.go` | Translate provider stream frames through neutral events |
| `pkg/extproc/processor_res_terminal.go` | Settle terminal response, usage, replay, cache, and response-side policy |
| `pkg/extproc/processor_response_object.go` | Handle stored Response objects without introducing wire state into policy |

The main processor files remain phase orchestrators. Provider normalization,
streaming accumulation, terminal settlement, cache persistence, replay persistence,
and warning shaping stay on separate seams.

## Other API surfaces

These APIs are intentionally outside the inference codec matrix:

| Surface | Location | Contract |
| --- | --- | --- |
| Models list | `pkg/publicmodels` | OpenAI-compatible model list plus bounded Router discovery metadata |
| Stored Responses service | `pkg/responseapi` | Persistence and object lifecycle behind the Responses endpoints |
| Files and vector stores | `pkg/openai` | Narrow outbound clients covered by wire compatibility tests |
| Image generation backend | `pkg/imagegen` | Plugin-owned outbound contract |
| E2E request fixtures | `e2e/pkg/fixtures` | Test-isolated minimal wire structures |

## Regression requirements

Any change to `pkg/llmprotocol`, `pkg/protocolcodec`, or the ExtProc protocol seam must
cover:

- malformed and bounded input;
- buffered request and response translation;
- transport error translation;
- all source-to-target format pairs;
- split streaming frames, tool events, terminal events, and final usage;
- unknown-field and unsupported-capability failure behavior;
- trusted metadata isolation; and
- an ExtProc request/response regression using Envoy's normal transport boundary.

Adding a codec is incomplete until it is registered, declares capabilities, and
passes the complete pairwise matrix.

Authoritative upstream contracts used by the current inventory:

- OpenAI OpenAPI at `172101000e7be21103c405aa8bedf918039f886f`:
  <https://github.com/openai/openai-openapi>
- Anthropic Go SDK generated API types at
  `f6f796100d7bb958d84580f44060a0a2b21bfe04`:
  <https://github.com/anthropics/anthropic-sdk-go>
