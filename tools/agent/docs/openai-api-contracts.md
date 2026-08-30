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

Request decoding has two explicit modes. An unchanged same-format codec call may
retain bounded source bytes for exact replay. Router ingress uses mutation-aware
strict decoding because routing always changes at least the provider model; a field
that cannot survive the neutral contract is rejected before policy runs instead of
being accepted and erased during dispatch.

## Built-in wire formats

| Format | Buffered codec | Stream codec | Compatibility tests |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `pkg/protocolcodec/codec_openai_chat.go` | `pkg/protocolcodec/stream_chat.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |
| OpenAI Responses | `pkg/protocolcodec/codec_openai_responses.go` | `pkg/protocolcodec/stream_responses.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |
| Anthropic Messages | `pkg/protocolcodec/codec_anthropic_messages.go` | `pkg/protocolcodec/stream_anthropic.go` | `pkg/protocolcodec/matrix_test.go`, `stream_contract_test.go` |

Each codec declares capabilities. Unsupported tools, multimodal content, reasoning,
structured output, multiple candidates, or streaming behavior must fail explicitly;
translation must not silently discard semantic fields.

OpenAI Responses hosted image generation is an explicit neutral capability. Its
request options, image tool choice, buffered output item, and four progress-event
variants are typed; Chat Completions and Anthropic targets reject that capability
instead of coercing it into ordinary image content.

Provider value domains remain provider-specific. Chat Completions permits a zero
output-token limit, Responses requires `max_output_tokens >= 16`, and Anthropic
permits `max_tokens: 0` for prompt-cache prewarming. A value that is valid at the
source but invalid at the target produces a typed capability error. Anthropic
enabled thinking additionally requires `budget_tokens >= 1024` and strictly below
`max_tokens`; its `display` control is preserved for both enabled and adaptive
thinking and fails explicitly for targets that cannot represent it.

The compatibility inventory is pinned to the upstream OpenAI OpenAPI schemas for
Chat Completions and Responses and the generated Anthropic Messages API types. The
closed-field tests in `pkg/protocolcodec/official_schema_contract_test.go` fail when
an upstream top-level field is neither represented by the wire codec nor classified
as a Router extension. The stream contract applies the same rule to Chat chunks,
Responses event discriminators, Anthropic event and delta unions, and protocol-native
terminal events. Semantically unsupported request fields fail with a typed
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
| `pkg/extproc/processor_res_transport_error.go` | Translate non-2xx provider bodies without treating them as model responses |
| `pkg/extproc/processor_res_semantic_stream.go` | Translate provider stream frames through neutral events |
| `pkg/extproc/processor_res_terminal.go` | Settle terminal response, usage, replay, cache, and response-side policy |
| `pkg/extproc/processor_response_object.go` | Handle stored Response objects without introducing wire state into policy |

The main processor files remain phase orchestrators. Provider normalization,
streaming accumulation, terminal settlement, cache persistence, replay persistence,
and warning shaping stay on separate seams.

Chat stream accounting has two independent contracts. ExtProc asks a Chat backend
for authoritative usage even when the public client did not request a usage chunk.
The original `stream_options.include_usage` preference remains in the neutral
request; same-format streams remove only the internally requested usage fields while
preserving provider extensions, and translated streams render usage exactly once
only when requested. Explicit OpenAI `include_obfuscation` is represented as a
transport preference and covered by both request and stream-output tests.

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
- non-object JSON documents, duplicate or unknown transport-error fields, and
  missing or whitespace-only error discriminators and messages;
- exact and case-folded duplicate JSON members before Go's case-insensitive
  struct matching can overwrite a routed or authenticated value;
- buffered request and response translation;
- transport error translation;
- status-aware ExtProc separation of non-2xx transport errors from successful or failed model-response resources;
- all source-to-target format pairs;
- split streaming frames, tool events, terminal events, and final usage;
- parallel tool calls with interleaved argument fragments and independently
  ordered item completion;
- hosted image-generation options, null-versus-empty results, strict base64 and
  size limits, monotonic progress, contiguous partial indexes, and matching item
  completion;
- arbitrary SSE byte boundaries, CR/LF variants, incomplete UTF-8 fragments,
  duplicate terminals, decreasing usage, cumulative stream-body limits, valid
  frames coalesced with a malformed trailing frame, and errors after partial
  output;
- UTF-8 validity across the complete SSE frame, comment-only keepalives versus
  explicit empty data events, and a poisoned state after the first stream
  failure;
- provider resource and event discriminator mismatches, unknown terminal enums,
  missing union members, contradictory lifecycle status, duplicate or sparse
  candidate indexes, missing or negative stream indexes, coupled terminal
  fields, null required members, non-UTF-8 JSON, tool identity changes,
  duplicate lifecycle events, output after a terminal event, response
  identity changes within one stream, unpaired Unicode surrogates, and usage
  counters whose known components overflow even when the source omits a total;
- unknown-field and unsupported-capability failure behavior;
- stateful Responses continuations whose retained history contains ordered
  messages, reasoning, function calls, and function outputs; materialization
  must reuse the Responses codec, reconcile deferred tool links, and reject a
  broken stored lifecycle without partially mutating the live request;
- missing `previous_response_id` history returns a typed not-found response,
  while an unavailable history store returns service unavailable; neither case
  may degrade into a stateless 200 response;
- trusted metadata isolation; and
- an ExtProc request/response regression using Envoy's normal transport boundary.

Adding a codec is incomplete until it is registered, declares capabilities, and
passes the complete pairwise matrix.

## Pairwise E2E matrix

Every client format is exercised against every backend format through Envoy and
ExtProc. Buffered and streaming contracts are separate gates; passing one does not
stand in for the other.

| Backend format | Chat client | Responses client | Messages client | E2E gate |
| --- | --- | --- | --- | --- |
| Chat Completions | native | translated | translated | `protocol-codec-chat-backend-{buffered,streaming}-matrix` |
| OpenAI Responses | translated | native | translated | `protocol-codec-responses-backend-{buffered,streaming}-matrix` |
| Anthropic Messages | translated | translated | native | `protocol-codec-anthropic-backend-{buffered,streaming}-matrix` |

The matrix rejects backend event leakage, requires exactly one protocol-native
terminal, checks final authoritative usage, and turns an incomplete upstream stream
into a failure terminal. `anthropic-chat-cache-control` additionally verifies that
OpenAI-compatible cache markers reach an Anthropic backend in buffered and streaming
requests and that a repeated buffered request reports cached input tokens.
The three `protocol-codec-*-backend-tool-lifecycle` gates send official flat
Responses function tools, consume the returned call, and submit an inline tool
result through each backend format. Both turns run once buffered and once as a
Responses SSE stream; the streamed gate checks lifecycle order, call identity,
fragment reconstruction, target-native terminals, and absence of backend wire
leakage.
The three `protocol-codec-*-backend-structured-output` gates send the complete
JSON Schema constraint through each client format to each native backend format.
Each client/backend pair runs buffered and streaming, and the backend echo proves
that routing mutation and provider encoding preserve the schema itself rather than
only returning a valid public response.
The three `protocol-codec-*-backend-error-matrix` gates force a provider-side
rate-limit response and require the original non-success status plus the native
error envelope for every client format. A backend failure must never become an
empty successful model response. The three
`protocol-codec-*-backend-midstream-error-matrix` gates deliver partial output
before a protocol-native provider error and require exactly one client-native
failure terminal with no later success terminal.
Failed Responses resources retain authoritative token usage when a provider reports
work completed before failure. Targets whose error envelope cannot carry usage emit
a bounded accounting diagnostic while settlement keeps the neutral usage evidence.

## Validation tiers

Protocol changes must pass every tier below. A codec unit test does not replace
the ExtProc seam or Envoy E2E gate.

| Tier | Contract | Required gate |
| --- | --- | --- |
| Published schema | Closed request, response, nested, tool, content, usage, and stream discriminator inventories | `go test ./pkg/protocolcodec -run 'Official'` |
| Reviewable wire goldens | Numbered request, response, transport-error, stream-transcript, target-specific capability, and typed-rejection JSON fixtures translated across every 3×3 protocol pair | `go test ./pkg/protocolcodec -run 'Golden'` |
| Neutral semantics | Buffered request, buffered response, tools, multimodal order, usage provenance, errors, and streaming for all 3×3 format pairs | `go test ./pkg/llmprotocol ./pkg/protocolcodec` |
| Data-plane seam | Client/backend orientation, provider model rewrite, API-root paths, buffered response, and streaming for all 3×3 pairs | `go test ./pkg/extproc` |
| Concurrency | Immutable registry plus parallel 3×3 request, response, transport-error, and stream translations proving request-scoped model, error, lifecycle, and accumulator state | `go test -race ./pkg/llmprotocol ./pkg/protocolcodec ./pkg/extproc` |
| Robustness | Malformed buffered and arbitrarily chunked stream input across all format pairs | protocol codec fuzz targets with a bounded CI or release time budget |
| Envoy transport | Native and translated buffered, streaming, tools, provider errors, incomplete streams, and midstream errors | `make e2e-test E2E_PROFILE=response-api` and `make e2e-test E2E_PROFILE=anthropic-shim` |
| Routing regression | Existing routing, cache, tool selection, replay, and streaming behavior | `make e2e-test E2E_PROFILE=envoy-ai-gateway` and `make e2e-test E2E_PROFILE=streaming` |

The published-schema tier currently closes three Chat Completions object
`tool_choice` discriminators, 30 Responses input-item variants, 28 Responses
output-item variants, three input-
content variants, two output-message content variants, one reasoning-content
variant, 16 Responses tool variants, 15 object `tool_choice` discriminators, and
58 Responses stream events. It also closes four Anthropic `tool_choice`
discriminators, 16 request content blocks, 12 response content blocks, 21 tools,
eight stream events, 12 stream content blocks, and five stream delta variants. Content unions
are validated by discriminator and position, so a known field from a different
variant cannot be accepted and then silently discarded by a shared wire struct.
The reviewable goldens live under
`pkg/protocolcodec/testdata/golden/{request,response,error,stream,capability,rejection}`.
Their numbered file names pair each source document with all three expected backend
documents, so wire-shape changes remain visible in code review instead of being
hidden in Go assertions. Capability fixtures record either a rendered body or a
stable typed error plus bounded fidelity diagnostics per target. The official-field
capability fixtures enumerate every top-level request and response field and are
closed against their wire types, preventing a new field from entering without a
reviewable result for every target protocol. Rejection fixtures
pin malformed unions, duplicate and trailing JSON, invalid tool lifecycles,
decreasing usage, out-of-order events, unknown discriminators and enums,
event/status contradictions, incomplete lifecycle objects, and stream identity
changes. Transport-error rejections separately pin malformed envelopes, every
non-object top-level JSON kind, unknown and duplicate fields, invalid field
types, absent details, and blank required values. Malformed Unicode escapes and
int64 accounting overflow are rejected before either can be normalized into a
different semantic value.

Value-domain fixtures also pin required-versus-null members, protocol-specific token
limits, coupled thinking controls, invalid enum combinations, strict JSON-object tool
arguments, duplicate nested keys, and failed-resource accounting. Capability cases
that are valid for only one provider are still exercised against every target so an
unsupported translation remains a reviewable result rather than an implicit drop.

Every stream golden is replayed both at its authored boundaries and one byte at a
time. Cancellation, deadline expiry, incomplete EOF, malformed trailing frames,
valid output preceding a malformed frame in the same transport chunk, partial
output followed by provider failure, Unicode splits, inclusive per-frame and
cumulative byte limits, fail-closed success terminals, and terminal idempotency
are part of the matrix contract. A successful provider terminal remains pending
until clean HTTP end-of-stream, closing the case where a later transport read
contains an unterminated malformed frame.

## Compatibility regressions

The protocol gate keeps focused tests for failure modes that are easy to reintroduce
during transport refactors:

| Contract | Required coverage |
| --- | --- |
| Flat Responses tools and inline tool results | buffered and streamed tool lifecycles across all source/target pairs and all three backend E2E profiles |
| Logical model aliases and provider model IDs | provider dispatch applies semantic controls before the provider ID rewrite |
| Provider API roots | Chat, Responses, and Messages paths are joined once without duplicating `/v1` |
| Official unions | request, response, content, media-source, tool, and stream variants decode strictly; unsupported official discriminators fail with typed capability errors |
| Tool selection | one neutral mutation before the final provider encoding for every backend format |
| `tool_choice` | omitted means `auto`; `auto`, `none`, `required`, and named choices survive every protocol pair |
| Cache directives | same-format mutation, Chat-to-Messages translation, malformed input, cache usage, and unsupported block errors |
| Structured output | the complete JSON Schema survives routing mutation |
| Anthropic `output_config` | reasoning effort and strict JSON Schema survive same-format mutation and OpenAI translation |
| Reasoning streams | both OpenAI-compatible reasoning aliases and ordered Anthropic thinking/signature blocks remain explicit |
| Terminal reasons | Anthropic pause/context-window stops and Responses content-filter incompletes remain exact or fail explicitly when the target cannot represent them; an Anthropic matched stop sequence is a separate response capability and can never silently collapse into a generic OpenAI `stop` |
| Stream indexes | Source-private neutral indexes are compacted by target encoders, while provider output indexes must be non-negative, contiguous, and bound to one active item identity |
| Tool streams | Tool IDs and names are present before publication, immutable for the item lifetime, bounded, and paired with one final JSON-object argument value |
| Parallel tool streams | Each call retains an independent identity, argument buffer, lifecycle, and compact target index even when deltas interleave or items finish out of order |
| Hosted image generation | Every published Responses option, tool choice, output item, and progress event is typed; result presence, progress order, partial indexes, item identity, and terminal state are validated, while unsupported targets fail explicitly |
| Retained Responses tools | `previous_response_id` materializes the complete stored call/result/reasoning sequence through the neutral codec before provider encoding; missing history and orphaned stored results fail closed |
| Responses stream ordering | Every event carries a contiguous `sequence_number`; an official response terminal is final and `[DONE]` is rejected as a Chat-only sentinel |
| Custom tools | Unsupported custom or freeform tool discriminators fail before inference and are never coerced into function calls |
| Stream usage | intermediate evidence cannot terminate content or create duplicate usage events; backend accounting cannot change public `include_usage` behavior |
| Incomplete streams | every client format receives one failure terminal and never a success terminal |
| Provider errors | buffered and streaming errors are translated to the client protocol without model-output synthesis |

These tests validate semantic behavior rather than serialized field coincidence. A
same-format request is forced through a routing mutation where relevant so envelope
replay cannot hide a missing neutral field.

### Active regression ledger

Protocol refactors must retain an executable regression for each accepted or open
wire-contract defect. The ledger records behavior, not implementation ownership, so
it remains valid when codec internals move.

| Regression | Unit or seam contract | Envoy E2E contract |
| --- | --- | --- |
| [#2492](https://github.com/vllm-project/semantic-router/issues/2492) Responses streaming must preserve its public event taxonomy, tools, reasoning, refusal, headers, cache, and continuity across backend formats | complete official event inventory, 3×3 stream matrix, request-context propagation, and response-header rewrite tests | Responses-native plus Chat- and Messages-backed streaming matrices |
| [#2486](https://github.com/vllm-project/semantic-router/issues/2486) FullDuplexStreamed must not replace an in-flight request with an empty body | non-EOS ExtProc responses defer mutation and EOS emits one complete body | `agentgateway-full-duplex-multiturn` |
| [#2316](https://github.com/vllm-project/semantic-router/issues/2316) provider SSE frames may split at arbitrary Envoy chunk boundaries | SSE carry-buffer, one-byte replay, CR/LF, and split UTF-8 tests | three backend streaming matrices |
| [#3045](https://github.com/vllm-project/semantic-router/pull/3045) Responses clients must receive Responses SSE when the selected backend speaks Anthropic Messages | complete Responses event taxonomy and 3×3 stream matrix | Anthropic backend streaming matrix |
| [#913](https://github.com/vllm-project/semantic-router/issues/913) a buffered semantic-cache entry must replay as valid SSE | neutral cached response is encoded through the client stream codec | `streaming-sse-cache` requires a cache hit, valid SSE, matching content, and one `[DONE]` terminal |
| [#2846](https://github.com/vllm-project/semantic-router/pull/2846) intermediate usage chunks must not terminate or truncate streaming content | per-chunk usage continuation and one-terminal stream-state tests | `streaming` profile |
| [#3068](https://github.com/vllm-project/semantic-router/issues/3068) Anthropic backend profile bypasses a second protocol processor and runs in affected-change CI | profile topology and classifier tests | `anthropic-shim` profile |
| [#3067](https://github.com/vllm-project/semantic-router/issues/3067) flat Responses tools and inline call results | 3×3 request and tool-history matrices | three backend tool-lifecycle cases |
| [#3065](https://github.com/vllm-project/semantic-router/issues/3065) provider API-root joining | protocol × base-URL × custom-path table | all three backend profiles use their native route |
| [#3064](https://github.com/vllm-project/semantic-router/issues/3064), fixed by [#3066](https://github.com/vllm-project/semantic-router/pull/3066), provider model ID rewrite | request dispatch tests for Chat, Responses, and Messages targets | buffered 3×3 backend matrices |
| [#3063](https://github.com/vllm-project/semantic-router/issues/3063) midstream provider failure | 3×3 stream failure matrix | three backend midstream-error matrices |
| [#3058](https://github.com/vllm-project/semantic-router/issues/3058) and [#2991](https://github.com/vllm-project/semantic-router/pull/2991) provider error must not become an empty success | transport-error 3×3 matrix and ExtProc response seam | three backend error matrices |
| [#3056](https://github.com/vllm-project/semantic-router/issues/3056) ordered Anthropic thinking and signatures | buffered and streaming order/signature tests | Anthropic buffered and streaming matrices |
| [#3055](https://github.com/vllm-project/semantic-router/issues/3055) cache directives survive Chat-to-Messages translation | cache-directive schema and pair tests | `anthropic-chat-cache-control` |
| [#3052](https://github.com/vllm-project/semantic-router/issues/3052) omitted tool choice has `auto` semantics | semantic-default and four-mode 3×3 matrices | tool-selection plus backend tool lifecycle |
| [#3051](https://github.com/vllm-project/semantic-router/issues/3051) tool selection precedes provider encoding | exactly-one neutral mutation seam test | backend tool lifecycle for every target format |
| [#3024](https://github.com/vllm-project/semantic-router/issues/3024) routed model rewrites retain the complete JSON Schema contract | structured-output 3×3 goldens and `TestExtProcStructuredOutputRequestProtocolMatrix` | three `protocol-codec-*-backend-structured-output` cases, each buffered and streamed |
| [#3013](https://github.com/vllm-project/semantic-router/issues/3013) Responses clients never receive Chat SSE from an Anthropic backend | client/backend-independent 3×3 stream encoder matrix | Messages-backed Responses streaming profile |
| [#2947](https://github.com/vllm-project/semantic-router/issues/2947) streamed cache accounting survives intermediate and terminal usage updates | cumulative usage-state tests plus Anthropic-to-every-target cache matrix | cached Messages stream through native backend profile |

Adding or fixing a protocol issue updates this ledger and the corresponding test in
the same change. Closing an issue does not permit deleting its regression.

Authoritative upstream contracts used by the current inventory:

- OpenAI OpenAPI at `690521b1753dce0c6d6b275f583d22537679cff9`:
  <https://github.com/openai/openai-openapi>
- Anthropic Go SDK generated API types at
  `d19dea9ed85bbb5fdb2d6f20fb6f903920ed23fa`:
  <https://github.com/anthropics/anthropic-sdk-go>

### Advancing a schema revision

A schema refresh is a contract change, not a generated-code update. Pin the new
upstream revision in the top-level and nested inventory tests, classify every
added field or union discriminator as semantic, transport-only, diagnostic, or
typed unsupported behavior, and add literal JSON evidence for it. Regenerate
all three target outputs for every affected input, review the diff, then run the
buffered, streaming, arbitrary-chunk, race, bounded-fuzz, ExtProc, and Envoy E2E
gates. A revision must not merge with unclassified fields, stale outputs, or a
same-format-only test that leaves cross-protocol behavior undefined.
