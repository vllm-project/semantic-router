# Protocol translation goldens

These fixtures make the public protocol matrix reviewable without reading Go
structs. Each input is translated through the neutral contract after changing
the requested model to `routed-model`; one expected output exists for every
built-in backend protocol.

File names use:

```text
NNN-{client-protocol}-{case}-in.json
NNN-{client-protocol}-{case}-{backend-protocol}-out.json
```

The protocol tokens are `chat`, `responses`, and `anthropic`. Request,
response, transport-error, streaming, and typed-rejection fixtures live in
separate directories. The `capability` directory records a successful wire
body or a stable typed error independently for each target, so protocol-only
features cannot disappear in an otherwise green matrix. Its official-field
fixtures contain one named case for every top-level request and response field;
tests close those case names against the codec wire structs, so a new field
cannot bypass a human-readable translation decision. Add the input first, then
review all three generated
outputs with:

```bash
UPDATE_PROTOCOL_GOLDENS=1 go test ./pkg/protocolcodec -run Golden
git diff -- pkg/protocolcodec/testdata/golden
```

Nested objects and unions are indexed in `testdata/contracts`. Those JSON
inventories are closed against the Go wire types and point to the concrete
request, response, error, or stream inputs that exercise each object. This
keeps complete field coverage inspectable without duplicating one enormous
payload or hiding the contract in reflection-only tests. CI recursively reads
those fixtures, including JSON embedded in SSE `data:` records, and requires
literal human-readable evidence for every declared nested field. Capability
files may group named stream cases; every case is translated to all three
targets and replayed one byte at a time independently.
`testdata/contracts/matrix-cases.json` separately requires Chat, Responses,
and Messages source inputs for every core buffered and streaming scenario; each
listed input is still translated to all three targets by the golden harness.

Generation is opt-in. CI only compares fixtures and fails on any semantic JSON
drift. It also requires an exact three-target output inventory for every input,
so missing results and stale files from removed cases both fail the gate. The
numbered cases intentionally cover basic text, tool lifecycles,
ordered multimodal content, structured output, usage, provider failures,
malformed unions, duplicate/trailing JSON, unknown stream events, incomplete,
canceled and deadline-exceeded streams, explicit stream usage/obfuscation
options, errors after partial output, invalid provider resource discriminators,
unknown terminal enums, contradictory response status, missing union members,
missing or negative stream indexes, duplicate candidate indexes, coupled stop
reason/sequence mismatches, null required members, sparse output indexes,
mutable or reused item/tool identities, duplicate lifecycle events, output after a terminal
event, SSE event/type mismatches, response identity changes within a stream,
unpaired UTF-16 surrogates, and token-accounting overflow with or without a
declared total. Empty and whitespace-only bodies, invalid top-level JSON kinds,
unknown top-level and nested members, duplicate keys, trailing and truncated
JSON documents, UTF-8 byte-order marks, missing required model fields,
fractional integers, and integer overflow are pinned as stable typed failures.
Case-folded duplicate names are rejected too, preventing Go's
case-insensitive struct matching from turning `model` and `Model` into an
order-dependent overwrite.
Malformed transport-error envelopes are also explicit rejection cases: missing
or null error details, missing error types or messages, whitespace-only required
members, and an invalid Anthropic top-level discriminator can never be
reinterpreted as a successful model response.
Successful stream inputs cover a leading UTF-8 BOM, while a BOM before any
later event is a stable typed rejection,
both LF and CRLF framing, and legal SSE keepalive comments. Invalid UTF-8 in
any SSE field is rejected, and an explicit empty `data:` event is not confused
with a comment-only keepalive. The stream contract also accepts a final valid
SSE event delimited by transport EOF instead of an extra blank line. A provider
success terminal is published only
after the HTTP body reaches a clean end-of-stream; malformed trailing bytes can
therefore never escape behind an already-visible success sentinel. Responses streams
also reject missing or non-contiguous `sequence_number` values, a non-zero
initial sequence, null/non-integer/overflowed sequence values, and the
Chat Completions-only `[DONE]` sentinel. Stream response and model identities
are immutable after the first provider event for all three protocols. Exact
allowed values and one-step-outside boundary cases cover
sampling probabilities, penalties, candidate counts, stop sequences, output
token limits, Anthropic thinking budgets, and malformed numeric types. The
successful stream matrix also includes parallel tool calls whose argument
fragments arrive interleaved and whose items complete in a different order, so
call identity and argument buffers cannot bleed between items. A mixed
reasoning, tool-call, and final-text transcript pins output-item ordering across
all target protocols, and terminal Responses resources retain the completed
output array rather than forcing clients to reconstruct it from deltas.
Responses content fixtures also pin the complete structural lifecycle: content
parts start contiguously, deltas and annotations precede their matching done
events, item snapshots match the accumulated content, function-call arguments
finish before the item, and terminal output is identical to the ordered set of
completed items. A valid stream that pauses one output item while another item
finishes records the target-specific result explicitly; targets whose wire
grammar cannot preserve that interleaving fail with a typed capability error.
Buffered response inputs include native function calls
from all three source protocols, not only translated request histories. Every
stream fixture is additionally replayed one byte at a time to prove that
transport chunking cannot change its semantic output; native multilingual
fixtures make this include splits inside UTF-8 code points. Stream inputs
retain their deliberate byte boundaries; stream outputs
are order-preserving JSON transcripts of the exact SSE event and data frames,
including terminal frames. Random obfuscation values are normalized to
`<random>` in expected transcripts while their presence remains contractual.
The configured body limit is inclusive for buffered documents and is also the
cumulative upstream byte limit for a stream, so many individually valid SSE
frames cannot bypass the same resource bound. Reviewable rejection fixtures
pin request, response, transport-error, and streaming body limits for every
source protocol, plus per-frame SSE and cumulative semantic-event limits; the
unit matrix separately proves that the exact configured boundary remains
accepted.

Stateful Responses continuation is covered at the ExtProc seam rather than by
pairwise wire goldens. Stored message, reasoning, function-call, and
function-output items are replayed through the same Responses codec before
backend encoding. Deferred call links are reconciled only after the complete
history is present; missing calls, malformed stored items, or partial
materialization fail closed. A nonexistent previous response returns not found,
and an unavailable history store returns service unavailable; stateful requests
never silently degrade into stateless generation. The deployment-level tool
lifecycle suite continues a stored function call using only
`previous_response_id` plus the function output, in both buffered and streamed
mode, against each backend protocol. Test backends accept the result only when
the matching call ID is present earlier in the reconstructed history.

Capability cases also cover response-only fidelity. In particular, the exact
sequence matched by an Anthropic `stop_sequence` terminal is preserved for a
Messages target and produces a typed capability error for target formats that
only expose a generic stop reason. Signed reasoning, refusal, citations, and
reasoning streams are also translated or rejected explicitly for every target.
The complete published Responses and Messages tool discriminator inventories
are represented as named cases: supported function/custom tools translate, and
server-side or freeform tool kinds without a neutral contract fail explicitly
instead of being coerced into a different tool kind.
Stream capability cases are replayed with the same one-byte fragmentation
check as ordinary successful streams.
