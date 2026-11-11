---
title: System One API schemas
description: Precise, versioned schemas for every decision-model API the Router serves, calls or forwards to, one API registry for listeners and upstreams, and one codec layer between them.
---

# System One API schemas

**Status:** implemented.

## Goals

1. **Precise, versioned schemas everywhere.** Every decision-model API the
   Router exposes, calls internally or calls in a backend pool follows a
   precisely defined, versioned schema, kept in the repository and
   conformance-tested.
2. **`state` is always standard.** vllm-sr only adds optional fields in natural
   places, in inputs and outputs, so a standard client or server is never
   affected. A batch holds complete standard requests.
3. **One codec layer** converts between schemas on the client and server sides,
   mirroring the Router's LLM protocol codecs (`llmprotocol` and
   `protocolcodec`): solid, pluggable and maintainable.
4. **Schema docs** are generated from the specifications, with guides on top.

## Schemas

| ID | Definition |
| --- | --- |
| `systemone/0.2` | The standard: the public System One API, version 0.2. The published OpenAPI document carries no license, so it is not copied into the repository; `config/schemas/systemone/systemone-0.2.schema.yaml` is our own description of the interface (field names, types and structure, in our own words) with the source URL and the sha256 it was checked against. `make systemone-schemas-drift` compares it with the published document. |
| `vllm-sr/5.0` | `systemone/0.2` plus the vllm-sr extension set 5.0, `config/schemas/systemone/vllm-sr-5.0.extensions.yaml`. It replaces the runtime contract 4.0.0. |
| `vllm-sr/4.0` | Legacy, kept as a compatible superset: everything the runtime contract 4.0.0 accepts (implicit roles from state keys, `choices`/`levels`/`labels`, `images`/`videos`, preset-only questions, `states` batches on `/v1/systemone`, `{error}` bodies, the `data` listing). On a listener it also accepts every 5.0 form, so new clients can use 5.0 on the same endpoint while existing clients keep working. Its frozen contract is `config/schemas/systemone/vllm-sr-4.0.openapi.yaml`. |

A new version of the standard is a new ID (for example `systemone/0.3`); older
IDs stay while a deployment may name them. The standard and the extension set
version independently: a 5.x minor only adds fields, and a breaking change is a
new major.

### One API registry for listeners and upstreams

One registry (`apiregistry`) describes every API the Router speaks. Each entry
declares its name, its paths, its schema IDs and their defaults, its
settings, the kind of model it serves and the codec of each schema:

| API | Paths | Default schema | Other schemas |
| --- | --- | --- | --- |
| `chat_completions` | `/v1/chat/completions` | `openai/v1` | |
| `responses` | `/v1/responses` | `openai/v1` | |
| `messages` | `/v1/messages` | `anthropic/2023-06-01` | |
| `images` | `/v1/images/*` | `openai/v1` | |
| `systemone` | `/v1/systemone`, `/v1/systemone/batch`, `/v1/decisions` | `vllm-sr/5.0` on listeners, `systemone/0.2` for upstreams | `vllm-sr/4.0` |

APIs are named by the API, not the vendor: one vendor ships several APIs, and
many vendors implement the same one. `openai` and `anthropic`, the names
earlier configurations used, are aliases of `chat_completions` and `messages`,
resolved wherever an API is named; old configurations run unchanged. Giving one
API twice in one `apis` map (an alias and its name) is a validation error.

A new API is one registry entry with its paths, schemas and codec; a new
version of one is a schema ID and its codec. Neither changes the
configuration's shape or any caller.

### Listener APIs

A listener lists the APIs it serves in one `apis` map, keyed by the
registry's names. Every entry has the same shape: `schema` (only on an API
that has versions, today `systemone`), `models` (omitted: every model of the
API's kind) and the API's own settings (`max_deadline_ms` for `systemone`).

- Without `apis`, a listener serves `chat_completions`, `responses`,
  `messages` and `images` for every model; `systemone` stays off unless listed,
  because it publishes decision models directly.
- With `apis`, a listener serves exactly the listed APIs; any other API path
  answers 404 with a `detail` naming the listener. The generated Envoy listener
  routes only the served APIs' paths.
- `GET /v1/models` lists only the served APIs' models: OpenAI `data` for the
  LLM APIs, System One `models` for `systemone`.
- The listener-level `models` and `systemone` fields are removed;
  `vllm-sr config migrate` moves them into `apis`.

Only System One is versioned: Chat Completions, Responses and Messages evolve
additively, while System One has incompatible versions, and a path per version
would force old clients to change URLs.

### Where `schema` is configured

- **Server side:** `listeners[].apis.systemone.schema`, the contract a listener
  speaks to its clients; default `vllm-sr/5.0`.
- **Client side:** `schema` next to `api_format` on `providers.models[]` and
  next to `provider: systemone` on deployments; default `systemone/0.2`.
  `deviations` and `max_concurrency` are client-side settings of deployments.

### A `vllm-sr/4.0` listener

Measured on a production deployment, a large share of a public endpoint's
requests, mostly from external clients, use forms only 4.0 accepts, chiefly
inline `images` and `states` batches. A 4.0 listener therefore accepts both
forms on the same paths, with no header:

- The 4.0-only forms are exactly the ones a 5.0 server refuses, so reading them
  cannot change the meaning of a valid 5.0 request, with two overlaps, where a
  valid 5.0 request keeps its 4.0 reading on a 4.0 listener: an object state
  without `roles` keeps 4.0's implicit roles, and a state of asset
  placeholders in front of one item stays an array that labels its assets,
  where 5.0 reads the item with the media in front of it. The listener counts
  both.
- Answers keep the 4.0 shape; error bodies carry both `detail` (the standard)
  and `error` (4.0), so either kind of client can read them.
- A per-listener counter of the 4.0-only forms clients use, by form, tells the
  operator when switching to `vllm-sr/5.0` is safe.

## Design rules

1. **Standard first.** Whatever the standard can express is expressed the
   standard way; there are no second spellings.
2. **Every vllm-sr message is a complete standard message.** Every question
   carries standard `instructions` and `criteria`; extensions are additive
   hints. For example `preset` lets a vllm-sr server reach a trained head.
3. **Extension fields are flat, in their natural place,** with no namespace
   object and no request headers, as vLLM's extra parameters: SDK users pass
   them with `extra_body`.
4. **Unknown fields are refused with 400 naming them,** on every vllm-sr
   server, so nothing is silently ignored and no capability negotiation is
   needed.
5. **The state is standard content.** Its keys carry no hidden meaning.
   Role-aware families read an explicit `roles` map; without `roles`, the whole
   state is the user's content on every family. Each family keeps its own
   rendering of an object state into model input, like an LLM's chat template:
   rendering is a model detail behind the API, and the models' startup
   self-tests keep their inputs.
6. **Three layers of names; the part names stay.** API fields are defined by
   the two specifications. Keys inside a state are the client's own; when the
   Router builds a multi-part state itself it keeps `request`, `history`,
   `previous_turns`, `previous_answer`, `tool_results`, `candidate`, `answer`
   and `context`, which the models were trained and calibrated on. The Router
   configuration's `input:` names are the same set. No model input changes, so
   nothing is recalibrated.
7. **Follow the standard's limits and behaviours:** status codes (400, 422
   with a `detail` list, 429 and 529 with `Retry-After`), the long-context
   request header, and bodies up to 32 MiB.
8. **Batches are a separate extension endpoint,** `POST /v1/systemone/batch`,
   whose `requests` hold complete standard requests; `/v1/systemone` takes
   exactly one request. `/v1/decisions` stays the native endpoint, with
   per-question errors inside a 200; a request none of whose questions is
   valid is 400 `invalid_request`, listing each question's error, on the
   Router and the runtime alike.
9. **`vllm-sr/auto` routes by capability.** A request goes only to pool
   members that answer all of its question types and read its media (box and
   image, video, full input, …), by the member's card or listing, else its
   deployment's declared `capabilities`. If none can, it is refused with 400
   `unsupported_capability`, whose `detail` names what is missing and each
   member.

## The extension set (vllm-sr/5.0)

| Level | Fields |
| --- | --- |
| Request | `roles` (state key to `user`, `context` or `answer`); `options` (`deadline_ms`, `return_meta`, `max_tokens`) |
| Question | `preset`, `head`, `decision_adapter`, `over`, `overflow`, `require_full_input`, `threshold`, `at`, `rate` |
| Question types | `span` and `set`, with options in `criteria`; `track` on videos |
| Asset types | `video` |
| Answer | `abstain_probability`, `input_coverage`; on `/v1/decisions`, `error`, `reason` and `message` |
| Response | `spans`, `sets`, `thresholds`, `span_heads`, `meta`, `routing` |
| Error | the standard `detail`, plus flat `code`, `question`, `state`, `reason`, `limit` |
| Listing entry | `version`, `features`, `card` |
| Endpoints | `POST /v1/systemone/batch`, `POST /v1/decisions`, `GET /health` |

Removed, because the standard expresses them: `choices`, `levels` and `labels`
(`criteria`); `images` and `videos` (`assets`); top-level `states` (the batch
endpoint); `{error: {...}}` (`detail` and the flat fields);
`options.long_context` (the standard's header); implicit roles (`roles`).

The [extension reference](../system-one/reference/extension-reference.md) and
the [compatibility matrix](../system-one/reference/compatibility-matrix.md)
are generated from the extension specification.

## The Router as a client

```yaml
deployments:
  gateway-d3:
    provider: systemone
    endpoint: https://gateway.example.com/v1/systemone
    model: d3
    schema: systemone/0.2      # default; vllm-sr/5.0 for vllm-sr serve targets
    deviations: [state_text]   # only for a declared, tested non-conformance
    max_concurrency: 8         # default: split calls in flight
```

- **`systemone/0.2`** sends only standard question types and fields. Presets
  are expanded into explicit `instructions` and `criteria` from the Router's
  canonical preset texts; extension fields are stripped, by a list derived
  from the two specifications. What the standard cannot carry (span and set
  questions, `over`, `head`, `decision_adapter`, video, full-coverage
  questions) is refused at load with a one-line error naming the rule and the
  deployment. Batches and stage bundles are split under `max_concurrency` and
  the caller's deadline, answers and per-state errors reassembled; a call that
  misses the deadline leaves only its own questions unknown. No deadline field
  is sent. 400 `detail`, 422 `detail[]` and `Retry-After` are honoured, and the
  declared `capabilities` apply, since the standard has no model card.
- **`vllm-sr/5.0`** speaks the full extension set, the batch endpoint and the
  cards.
- **Deviations** are composable rewriters applied after the codec, each with
  its own test and documentation row. The first is `state_text`: an object or
  array state goes as its exact `json.dumps(state, ensure_ascii=False)` text,
  which Decision 3 renders identically and Vela parses back. An enterprise LLM
  gateway that refuses object states is declared `systemone/0.2` with
  `[state_text]`.
- Both Router paths use the codecs: `Decide` (the Router's own decision
  signals, judges, escalate checks and selectors) and `Forward` (the
  `vllm-sr/auto` pools), whose answers and errors are re-encoded for the
  client, including batch reassembly.

## Codec architecture

### The IR is the only pivot

`systemoneprotocol` defines the schema-neutral representation: requests,
batches, questions, answers, errors and listings, with JSON content kept as
order-preserving values. Roles have three modes: implicit (4.0's, from the
state's keys), none (5.0 without `roles`) and explicit. Every translation goes
through the IR: a listener's codec reads a client's request into it, and a
target's codec writes it in the target's schema.

### The internal wire form

The hops inside a deployment (the Router to the model runtime's `/v1/bundle`
and `/v1/decisions`, the Engine to its workers) carry the IR in the internal
form: the 4.0 request forms plus `roles`, where an absent `roles` is 4.0's
implicit roles and `roles: null` is none. The internal codec is lossless:
member order, the 4.0 spellings and number literals survive, and a request it
reads writes back unchanged. A round-trip test checks it over a golden corpus
(the model families' own self-test requests, extracted from their sources by
the schema generator, and the 4.0 request forms) and a fuzz corpus. So every
model input the Router builds is byte for byte what it was.

### Plans

A listener decides how each request goes on:

- **Pass through:** a request the internal form reads (every 4.0 request on a
  4.0 listener, and `/v1/decisions` on every listener) is decoded and checked,
  then forwarded with the client's own bytes.
- **Translate:** anything else is written from the IR in the internal form: a
  5.0 listener's requests, the batch endpoint, a request with the
  long-context header.

A client-side codec plans the wire calls of a batch: one batch call, or one
call per request under the sliding window. Splitting, preset expansion and
reassembly are decided by the codec's plan, never in callers. The diagnostics
of both kinds of plan feed metrics and logs:
`vsr_systemone_listener_diagnostics_total` counts each listener's legacy forms
and its pass-throughs and translations.

```go
type SchemaCodec interface {
    ID() SchemaID                    // "systemone/0.2", "vllm-sr/5.0"
    Spec() SpecRef                   // the specification file and its sha256
    Capabilities() CapabilitySet     // batch, roles, span, set, video, presets, ...
    Stateless() bool
    Plan(Batch, Target, Policy) (Plan, Diagnostics, error)
    DecodeReply(Plan, WireCall, WireReply) ([]ItemResult, error)
    ListingPath() string
    DecodeListing(status int, body []byte) (Listing, error)
}

type ListenerCodec interface {
    Read(path string, body []byte, header http.Header) (ClientRequest, error) // the IR and its plan
    AnswerSuccess(*ClientRequest, []byte) ([]byte, error)
    AnswerError(*ClientRequest, path string, status int, body []byte) (int, []byte)
    AnswerListing([]byte) ([]byte, error)
}

type Deviation interface {
    ID() string                      // "state_text"
    AppliesTo(SchemaID) bool
    RewriteRequest(*WireCall) (Diagnostics, error)
    RewriteReply(*WireReply) error
}
```

- **The registry** is immutable and validates at startup: the ID pattern,
  statelessness, no duplicates, each specification's sha256, and that each
  deviation applies to a registered schema. Configuration resolves `schema`
  and `deviations` through it; an unknown ID is a load error listing the
  supported ones.
- **One source of truth.** The extension specification defines each field
  once: its level, type, the version it appeared in and the families that
  honour it. The generator writes the Go field tables, the model runtime's
  field tables, the strip list for standard targets, the internal round-trip
  corpus, and the generated reference pages.
- **CI checks:** generated artifacts are fresh, a field defined by both
  specifications with different meanings fails the build, and the conformance
  tests pass.
- **Tests:** golden tests per codec and per deviation, spec-driven request and
  response checks, the IR round trip, cross-family equivalence (an object state
  and its `state_text`), and fuzzing the canonical JSON encoder against Python.

## vllm-sr serve as a standard server

The model runtime and the Router and Engine frontends serve the standard:
standard requests exactly as the standard defines them, plus the extension
fields, any other field refused with 400. `vllm-srun serve --systemone-schema`
sets a runtime's schema: `vllm-sr/5.0` by default, or `vllm-sr/4.0` for a
runtime that serves 4.0 clients directly. The runtime contract is 5.0.0.

## Rollout

The work shipped in one pull request: the specifications, the IR, the
registries and codecs, listener `apis`, upstream `schema`, `systemone/0.2`
with `state_text`, `vllm-sr/5.0`, `vllm-sr/4.0`, the runtime's 5.0 surface,
the migration and the documentation.

A hosted Decision API built on the Router or the Engine upgrades in two steps.
First, upgrade every component, backends included, with `vllm-sr/5.0` between
the Router or Engine and the runtimes, and keep the public listeners at
`apis.systemone.schema: vllm-sr/4.0`. Later, move in-house callers to 5.0
forms, and switch a listener to `vllm-sr/5.0` once its legacy-form counter
shows that external 4.0 traffic is gone.

## Risks and follow-ups

- **The standard is young (0.2).** Mitigated by pinning, version IDs and the
  drift and collision checks.
- **Zero-shot answers on standard targets.** Expanded presets get zero-shot
  answers instead of trained heads, so thresholds tuned on trained heads need
  recalibration on a standard target.
- **Unintended input drift.** Explicit roles reproduce 4.0's implicit roles
  exactly for the Router's part names; tests cover the families' readers, and
  the internal form keeps every Router-built input unchanged.
- **5.0 breaks external clients** that rely on implicit roles, second
  spellings or `states` on `/v1/systemone`; a 4.0 listener keeps them working.
- **Batch splitting** multiplies upstream calls; `max_concurrency` and the
  deadline bound them.
- **Follow-up:** the runtime still requires two options for a choice or a
  score, and instructions or described criteria for a noul, where the standard
  allows one option and a bare noul.
- **Follow-up:** `vllm-sr/auto` routes `box` questions by capability; `set`,
  `span` and `track` answers are family-specific (`sets`, `spans`), so auto
  validates and routes them in a later change.
