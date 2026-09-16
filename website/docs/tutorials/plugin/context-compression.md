# Context Compression

## Overview

Use `context_compression` on a decision when old conversation text or large tool
outputs cost more tokens than the answer needs. It changes the conversation sent
to the answer model; routing signals still evaluate the original request.
Compression is lossy, so test representative questions before enabling it.

## Reduce large tool outputs

Add this fragment to the decision's `plugins`. It reduces tool outputs above
2,000 estimated tokens toward 1,000 tokens, keeping text relevant to the task:

```yaml
plugins:
  - type: context_compression
    configuration:
      enabled: true
      targets:
        tool_outputs:
          mode: extractive
          min_tokens: 2000
          target_tokens: 1000
```

`target_tokens` must be lower than `min_tokens`. Omitted settings use the
plugin's defaults: automatic request budgeting, BM25 relevance scoring, and
`fail_open` if an ordinary tool-output rewrite cannot be completed safely.
Use `mode: preserve` for content that must remain unchanged. History, RAG,
memory, and current-user text are preserved unless their target is enabled.

## Requests beyond the model context window

Encoder input truncation only bounds the router's classification view. It does
not shorten the conversation sent to the generation model. To accept oversized
plain-text user messages, opt into the existing plugin on the affected decision:

```yaml
plugins:
  - type: context_compression
    configuration:
      enabled: true
      targets:
        current_user:
          mode: truncate
        history:
          mode: extractive
          min_tokens: 2000
          target_tokens: 1000
        tool_outputs:
          mode: preserve
```

`current_user.mode` defaults to `preserve`. With `truncate`, older history can
be reduced first; the latest plain-text user message then keeps its beginning
and end, separated by `[... context omitted by route compression ...]`.
**The middle is omitted, not summarized.** Do not use this policy when every
part of the user's message must reach the model.

Configure accurate context and maximum-output limits for the candidate models.
The budget reserves the requested output as well as chat and tool formatting.
It uses a conservative UTF-8 byte estimate, so it can shorten a request even
when that model's tokenizer counts fewer than 32K tokens. It is not an exact
tokenizer count or a guarantee for arbitrary media or custom chat templates.
`budget.reserve_output_tokens` can increase the reserve, but cannot reduce the
actual requested output allowance.

System/developer instructions, protected authorization and safety text, tool
calls, schemas, media, citations, and JSON user payloads are not eligible for
current-user truncation. Tool call/result IDs stay paired. Tool-result text
changes only if its own target policy permits it; the example preserves it.

The Router checks the prepared request before choosing a model and again before
sending it to the backend. If protected content or the remaining conversation
cannot fit, it returns HTTP 400 with `context_length_exceeded`. `fail_open`
does not bypass that budget check, and no partially rewritten request is sent.

## Content handling

- Plain text is split into bounded chunks and ranked against the originating
  tool-call intent, falling back to recent user text.
- JSON object and array strings are compressed only through string leaves.
  Keys, arrays, objects, numbers, booleans, and null values keep their types.
- OpenAI array content compresses text blocks and preserves image blocks.
- Anthropic `tool_result` string and array content is supported; `tool_use_id`,
  `is_error`, images, and cache-control metadata are preserved.
- Large single-line, minified, CJK, emoji, and whitespace-free payloads use a
  conservative byte-aware token estimate.

For an ordinary rewrite failure, `fail_open` keeps the payload unchanged and
`fail_closed` fails the route. Neither option bypasses model budget admission.

History compression protects every system message, the live user turn, the
latest assistant turn, and complete tool exchanges. Optional `recoverable`
targets store original content in a shared Redis/Valkey store, inject the
reserved `vsr_context_retrieve` tool, and use the configured Looper endpoint for
a non-streaming follow-up. Recovery is request- and trusted-user-scoped, bounded
by TTL, bytes, and retrieval count. Streaming requests preserve recoverable
targets rather than exposing the internal tool.

## Request controls

Request controls are ignored unless the matched route enables them.

- `bypass` skips compression.
- `target=N` overrides the tool-output target and is clamped by
  `max_target_tokens`.

The default header is `x-vsr-compression-control`. Caller-provided namespaces,
recovery keys, and unbounded budgets are never accepted.

`scoring.method` supports `bm25`, `embedding`, and `hybrid`. Hybrid scoring uses
BM25 if its embedding service is unavailable.

## Management and preview

- `GET /api/v1/plugins/context_compression/capabilities`
- `GET /api/v1/plugins/context_compression/health`
- `GET /api/v1/observability/plugins/context_compression/stats`
- `POST /api/v1/plugins/context_compression/preview`
- `POST /api/v1/storage/context-recovery/invalidate`

Preview returns only plans, target indexes, token counts, scores, warnings, and
skip reasons. It never returns source or omitted content and requires
`compression.preview`. Scoped recovery invalidation requires
`compression.manage`; it accepts trusted recipe, decision, user, and request
coordinates and never returns the derived scope or recovery keys.

## Verify the result

Use the plugin preview to inspect the plan, then send a real request through
the affected decision. Check that the answer still uses the facts your task
requires. Preview alone does not test backend generation.

Metrics and Router Replay report whether compression ran, its strategy and
counting source, before/after counts, and omissions. These counts describe
compression estimates; they are not measured billing-token savings. Raw omitted
content and recovery keys are not included in those diagnostics.

For all available options, see the
[configuration reference](../../api/configuration-schema.mdx) and the
[complete tool-output example](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/context-compression/tool-output.yaml).
Enable recoverable mode only with an authenticated shared store and trusted
user identity.
