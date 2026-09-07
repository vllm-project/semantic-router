# Context Compression

## Overview

`context_compression` is a route-local request plugin that reduces large
tool/function outputs before the selected provider receives the request. It is
separate from router signal compression: routing evaluates the original
request, then this plugin performs the upstream body mutation.

Compression is local, extractive, query-aware, and fail-open. It uses
bounded BM25-style ranking, keeps leading and trailing context, and never
changes system, user, or assistant text.

## Key Advantages

- Reduces provider input tokens on tool-heavy routes.
- Keeps routing and safety signals on the original request.
- Applies per decision instead of changing every request.
- Fails open when the request cannot be parsed or rewritten.
- Preserves valid JSON structure and non-text multimodal blocks.
- Supports OpenAI tool/function messages and Anthropic `tool_result` blocks.

## What Problem Does It Solve?

Agent and retrieval workloads often carry tool outputs that are much larger
than the user question. Forwarding every low-relevance line increases latency
and cost without improving the answer.

## When to Use

Use it on decisions dominated by large text tool outputs. Do not enable it on
routes that require byte-identical tool payloads.

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: context_compression
    configuration:
      enabled: true
      mode: auto
      budget:
        trigger_tokens: auto
        target_tokens: auto
        reserve_output_tokens: auto
      targets:
        tool_outputs:
          mode: extractive
          min_tokens: 2000
          target_tokens: 1000
        history:
          mode: preserve
        rag:
          mode: preserve
        memory:
          mode: preserve
      scoring:
        method: bm25
      recovery:
        enabled: false
        ttl_seconds: 900
        max_bytes_per_request: 10485760
        max_total_bytes: 268435456
        max_retrievals: 8
      request_controls:
        enabled: false
        header: x-vsr-compression-control
        allowed: [bypass, target]
        max_target_tokens: 16000
      failure_mode: fail_open
```

`targets.tool_outputs.target_tokens` must be lower than `min_tokens`.
`budget` applies to the complete selected-model request; tool-output targets
retain their own per-item threshold and ceiling. `auto` derives the request
budget from the selected model context window and requested output reserve.

RAG and memory evidence are protected by typed provenance by default. Set the
corresponding target mode to `extractive` only when the route explicitly
accepts evidence compression.

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

If a payload cannot be reduced safely within the configured budget, it is sent
unchanged under `fail_open`, or the route fails under explicit `fail_closed`.

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

`scoring.method` supports `bm25`, `embedding`, and `hybrid`. Embedding work is
batched and held in a bounded memo cache; hybrid scoring falls back to BM25 when
the configured embedding runtime is unavailable.

## Management and preview

- `GET /api/v1/context-compression/capabilities`
- `GET /api/v1/context-compression/health`
- `GET /api/v1/context-compression/stats`
- `POST /api/v1/context-compression/preview`
- `POST /api/v1/context-compression/recovery/invalidate`

Preview returns only plans, target indexes, token counts, scores, warnings, and
skip reasons. It never returns source or omitted content and requires
`compression.preview`. Scoped recovery invalidation requires
`compression.manage`; it accepts trusted recipe, decision, user, and request
coordinates and never returns the derived scope or recovery keys.

## Runtime Order

The response cache checks an immutable canonical request first. On a miss, RAG
and memory may enrich a separate provider-bound working body, then
`context_compression` runs before provider request translation and provider
prompt-cache marker injection. The final Envoy body mutation always uses that
working body, including auto, specified-model, Response API, Anthropic, streamed
request, and Looper paths.

Compression diagnostics are recorded in metrics and Router Replay: selected
model, strategy, request/item budget, token-counter source, trigger reason,
tokens before/after/saved, content format, compressed message count, omitted
chunk count, recovery count, and fail-open or skip reason. Raw omitted content
and recovery keys are not recorded.

See a complete example:
[`config/fragments/plugin/context-compression/tool-output.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/context-compression/tool-output.yaml).
Compression changes provider-bound context and can remove details needed for a
correct answer. Keep fail-open behavior until the route has task-specific
quality tests; enable recoverable mode only with an authenticated shared store
and trusted user identity.
