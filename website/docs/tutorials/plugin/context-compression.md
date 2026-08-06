# Context Compression

## Overview

`context_compression` is a route-local request plugin that reduces large
tool/function outputs before the selected provider receives the request. It is
separate from router signal compression: routing evaluates the original
request, then this plugin performs the upstream body mutation.

The implementation is local, extractive, query-aware, and fail-open. It uses
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

```yaml
routing:
  decisions:
    - name: tool-heavy-route
      plugins:
        - type: context_compression
          configuration:
            enabled: true
            min_tokens: 2000
            target_tokens: 1000
            bypass_header: x-vsr-compression-bypass
            compress_rag: false
```

`target_tokens` must be lower than `min_tokens`. When `bypass_header` is set,
the plugin honors that header only when its value is `true`.

RAG tool messages are protected by default because retrieved evidence may be
required verbatim. Set `compress_rag: true` only when the route explicitly
accepts extractive compression of RAG results.

## Content handling

- Plain text is split into bounded chunks and ranked against recent user text.
- JSON object and array strings are compressed only through string leaves.
  Keys, arrays, objects, numbers, booleans, and null values keep their types.
- OpenAI array content compresses text blocks and preserves image blocks.
- Anthropic `tool_result` string and array content is supported; `tool_use_id`,
  `is_error`, images, and cache-control metadata are preserved.
- Large single-line, minified, CJK, emoji, and whitespace-free payloads use a
  conservative byte-aware token estimate.

If a payload cannot be reduced safely within the configured budget, it is sent
unchanged. The plugin does not inject a recovery tool or alter the client's tool
protocol.

## Runtime Order

The response cache checks an immutable canonical request first. On a miss, RAG
and memory may enrich a separate provider-bound working body, then
`context_compression` runs before provider request translation and provider
prompt-cache marker injection. The final Envoy body mutation always uses that
working body, including auto, specified-model, Response API, Anthropic, streamed
request, and Looper paths.

Compression diagnostics are recorded in metrics and Router Replay: estimated
tokens before/after, content format, compressed message count, omitted chunk
count, and fail-open or skip reason. Raw omitted content is not recorded.
