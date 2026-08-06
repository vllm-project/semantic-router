# Context Compression

## Overview

`context_compression` is a route-local request plugin that reduces large
tool/function outputs before the selected provider receives the request. It is
separate from router signal compression: routing evaluates the original
request, then this plugin performs the upstream body mutation.

The first implementation is local, extractive, query-aware, and fail-open. It
keeps leading, trailing, and query-relevant chunks and never changes system,
user, or assistant messages.

## Key Advantages

- Reduces provider input tokens on tool-heavy routes.
- Keeps routing and safety signals on the original request.
- Applies per decision instead of changing every request.
- Fails open when the request cannot be parsed or rewritten.

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
```

`target_tokens` must be lower than `min_tokens`. When `bypass_header` is set,
the plugin honors that header only when its value is `true`.

## Runtime Order

The response cache checks the original request first. On a miss, RAG and memory
may enrich the request, then `context_compression` runs before provider request
translation. Compression diagnostics are recorded in Router Replay.
