# Masking

## Overview

`masking` is a route-local plugin that detects personal data (emails, phone numbers, names, and other configured entity types) in the provider-bound request and replaces each occurrence with a deterministic placeholder, such as `[EMAIL_ADDRESS_0]`, immediately before the request is encoded and sent to the model provider. The same value gets the same placeholder everywhere in one request; two different values get different placeholders.

## What Problem Does It Solve?

Some routes send traffic to a model provider that should never see raw personal data — a third-party API, a lower-trust tier, or a route subject to a data-handling policy. The router can already detect PII and act on it by refusing a request or routing it to a safer model, but neither of those sends the request *with the PII removed*. `masking` rewrites the content in place so the provider only ever sees placeholders, without requiring every caller to redact input themselves.

Because every wire format (Chat Completions, Responses, Anthropic Messages) is decoded into one neutral request before plugins run, this happens identically regardless of which API a client used to reach the router.

## When to Use

- a route dispatches to a provider that must not receive personal data
- the same value should map to the same placeholder throughout one request, and different values should map to different placeholders
- a missing or unavailable PII classifier should block the request rather than silently send it unmasked

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: masking
    configuration:
      enabled: true
      threshold: 0.85
      entity_types:
        - EMAIL_ADDRESS
        - PHONE_NUMBER
      placeholders:
        EMAIL_ADDRESS: "[EMAIL_{index}]"
        PHONE_NUMBER: "[PHONE_{index}]"
```

| Field | Meaning |
|---|---|
| `enabled` | Turns masking on for this decision. |
| `threshold` | Minimum classifier confidence for a span to be masked. `0` falls back to the router's configured PII model threshold. |
| `entity_types` | An **include list**: only these types are masked. Empty means every detected type is masked. This is the opposite of `PIIRule`'s `pii_types_allowed`, which is an allow-list of types let through. |
| `placeholders` | Maps an entity type to a template. The template must contain `{index}`; a template without it would collapse distinct values onto one placeholder, so it is rejected at config load. Types absent from the map get the default `[TYPE_n]` shape. |

## Fail-closed, with no override

If the PII classifier is unavailable, or scanning fails for any reason, **the request is refused with a 503** rather than dispatched unmasked. This is unconditional: there is no `fail_open` configuration option, unlike `context_compression`'s `failure_mode`. A privacy control that can be configured to fail open is a footgun that gets flipped on during an incident and never flipped back. Operators who need availability over privacy should not enable this plugin.

A truncated classifier response — the provider only scanned part of the text — is treated the same way: the unscanned remainder cannot be vouched for, so the request is refused rather than partially masked.

## Router Replay does not capture request content on a masking route

[Router Replay](/docs/tutorials/plugin/router-replay) snapshots the neutral request before dispatch, which is *before* masking runs. Enabling `masking` on a decision therefore disables request-body and prompt capture in Router Replay for that decision: `RequestBody` and `Prompt` are left empty. Routing metadata, signals, and decisions are still recorded — only the content is not. Tool definitions are still recorded, since they are operator-authored, not user content.

Masking the stored Replay copy instead of suppressing it is a reasonable idea, but it needs a Replay update path that does not currently exist. It is not implemented here.

## Masking is not reversible

This plugin only ever removes information going forward to the provider. It does not:

- remember the value-to-placeholder mapping beyond the lifetime of one request
- restore a placeholder back to its original value anywhere, including in the model's response
- give an operator, an API, or a log line any way to recover a masked value

If a response happens to echo a placeholder like `[EMAIL_ADDRESS_0]` back to the client, that placeholder is not decoded — the client sees exactly that string. Restoring placeholders in responses is an explicit non-goal, not a missing feature.

## What is deliberately not masked

The masking walk touches message content, system/developer instructions, and structured tool-call arguments — and nothing else. Specifically left alone:

- **Reasoning blocks.** Anthropic-style thinking blocks carry a provider signature over their text; rewriting the text would invalidate it and the provider would reject the request.
- **Tool definitions** (`Tools[].Description`, `InputSchema`). These are operator-authored, not user input — masking them would corrupt the model's understanding of its own tools for no privacy benefit.
- **Media payloads and references** (image/audio/video/file `Data`, `URL`, `FileID`, `Filename`).
- **Identifiers** used for correlation: `ToolCall.ID`, `ToolResult.CallID`, `Request.Model`, `PreviousResponseID`, `ConversationID`.
- **`Request.Metadata`** (client-supplied key/value pairs) — out of the issue's stated scope.
- **JSON object keys** inside tool-call arguments. Only string *values* are scanned; a key that happens to look like PII is never touched.

## Citations

If a text block carries citations and part of it gets masked, each citation is adjusted: one whose range overlaps a masked span is dropped, because its offsets no longer point at anything meaningful once the text has changed length. A citation that does not overlap any masked span is kept, with its offset corrected for the length change.

## Observability

Detected values never reach a log, a metric label, or a trace attribute. What is recorded:

- the entity **type** (e.g. `EMAIL_ADDRESS`), never the value or the placeholder
- a count of masked spans, by entity type and decision
- byte/offset positions and confidence scores, where logged at all

See a complete example:
[`config/fragments/plugin/masking/basic.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/masking/basic.yaml).
