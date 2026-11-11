# Masking

## Overview

`masking` is a route-local plugin that detects personal data (emails, phone numbers, names, and other configured entity types) in the provider-bound request and replaces each occurrence with a deterministic placeholder, such as `[EMAIL_ADDRESS_0]`, immediately before the request is encoded and sent to the model provider. The same value gets the same placeholder everywhere in one request; two different values get different placeholders.

## What Problem Does It Solve?

Some routes send traffic to a model provider that should never see raw personal data — a third-party API, a lower-trust tier, or a route subject to a data-handling policy. The router can already detect PII and act on it by refusing a request or routing it to a safer model, but neither of those sends the request *with the PII removed*. `masking` rewrites the content in place so the provider only ever sees placeholders, without requiring every caller to redact input themselves.

Because every wire format (Chat Completions, Responses, Anthropic Messages) is decoded into one neutral request before plugins run, this happens identically regardless of which API a client used to reach the router.

## When to Use

- a route dispatches to a provider that must not receive personal data
- the same value should map to the same placeholder throughout one request, and different values should map to different placeholders
- a missing or unavailable PII detector should block the request rather than silently send it unmasked

## Configuration

Masking masks what a [pii rule](../signal/application/pii) detects. Declare the
rule under `recipes[].signals.pii` and name it in the plugin under
`recipes[].decisions[].plugins`:

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: contact_details
          predicate: {gte: 0.85}
          description: Contact details masked before the request leaves the Router.
    decisions:
      - name: support
        # ...rules, modelRefs...
        plugins:
          - type: masking
            configuration:
              enabled: true
              signal: contact_details
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
| `signal` | The pii rule whose detections are masked: its deployment, its `predicate`, and the labels its `ignore_labels` lets through, which are not masked. No decision has to read the rule. |
| `entity_types` | An **include list** of PII labels (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `LOCATION`, `IP_ADDRESS`, `DATE_TIME`, `US_DRIVER_LICENSE`, `US_PASSPORT`, `IBAN_CODE`): only these types are masked. Empty means every type the rule detects is masked. Any other label is rejected at config load. |
| `placeholders` | Maps a PII label to a template. The template must contain `{index}`; a template without it would collapse distinct values onto one placeholder, so it is rejected at config load. Types absent from the map get the default `[TYPE_n]` shape. |

The rule reads the request as it is sent, after every plugin that changes it
has run, so personal data a [RAG](./context) document or a system prompt brought
in is masked too. The rule's text is scanned once per request: a decision that
also reads the rule at routing reuses the scan of the same text.

The rule's detector must answer with positions. A rule that resolves to a
deployment that cannot answer span questions, such as a decision model without
span support, fails at load with `span_required`.

A configuration that still sets `masking.threshold`, or names no `signal`,
fails at load with a one-line hint; `vllm-sr config migrate` adds a pii rule
with the plugin's threshold and names it.

## Fail-closed, with no override

If the rule's detector is unavailable, or scanning fails for any reason, **the request is refused with a 503** rather than dispatched unmasked. This is unconditional: there is no `fail_open` configuration option, unlike `context_compression`'s `failure_mode`. A privacy control that can be configured to fail open is a footgun that gets flipped on during an incident and never flipped back. Operators who need availability over privacy should not enable this plugin.

A truncated answer, where the deployment scanned only part of the text, is treated the same way: the unscanned remainder cannot be vouched for, so the request is refused rather than partially masked.

## Router Replay does not capture request content on a masking route

[Router Replay](/docs/tutorials/plugin/router-replay) snapshots the neutral request before dispatch, which is *before* masking runs. Enabling `masking` on a decision therefore drops every request-derived field Replay would otherwise store for that decision: `RequestBody`, `Prompt`, and the request tool trace (user text, tool-call arguments, tool results) are all left empty. That applies on the internal Looper path and when masking itself fails, both of which also record before the final mask. Routing metadata, signals, and decisions are still recorded — only request content is not. Tool definitions are still recorded, since they are operator-authored, not user content. The response tool trace is unaffected: the model only ever saw masked content.

Masking the stored Replay copy instead of suppressing it is a reasonable idea, but it needs a Replay update path that does not currently exist. It is not implemented here.

## Masking is not reversible

This plugin only ever removes information going forward to the provider. It does not:

- remember the value-to-placeholder mapping beyond the lifetime of one request
- restore a placeholder back to its original value anywhere, including in the model's response
- give an operator, an API, or a log line any way to recover a masked value

If a response happens to echo a placeholder like `[EMAIL_ADDRESS_0]` back to the client, that placeholder is not decoded — the client sees exactly that string. Restoring placeholders in responses is an explicit non-goal, not a missing feature.

An agent that keeps working with placeholders sends them back in its later requests, for example in a file it writes with `Hi [PERSON_0],`. A detected span that lies inside text shaped like a placeholder this plugin produces, the default `[TYPE_n]` shape or a configured template that has a letter besides `{index}`, with digits for `{index}`, is not masked again, even when the detector reads part of it as personal data. A span that reaches past it, such as an email address whose local part is a placeholder, is still masked, and a template such as `{index}` or `#{index}` is not recognized, since its digits could be part of a real phone number. Without that, the agent would see its own earlier output turn into `[[PERSON_1]]` on the next turn and its edits would stop matching the file it wrote.

The same holds for the `[vsr:<tag>]` marks, such as `[vsr:email]`, that the vSR Claude Code plugin leaves where its redact app hid a value from a tool's output. With the plugin and the gateway running together, a tool result reaches the Router already redacted, and the marks pass through as they are.

## What is deliberately not masked

The masking walk touches message content, system/developer instructions, tool-call arguments, and tool results — and nothing else.

Tool payloads are masked according to their kind. A function call's JSON arguments and a structured tool result are decoded first and masked at their string leaves, so object keys stay intact, the payload stays valid JSON, and an escaped value such as `alice@example.com` is still caught. A custom tool's free-form input and a plain-text tool result are masked as text.

A tool result is treated as structured when it opens with `{` or `[`. Such a payload must decode faithfully — exactly one JSON value, nothing after it, and no object repeating a member name — or masking fails closed (D4). Both of these are refused rather than forwarded:

- `{"note":"safe"} alice@example.com` — more than one value. Decoding reads only the first, leaving the rest unscanned.
- `{"value":"alice@example.com","value":"safe"}` — a repeated member name. Decoding keeps only the last one.

Masking as text is not used as a fallback for these, because the text path scans the undecoded bytes: it would miss an escaped value such as `{"value":"alice@example.com"} note`, which the provider still reads as `alice@example.com`. A payload that does not open as JSON is ordinary text and is masked as text, where the whole string is scanned.

Specifically left alone:

- **Reasoning blocks.** Anthropic-style thinking blocks carry a provider signature over their text; rewriting the text would invalidate it and the provider would reject the request.
- **Tool definitions** (`Tools[].Description`, `InputSchema`). These are operator-authored, not user input — masking them would corrupt the model's understanding of its own tools for no privacy benefit.
- **Media payloads and references** (image/audio/video/file `Data`, `URL`, `FileID`, `Filename`).
- **Identifiers** used for correlation: `ToolCall.ID`, `ToolResult.CallID`, `Request.Model`, `PreviousResponseID`, `ConversationID`.
- **`Request.Metadata`** (client-supplied key/value pairs) — out of the issue's stated scope.
- **JSON object keys** inside tool-call arguments and structured tool results. Only string *values* are scanned; a key that happens to look like PII is never touched.

## Citations

If a text block carries citations and part of it gets masked, each citation is adjusted: one whose range overlaps a masked span is dropped, because its offsets no longer point at anything meaningful once the text has changed length. A citation that does not overlap any masked span is kept, with its offset corrected for the length change.

## Observability

Detected values never reach a log, a metric label, or a trace attribute. What is recorded:

- the entity **type** (e.g. `EMAIL_ADDRESS`), never the value or the placeholder
- a count of masked spans, by entity type and decision
- byte/offset positions and confidence scores, where logged at all

See a complete example:
[`config/fragments/plugin/masking/basic.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/masking/basic.yaml).
