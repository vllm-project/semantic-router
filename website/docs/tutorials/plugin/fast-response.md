# Fast Response

## Overview

`fast_response` is a route-local plugin that returns a deterministic fallback
message immediately, or, with `block: true`, refuses the request with it.

## Key Advantages

- Short-circuits expensive routes when a lightweight fallback is enough.
- Keeps overload behavior local to the route that needs it.
- Makes the fallback message explicit in config.

## What Problem Does It Solve?

Some routes should degrade gracefully instead of waiting for the full model path. `fast_response` gives those routes an immediate response path without changing global behavior.

## When to Use

- the route needs a cheap fallback under overload or maintenance conditions
- a deterministic response is acceptable for this traffic class
- fallback behavior should stay local to one route
- the requests a decision matches must be refused

## Configuration

Add the plugin under `recipes[].decisions[].plugins`:

```yaml
plugins:
  - type: fast_response
    configuration:
      message: The primary model is unavailable. Try again shortly.
```

No model is called and no response content is generated. Keep the message free
of request data and do not use this plugin as an authentication or rate-limit
control. The plugin does not measure overload; the decision must match the
traffic that should receive the fallback. `enabled: false` turns off a
`fast_response` the decision inherits from `global.plugins` or its recipe's
plugins. See a complete example:
[`config/fragments/plugin/fast-response/busy.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/fast-response/busy.yaml).

## Block a Request

By default the message reaches the client as the model's answer, with HTTP
`200`, and Router Replay records the request as completed. `block: true`
refuses the request instead:

```yaml
plugins:
  - type: fast_response
    configuration:
      message: Subagents may not send personal data.
      block: true
```

- A request that does not stream gets HTTP `400` with the message as an
  `invalid_request_error` in its own protocol, with code `request_blocked` on
  Chat and Responses. A `400` is what a provider's content filter answers:
  Claude Code 2.1.325 reads a `401` or `403` as a failed login and retries it.
- A streaming request gets the message as its answer: a `200` stream that
  ends the way a finished answer does (`end_turn` on Messages, `stop` on Chat,
  `response.completed` on Responses), with zero usage, since no model ran.
  Its ids have the shapes a model's answer has: `chatcmpl-` on Chat, `msg_`
  on Messages, and `resp_` with a `msg_` message on Responses, whose clients
  send the message back with its id to a backend that refuses another shape.
  Agent harnesses stream, and Claude Code 2.1.325 does not take a `400` as the
  end of a turn: it sends the refused request six times in all, turning
  features off between tries, then puts the next prompt into the refused turn,
  which the block refuses again until `/clear`. Given the answer, Claude Code,
  Codex 0.166.0 and Pi 1.5.2 send the blocked prompt once, show the message as
  the reply, and send the next prompt as a turn of its own.
- `x-vsr-response-path` is `blocked` either way, standalone and behind Envoy.
  It is the value a response guard's block gives an answer, and a streaming
  client that must tell a block from an answer reads it.
- The Replay record ends `failed` with `terminal_reason: request_rejected`
  and the status the client got: `400`, or `200` for a stream.

No model is called either way. A decision's conditions say what to block, for
example an [`agent`](../signal/heuristic/agent#block-a-request) rule with a
`pii` rule, or a `safety` rule.

### Later Requests Carry the Blocked Prompt

A streamed block ends the turn as an answer. The harness keeps the blocked
prompt, with the message as its answer, in its history, and sends both again
with every later prompt of the conversation. A rule that reads only the
newest turn does not refuse those later requests, so the blocked prompt
reaches the model as history once the next prompt is routed elsewhere. By
default a `keyword` rule reads the newest user message, a `pii` rule the
current user turn, and a `jailbreak` rule the user and tool messages after
the last assistant message.

- Content that must never reach a backend, such as personal data or an
  internal identifier, needs the [`masking`](./masking) plugin on every
  decision that sends requests out. It masks every message of the request,
  the history included, before the request leaves the Router, with a `pii`
  rule or a `keyword` rule as its detector.
- A conversation that must stay refused once it was blocked needs a rule that
  also reads earlier turns: a `keyword` rule with `include_history: true`,
  which reads the 20 most recent earlier user messages and no more, or a `pii`
  or `jailbreak` rule with `input.history: true`. Each later request then gets
  the message too. A `pii` or `jailbreak` rule reads a long history in
  windows and, with `on_unscanned: best_effort`, the default, decides on the
  windows read before its deadline, so a history read only in part can pass.
  Set `on_unscanned: match` or a `min_coverage` where that must not happen (see
  [When only part of the input was read](../signal/application/jailbreak#when-only-part-of-the-input-was-read)).

The answer also ends the blocked turn for a `jailbreak` rule. A `400` left no
answer between the blocked prompt and the next one: Codex 0.166.0 and Pi
1.5.2 sent them as two user messages in a row, which the rule read as one
current turn and refused again, while the answer leaves the next prompt alone
in its turn. A client that does not stream still gets the `400`, and its next
request is read the same way when it sends the two prompts in a row.
