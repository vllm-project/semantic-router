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

- The client gets HTTP `400` with the message as an `invalid_request_error`
  in its own protocol, with code `request_blocked` on Chat and Responses. A
  streaming request gets the same `400`, not a stream. A `400` is what a
  provider's content filter answers, and clients show its message without
  retrying: Claude Code 2.1.325 reads a `401` or `403` as a failed login and
  retries it.
- `x-vsr-response-path` is `blocked`, the value a response guard's block
  gives an answer.
- The Replay record ends `failed` with `terminal_reason: request_rejected`
  and status `400`.

No model is called either way. A decision's conditions say what to block, for
example an [`agent`](../signal/heuristic/agent#block-a-request) rule with a
`pii` rule, or a `safety` rule.
