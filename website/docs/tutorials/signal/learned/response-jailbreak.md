# Response Jailbreak Signal

## Overview

`response_jailbreak` scores the model's own output for jailbreak content. Define
rules under `routing.signals.response_jailbreak`.

It is a response-stage signal. Unlike `jailbreak`, which reads the request, this
one cannot exist until the model has answered, so a decision that reads it is
not evaluated while the request is still being routed.

## Key Advantages

- Detection is evidence, not plugin-internal state: it runs from the declared
  rules whether or not an enforcement plugin is enabled.
- The `response_jailbreak` plugin consumes that evidence instead of classifying,
  so detection configuration and enforcement configuration stay separate.
- An unresolved detector is reported through `SignalErrors`, the way every other
  signal reports one, rather than looking like a clean response.

## What Problem Does It Solve?

A model can be steered into producing unsafe content even when the prompt looked
benign, so a request-stage check alone does not cover it. Keeping that detection
inside a plugin also puts it outside the routing graph, where nothing else can
read it and no recipe can combine it with anything.

`response_jailbreak` makes the observation a signal, and leaves blocking or
warning to the plugin that consumes it.

## When to Use

Use `response_jailbreak` when:

- model output must be checked before it reaches the caller
- an unsafe answer should be blocked, replaced, or sent to a safer model
- detection and enforcement policy need to be configured separately

## Configuration

```yaml
routing:
  signals:
    response_jailbreak:
      - name: unsafe_completion
        threshold: 0.7
        description: Detect jailbreak content in the model's own output.
```

| Field | Description |
| --- | --- |
| `name` | Rule name. Identifies the observation in `SignalConfidences` and `SignalErrors` as `response_jailbreak:<name>`. |
| `threshold` | Minimum P(jailbreak) for the rule to match. |
| `description` | Human-readable note. |

A rule carries no `include_history` or pattern list. A single response has no
conversation history, and pattern comparison is a request-stage question.

A decision that reads a response-stage signal is not evaluated while the request
is still being routed, since the model has not answered yet. At least one
decision must therefore stay resolvable from request-stage signals alone, which
is checked when the configuration loads.
