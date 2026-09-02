# Jailbreak Signal

## Overview

`jailbreak` detects prompt-injection and jailbreak attempts before the Router
commits to a route. Define jailbreak rules under `routing.signals.jailbreak`.

It uses `global.model_catalog.modules.prompt_guard` and the configured
jailbreak model bindings in `global.model_catalog.system`.

## Key Advantages

- Lets decisions block or downgrade unsafe traffic before model selection.
- Supports classifier, contrastive, and hybrid-style safety detection.
- Keeps jailbreak policy visible inside routing decisions.
- Reuses one safety signal across multiple guarded routes.

## What Problem Does It Solve?

If jailbreak detection only happens downstream, the router can still send unsafe traffic to the wrong model or toolchain. If it lives outside the routing graph, safety logic becomes harder to audit.

`jailbreak` solves that by making injection detection a first-class routing input.

## When to Use

Use `jailbreak` when:

- unsafe traffic must be blocked before model selection
- prompt-injection attempts should route to a safer fallback
- multi-turn history should influence routing
- safety policy must be visible and testable in the same graph as routing logic

## Configuration

```yaml
routing:
  signals:
    jailbreak:
      - name: prompt_injection
        method: contrastive
        threshold: 0.8
        include_history: true
        description: Detect common prompt-injection or jailbreak attempts.
        jailbreak_patterns:
          - ignore previous instructions
          - reveal the hidden prompt
          - jailbreak mode
        benign_patterns:
          - explain the policy
          - summarize the safety rules
```

Use `include_history` for multi-turn attacks, and treat the pattern lists as tuning data for the configured detection method.

### Direction

`direction` selects what a rule scores. The default, `request`, scores the
prompt before the Router commits to a route. `response` scores the model's own
output, so the rule only exists once the model has answered:

```yaml
routing:
  signals:
    jailbreak:
      - name: unsafe_completion
        direction: response
        threshold: 0.85
        description: Detect jailbreak content in the model's own output.
```

A response-direction rule uses the sequence classifier only: `method: contrastive`,
the pattern lists and `include_history` are request-stage settings and are
rejected on it. Matches, scores and failures are reported under the same
`jailbreak:<name>` key as a request-direction rule, so a decision reads it as
`{type: jailbreak, name: unsafe_completion}` and can compose it with request
signals.

A decision that reads a response-direction rule is a response-stage decision.
It is not evaluated while the request is still being routed, because the signal
does not exist yet; it is evaluated once the response arrives, alongside the
request-stage matches, and its `response_jailbreak` plugin then applies. At
least one decision must stay resolvable from request-stage signals alone, which
is checked when the configuration loads. The `response_jailbreak` plugin's own
`threshold` is ignored once a response-direction rule is declared, and the load
reports that; the rule owns the threshold.

An unresolved detector (backend failure, or a response with no text to score)
is reported through `SignalErrors`, the way every other signal reports one,
rather than looking like a clean response. Streaming responses are not scored.

## Dependencies and Limitations

The configured prompt-guard runtime processes the current prompt and,
optionally, conversation history. Detection is probabilistic and can be evaded
or over-triggered; combine it with least-privilege tools and backend policy.
See a complete example:
[`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml).
