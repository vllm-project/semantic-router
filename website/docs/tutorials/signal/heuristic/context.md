# Context Signal

## Overview

`context` detects requests that need a larger effective context window. Define
context rules under `routing.signals.context`.

This family is heuristic: it routes from token-window requirements rather than classifier inference.

## Key Advantages

- Keeps long-context routing explicit instead of burying it in model defaults.
- Prevents short prompts from paying the cost of oversized context models.
- Reuses one context threshold across multiple decisions.
- Works well alongside domain or complexity signals.

## What Problem Does It Solve?

Two prompts can ask about the same topic but require very different context windows. If routing only looks at domain, long documents can land on models that truncate or fail.

`context` solves that by making context-window needs a first-class routing input.

## When to Use

Use `context` when:

- some routes need 32K, 128K, or larger context support
- long-document traffic should use a different model family
- you want short requests to stay on cheaper or faster models
- routing depends on context size rather than topic alone

## Configuration

```yaml
routing:
  signals:
    context:
      - name: long_context
        min_tokens: 32K
        max_tokens: 256K
        description: Requests that need a larger effective context window.
```

Use `context` when the router should switch candidates based on prompt length or expected context demand.

## Range Semantics

Each rule is an inclusive token band: it matches when
`min_tokens <= token_count <= max_tokens`.

- Both limits are optional, but at least one must be set. A missing
  `min_tokens` means 0.
- Omit `max_tokens` to make the band open-ended. Every request at or above
  `min_tokens` matches, with no upper limit. Use this on the last band so
  overflow above your largest bounded band still carries a context signal.
- Setting `min_tokens` equal to `max_tokens` is an exact-match band for that
  one token count.
- Every matching rule is reported, in configuration order. Overlapping bands
  are allowed and both names appear in `x-vsr-matched-context`.
- Gaps and overlaps between bands are logged as warnings when the config
  loads. Requests inside a gap match no context rule.
- Validation rejects a rule with neither limit, unparsable, negative, or
  oversized values, and `min_tokens` above `max_tokens`. The Router, the `vllm-sr`
  CLI, and the Dashboard apply the same rules, so a band that passes
  `vllm-sr validate` also loads in the Router.

Values accept `K` and `M` suffixes (`1.5K`, `0.5M`).

```yaml
routing:
  signals:
    context:
      - name: short_context
        min_tokens: 0
        max_tokens: 8K
      - name: medium_context
        min_tokens: 8001
        max_tokens: 64K
      - name: long_context
        min_tokens: 64001
        description: Open-ended band; matches everything above 64K tokens.
```

Context bands are routing signals only. They do not change model
context-window limits, which the Router still enforces separately when it
filters candidates.

## Dependencies and Limitations

Token estimates depend on the request representation and are not a guarantee
that a backend accepts the resulting prompt. Keep model-card context windows
accurate and allow room for generated output. Before selection, the Router
removes decision candidates whose configured, positive context window is
smaller than the estimated request. Missing context metadata remains eligible
for backward compatibility; if every candidate has a known insufficient
window, the Router rejects the request instead of forwarding it to an
ineligible backend. See a complete example:
[`config/fragments/signal/context/long-context.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/context/long-context.yaml).
