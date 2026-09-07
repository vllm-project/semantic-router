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
`jailbreak:<name>` key as a request-direction rule. Router Replay records the
observation as one outcome per response-direction rule, with the verdict
(`detected`, `not_detected` or `unavailable`), the score it thresholded or the
failure code, and the action the plugin applied; with `x-vsr-debug`, the
`x-vsr-matched-jailbreak` header carries the matched response rules after the
request ones.

A response-direction rule is not a decision input. Decisions are selected while
the request is being routed, before the model has answered, so a decision that
reads one, directly in its rules or through a projection, is rejected when the
configuration loads. The observation is
consumed by the `response_jailbreak` plugin of the decision selected for the
request, which applies its configured action to it. The rule is read from the
recipe the request resolved to, so a rule declared on one entrypoint's recipe
scores only that entrypoint's responses. The plugin's own `threshold` is ignored
once a response-direction rule is declared, and the load reports that; the rule
owns the threshold. A decision whose `response_jailbreak` plugin runs with no
response-direction rule declared is also reported at load: the plugin is then
classifying the response itself, which is the compatibility path. Either
consumer is enough to provision `prompt_guard` for the recipe: the jailbreak
model and its label mapping are loaded for the response stage even when no
decision rule reads a jailbreak signal.

An unresolved detector (backend failure, or a response with no text to score)
is reported through `SignalErrors`, the way every other signal reports one,
rather than looking like a clean response. A response is clean only when every
chunk of it was scored: a chunk the backend failed on leaves the rule
unresolved unless the score the other chunks produced already matches it. The
response is scored once and each rule draws its own line across that score, so
a partial scan is resolved per rule: a score of 0.5 matches a rule at 0.4 and
leaves a rule at 0.9 unresolved, because the chunk that was never scored is
where a higher score would have been. Streaming responses are not scored.

## Dependencies and Limitations

The configured prompt-guard runtime processes the current prompt and,
optionally, conversation history. Detection is probabilistic and can be evaded
or over-triggered; combine it with least-privilege tools and backend policy.
See a complete example:
[`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml).
