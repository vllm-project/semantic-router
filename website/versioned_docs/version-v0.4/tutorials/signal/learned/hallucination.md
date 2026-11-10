# Hallucination Signal

## Overview

`hallucination` checks the model's answer against the grounding context the
request carried, such as tool results or retrieved documents, and reports the
claims that context does not support. Define its rules under
`routing.signals.hallucination`.

This family is learned: it relies on the hallucination detector under
`global.model_catalog.modules.hallucination_mitigation.hallucination_model`,
and on the explainer NLI model when a rule asks for explanations.

## Key Advantages

- Separates detecting an unsupported claim from what the Router does about it.
- Reports detected, not-detected and unavailable outcomes as distinct states,
  so an answer that could not be checked never reads as verified.
- Scopes the check to the recipe a request resolved to, like every other rule.
- Records one Router Replay outcome per rule with the verdict, the confidence,
  the span count and the action the plugin applied.

## What Problem Does It Solve?

A model given evidence can still answer past it: a wrong number, an invented
name, a detail the context never states. Checking the answer inside the
`hallucination` plugin tied detection to enforcement: the check only ran where
the plugin was enabled, its result was visible only to that plugin, and a
failed or impossible check looked the same as a clean answer.

`hallucination` solves that by publishing the check as a response-stage
observation under the `hallucination:<name>` key, with the same shape Router
Replay and the plugins read every other signal in.

## When to Use

Use `hallucination` when:

- answers grounded in tool results or retrieved context must be checked before
  they are trusted
- a decision's `hallucination` plugin should enforce on evidence rather than
  produce it
- Router Replay should record whether an answer was checked, and with what
  result, whether or not a plugin acted on it
- unsupported claims should be reported with the spans they rest on

## Configuration

```yaml
routing:
  signals:
    hallucination:
      - name: ungrounded_claims
        use_nli: true
        description: Detect claims the grounding context does not support.
```

A rule has no threshold of its own: the detector's `threshold`,
`min_span_length` and `min_span_confidence` on `hallucination_model` decide
what counts as an unsupported span, and the rule matches when the detector
found one. `use_nli` asks the detector for span-level NLI explanations; it is
a detection setting, so it lives on the rule, and the plugin's own `use_nli`
is reported as ignored once a rule is declared.

### Stage

A hallucination rule is observed at the response stage: it checks the model's
answer, so it only exists once the model has answered. It is not a decision
input. Decisions are selected while the request is being routed, before the
model has answered, so a decision that reads one, directly in its rules or
through a projection, is rejected when the configuration loads. The
observation is consumed by the `hallucination` plugin of the decision selected
for the request, which applies `hallucination_action` to a match and
`unverified_factual_action` to an answer that had nothing to be checked
against.

The detector only has something to check when the request-stage
[fact-check](./fact-check) signal said the prompt makes claims worth grounding
and the request carried context to ground them against:

- the answer had grounding context and was checked: `detected` or
  `not_detected`, with the detector's confidence under `hallucination:<name>`
- the prompt needed grounding but the request carried no tool results or
  retrieved context: unavailable, reported through `SignalErrors` as
  `hallucination_context_unavailable`, which is what the plugin's
  `unverified_factual_action` acts on
- the detector failed or was never provisioned, or the response carried no
  text to check: unavailable, reported as `hallucination_evaluation_failed`
- the prompt made no claims worth grounding: the rule does not apply and is
  left unpublished; Router Replay records it as `not_applicable`

With `x-vsr-debug`, the `x-vsr-matched-hallucination` header carries the rules
that matched.

A streamed answer is checked once the stream ends and recorded with
`enforcement: not_enforced_streaming` in place of an action. Its bytes are
already with the client by then, so the `hallucination` plugin does not run and
neither `hallucination_action` nor `unverified_factual_action` applies. A stream
that never reaches a terminal answer is not checked, and nothing is recorded.

Declaring a rule is enough to provision the detector for the recipe, and
`use_nli: true` the explainer, even when no decision enables the plugin. A
decision whose `hallucination` plugin runs with no rule declared is reported at
load: the plugin is then classifying the answer itself, which is the
compatibility path.

## Dependencies and Limitations

The detector processes the answer and the supplied grounding context through
`global.model_catalog.modules.hallucination_mitigation.hallucination_model`.
It can identify text the context does not support; it cannot establish truth
without authoritative evidence, and an answer with no context to check
against is reported as unavailable, not clean. See a complete example:
[`config/fragments/signal/hallucination/grounded-answer.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/hallucination/grounded-answer.yaml).
