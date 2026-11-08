# Hallucination Signal

## Overview

`hallucination` checks the model's answer against the grounding context the
request carried, such as tool results or retrieved documents, and reports the
claims that context does not support. Define its rules under
`recipes[].signals.hallucination`.

This family is learned: the decision model (vela3-fast's span-set adapter by
default) answers it unless a `hallucination` binding names another deployment,
such as Vela Halu in the
[model runtime](../../../model-runtime/guides/hallucination.md); see
[Choose a model for a signal](../../../installation/choose-a-model-for-a-signal.md).
`global.model_catalog.modules.hallucination_mitigation.detector` holds the
detector's settings.

## Key Advantages

- Separates detecting an unsupported claim from what the Router does about it.
- Reports detected, not-detected and unavailable outcomes as distinct states,
  so an answer that could not be checked never reads as verified.
- Scopes the check to the recipe a request resolved to, like every other rule.
- Records one Router Replay outcome per rule with the verdict, the confidence,
  the span count and the action the guard applied.

## What Problem Does It Solve?

A model given evidence can still answer past it: a wrong number, an invented
name, a detail the context never states. A check that lives inside the plugin
acting on it ties detection to enforcement: its result is visible only to that
plugin, and a failed or impossible check looks the same as a clean answer.

`hallucination` solves that by publishing the check as a response-stage
observation under the `hallucination:<name>` key, with the same shape Router
Replay and the plugins read every other signal in.

## When to Use

Use `hallucination` when:

- answers grounded in tool results or retrieved context must be checked before
  they are trusted
- a decision's [`response_guard`](../../plugin/response-guard) should act on
  evidence rather than produce it
- Router Replay should record whether an answer was checked, and with what
  result, whether or not a guard acted on it
- unsupported claims should be reported with the spans they rest on

## Configuration

```yaml
recipes:
  - name: default
    signals:
      hallucination:
        - name: ungrounded_claims
          description: Detect claims the grounding context does not support.
```

A rule has no threshold of its own: the detector's `threshold`,
`min_span_length` and `min_span_confidence` under
`modules.hallucination_mitigation.detector` decide
what counts as an unsupported span, and the rule matches when the detector
found one. Earlier releases could ask for span-level NLI explanations with
`use_nli`; that explainer is retired, and `vllm-sr config migrate` removes the
setting.

### Stage

A hallucination rule is observed at the response stage: it checks the model's
answer, so it only exists once the model has answered. It is not a decision
input. Decisions are selected while the request is being routed, before the
model has answered, so a decision that reads one, directly in its rules or
through a projection, is rejected when the configuration loads. The
observation is read by the [`response_guard`](../../plugin/response-guard) of
the decision selected for the request, whose rules name it. A rule that also
reads the request-stage [fact-check](./fact-check) signal, as in
`AND[fact_check, hallucination]`, checks only the answers to prompts that make
claims worth grounding: `AND` stops at the false fact-check condition, and the
detector does not run.

The answer is checked against the documents RAG retrieved and the request's
own tool results:

- the answer had grounding context and was checked: `detected` or
  `not_detected`, with the detector's confidence under `hallucination:<name>`
- the request carried no tool results or retrieved context: unavailable,
  reported through `SignalErrors` as `hallucination_context_unavailable`
- the detector failed or was never provisioned, or the response carried no
  text to check: unavailable, reported as `hallucination_evaluation_failed`

A guard rule that reads an unavailable rule takes the guard's `on_unknown`, and
annotates with `unverified_factual` rather than `hallucination`. With
`x-vsr-debug`, the `x-vsr-matched-hallucination` header carries the rules that
matched.

A streamed answer is checked once the stream ends and recorded with
`enforcement: not_enforced_streaming` in place of an action. Its bytes are
already with the client by then, so no guard action applies. A stream that
never reaches a terminal answer is not checked, and nothing is recorded.

Declaring a rule is enough to provision the detector for the recipe, even
when no guard reads it.

## Vela 2.0

Bind `hallucination` to a Vela 2.0 deployment and the detector asks the
model's ready-made hallucination question, answered by its router span head,
about the answer, with the request and the grounding context as the other
parts of the question's state:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      hallucination: vela2
```

The model reads the whole answer and applies its own calibrated threshold, so
the detector's `threshold` applies to Vela 1.0 Halu only; `min_span_length`
and `min_span_confidence` filter the spans of both. The
check runs at the response stage, in that stage's call to the deployment. The
same deployment can answer the request's
[`systemone`](systemone.md) and
[`pii`](pii.md#vela-20) questions.

## Verdicts without spans

A `systemone` deployment whose capabilities list neither the `halu` preset nor
`span` questions answers the overall check yes/no: it asks a `noul` question
with the request, grounding context and answer as separate state parts, and
produces a supportedness verdict for the response observation and plugin,
without inventing claim locations. `threshold` applies to this probability;
span filters only apply when the model actually returns spans. The load
reports a `span_fallback` warning, and the Router counts each such answer in
`vsr_signal_span_verdict_fallbacks_total`.

A deployment that answers with spans needs native span support. Both paths
require complete admitted input and grounding context; a failed or incomplete
check is unavailable, not evidence that the answer is supported.

## Dependencies and Limitations

The detector processes the answer and the supplied grounding context on the
deployment that answers the `hallucination` signal.
It can identify text the context does not support; it cannot establish truth
without authoritative evidence, and an answer with no context to check
against is reported as unavailable, not clean. See a complete example:
[`config/fragments/signal/hallucination/grounded-answer.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/hallucination/grounded-answer.yaml).
