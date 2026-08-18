# Feedback Recovery Recipe Model Card

## Overview

Feedback Recovery changes the route when a conversation shows that the current
answer is not working. It distinguishes a request for clarification from an
incorrect answer, repeated dissatisfaction, failed code, and a request for
verification, then spends more model capacity only where recovery needs it.

Clients use the `vllm-sr/auto` entrypoint.

## Model details

| Recovery level | Reference alias | Role |
| --- | --- | --- |
| Local | `qwen/qwen3.5-rocm` | Default answers and clarification. |
| Specialist | `google/gemini-3.1-pro` | First recovery for general or coding failures. |
| Premium | `openai/gpt5.4` | Persistent failures and verification-heavy recovery. |

The aliases can be rebound to other models with similar cost and capability
roles.

## Intended use

Use this recipe for multi-turn assistants where users may:

- explicitly say that an answer is wrong;
- repeat or rephrase an unanswered question;
- report that generated code still fails;
- request a correction backed by verification; or
- ask for clarification without requiring a more expensive model.

It is not useful for stateless clients that do not send conversation history,
because repeated-question and prior-answer signals need earlier turns.

## Routing behavior

| Conversation evidence | Behavior |
| --- | --- |
| Clarification request | Stay on the local model and ask a focused follow-up. |
| First general correction or re-ask | Move to the specialist recovery lane. |
| First coding failure | Use the specialist with reasoning enabled. |
| Persistent general or coding failure | Escalate to the premium recovery lane. |
| Explicit verification need after a wrong answer | Use the strongest verified-recovery route. |
| No feedback signal | Use the local default. |

Repeated dissatisfaction has a bounded lookback. Explicit correction and
verification evidence take priority over the ordinary fallback.

## Requirements

- Reachable OpenAI-compatible endpoints for the three configured aliases.
- Conversation history in the request for re-ask and persistence detection.
- Feedback, fact-check, domain, context, and keyword signals configured in
  [`config.yaml`](config.yaml).
- A response-cache store when cache behavior is desired.

## Data handling and safety

The checked-in recipe does not enable Router Replay. Its clarification route
does enable an in-memory response cache with a 30-minute TTL, so an equivalent
request and its prior response can be retained and reused while the Router is
running. Other routes do not opt into that cache.

Conversation history is sent to the selected provider because feedback and
re-ask detection depend on earlier turns. Review provider logging, cache use,
and history retention before using the recipe with sensitive conversations.

## Quick start

```bash
vllm-sr validate --config config/recipes/feedback/config.yaml
vllm-sr serve --config config/recipes/feedback/config.yaml
```

## Evaluation

The probes cover first and repeated corrections, coding failures, verification
requests, clarification, rephrasing, and negative cases where ordinary traffic
must remain on the default lane. See [`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md).

## Limitations

- Missing or truncated history weakens re-ask and persistence detection.
- User dissatisfaction does not reveal the true cause of an error; escalation
  can improve capability but cannot guarantee a correct answer.
- Feedback and fact-check classifiers require calibration for the deployment's
  languages and domains.
- The reference aliases and prices are examples rather than service-level
  promises.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
