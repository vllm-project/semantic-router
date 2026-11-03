# Feedback Recovery Recipe Model Card

## Overview

Feedback Recovery changes the route when a conversation shows that the current
answer is not working. It distinguishes a request for clarification from an
incorrect answer, repeated dissatisfaction, failed code, and a request for
verification, then spends more model capacity only where recovery needs it.

The router's decision model answers four named questions about the latest user
message in one call, in any language, and its built-in re-ask judgment compares
that message with the user's earlier turns. Each routing decision reads those
answers. The recipe declares the decision model it was calibrated with,
Vela 2.0 0.8B.

Clients use the `vllm-sr/auto` entrypoint.

## Model details

| Recovery level | Reference alias | Role |
| --- | --- | --- |
| Local | `qwen/qwen3.5-rocm` | Default answers and clarification. |
| Specialist | `google/gemini-3.1-pro` | First recovery for general or coding failures. |
| Premium | `openai/gpt5.4` | Persistent failures and verification-heavy recovery. |
| Visual | `local/omni` | Image-bearing questions and follow-ups. |

The aliases can be rebound to other models with similar cost and capability
roles.

## Intended use

Use this recipe for multi-turn assistants where users may:

- explicitly say that an answer is wrong;
- repeat or rephrase an unanswered question;
- report that generated code still fails;
- request a correction backed by verification; or
- ask a question about an attached image; or
- ask for clarification without requiring a more expensive model.

It is less useful for stateless clients that do not send conversation history,
because the repeated-question judgment needs earlier turns.

## Decision questions

| Question | Type | Asks | Decides |
| --- | --- | --- | --- |
| `wrong` | Yes or no, at 0.5 | Does the latest message say that an earlier answer or fix was wrong, incorrect or did not work? | Whether the request takes a recovery lane. |
| `clarify` | Yes or no, at 0.6 | Does the latest message ask to explain an earlier answer again, more clearly, more simply or with an example? | Whether a follow-up that does not call the answer wrong stays on the low-cost clarification lane. |
| `code` | Yes or no, at 0.5 | Is the request about software, such as code, a program error or exception, a failing test or a bug? | Whether recovery takes a code-repair lane. |
| `sources` | Yes or no, at 0.6 | Does the user ask for sources, citations or a fact check of the answer? | Whether recovery takes the verified lane. |
| `persistently_dissatisfied` | Built-in re-ask judgment, at 0.8 | Does the current user turn repeat the last two user turns? | Whether recovery escalates to a premium lane. |

One conversation fact needs no model: `feedback_has_images` is true when the
request carries image content.

## Routing behavior

Decisions are tried by priority, and the first that matches wins.

| Priority | Decision | Reads | Candidate |
| --- | --- | --- | --- |
| 260 | `omni` | `feedback_has_images` | `local/omni` |
| 240 | `feedback_verified_recovery` | (`wrong` or `persistently_dissatisfied`) and `sources`, not `code` | `openai/gpt5.4` |
| 230 | `feedback_persistent_code_recovery` | `code` and `persistently_dissatisfied` | `openai/gpt5.4` |
| 220 | `feedback_code_recovery` | `code` and `wrong`, not `persistently_dissatisfied` | `google/gemini-3.1-pro` |
| 210 | `feedback_persistent_recovery` | `persistently_dissatisfied`, not `code` or `sources` | `openai/gpt5.4` |
| 200 | `feedback_general_recovery` | `wrong`, not `code`, `sources` or `persistently_dissatisfied` | `google/gemini-3.1-pro` |
| 180 | `feedback_need_clarification` | `clarify`, not `wrong` or `persistently_dissatisfied` | `qwen/qwen3.5-rocm` |
| 10 | `feedback_default` | Nothing; the fallback | `qwen/qwen3.5-rocm` |

A message that says an answer was wrong and also asks to explain it again is
a recovery, not a clarification: `wrong` is tried first, and the clarification
lane requires no `wrong`. Code repair takes a code failure even when it asks
for sources.

When the decision model cannot answer, every question is unknown, no recovery
lane matches, and the request stays on `feedback_default`. Keyword lists are
not needed as a fallback, because that low-cost default is the safe outcome
for an unknown answer.

## Calibration notes

The questions and thresholds were calibrated on the maintained probes with
Vela 2.0 0.8B:

- The single-message wrong-answer probes score 0.67 or more for `wrong`, and
  the clarification requests and ordinary messages 0.36 or less.
- The clarification requests score 0.93 or more for `clarify`, and ordinary
  messages 0.36 or less. Wrong-answer messages that also ask for a new
  explanation score high too, which is why the clarification lane excludes
  `wrong`.
- The code-repair probes score 0.84 or more for `code`, and the others 0.29
  or less.
- The verified-recovery probes score 0.87 or more for `sources`. The others
  score 0.45 or less, except a code failure that also asks for sources (0.75),
  which code repair takes.
- A single choice between a new request, a wrong answer and a clarification
  put "That answer was wrong. Explain it again" on clarification. Two separate
  questions, with `wrong` tried first, route it as a recovery.
- A question that asked for verification or evidence read "Prove that the sum
  of the first n integers is n(n+1)/2" as a request for evidence (0.90); asking
  only for sources, citations or a fact check scores it 0.45.

When you change a question or a threshold, or move to another decision model,
rerun the probes.

## Requirements

- Reachable OpenAI-compatible endpoints for the four configured aliases.
- Conversation history in the request for re-ask and persistence detection.
- The decision model for the four questions and the re-ask judgment. The
  recipe declares it as the `primary` deployment of Vela 2.0 0.8B, which
  `global.model_catalog.system.decision_model` names. The deployment sets no
  device, so the model runtime uses a GPU when the router has one and the CPU
  otherwise.
  In the CPU conformance run, with the router limited to 4 CPUs and 4
  requests in flight, the median request took 13.5 s and the 95th
  percentile 16.3 s.
  Serve it on a GPU for live traffic. With no device set, the runtime uses a
  GPU when the router has one; set `device: rocm` or `device: cuda` on the
  `primary` deployment to require one. For a smaller model, change its
  `artifact` (for example to `vllm-sr/Vela-2.0-0.3B`) or point
  `global.model_catalog.system.decision_model.deployment` at another
  deployment, and rerun the probes, because the thresholds were calibrated on
  the 0.8B.
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
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Feedback Routing** in
**Recipes**, assign each recovery lane, and publish the resulting
Mixture-of-Model Entrypoint.

## Evaluation

The 32 probes cover first and repeated corrections, coding failures,
verification requests, clarification, rephrasing, Spanish and Japanese
follow-ups, and negative cases where ordinary traffic must remain on the
default lane. CPU conformance runs them with the declared Vela 2.0 0.8B, which
reaches the expected decision for all 32. See [`probes.yaml`](probes.yaml) and
the [conformance guide](../CONFORMANCE.md).

## Limitations

- Missing or truncated history weakens re-ask and persistence detection.
- User dissatisfaction does not reveal the true cause of an error; escalation
  can improve capability but cannot guarantee a correct answer.
- The decision model reads only the latest message for the four questions, so
  it cannot check what the earlier answer said; it judges the user's reaction.
- The reference aliases and prices are examples rather than service-level
  promises.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
