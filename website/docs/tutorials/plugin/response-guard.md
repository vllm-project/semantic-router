# Response Guard

## Overview

`response_guard` is a route-local plugin that blocks or annotates the model's
answer on the signals it is checked with. Each rule names signals, as a
decision's rules do, and applies its action when its condition holds.

## Key Advantages

- One plugin for every check of an answer: jailbreak content in the output,
  claims the grounding context does not support, and any response-stage
  `systemone` question.
- Signals own their detectors and thresholds; a rule only names them, so one
  detection serves every rule that reads it.
- A condition on the request, such as a fact-check verdict, gates the costly
  checks of the answer: `AND` stops at its first false condition.

## What Problem Does It Solve?

Routing picks a model before it answers, so a decision cannot act on what the
model said. `response_guard` is the step after the answer: it reads the
response-stage signals over the answer and the request-stage ones over the
request, and refuses, flags or passes the answer accordingly.

## When to Use

- an answer that carries jailbreak content must not reach the client
- a grounded or tool-backed route should flag answers its context does not
  support
- a route needs a recorded verdict on its answers without changing them

## Configuration

Declare the signals under `routing.signals` and add the plugin under
`routing.decisions[].plugins`:

```yaml
routing:
  signals:
    jailbreak:
      - name: unsafe_completion
        direction: response
        threshold: 0.9
    fact_check:
      - name: needs_fact_check
        description: Prompts whose answers are worth checking against their context.
    hallucination:
      - name: ungrounded_claims
        description: Answers with claims their grounding context does not support.
  decisions:
    - name: grounded_support
      # ...rules, modelRefs...
      plugins:
        - type: response_guard
          configuration:
            rules:
              - when: {type: jailbreak, name: unsafe_completion}
                action: block
              - when:
                  operator: AND
                  conditions:
                    - {type: fact_check, name: needs_fact_check}
                    - {type: hallucination, name: ungrounded_claims}
                action: annotate
                details: true
            on_unknown: annotate
```

| Field | Meaning |
|---|---|
| `rules[].when` | A condition tree over named signals, as in `decisions[].rules`. Jailbreak rules with `direction: response`, hallucination rules and response-stage `systemone` rules read the answer; every other signal reads the request. `on_error` and `on_unknown` are not accepted on its nodes: the guard's `on_unknown` decides. |
| `rules[].action` | `block` refuses the answer with HTTP 403; `annotate` delivers it with the rule's warning codes in `x-vsr-response-warnings`; `none` only records the outcome. |
| `rules[].details` | With `annotate`, also writes a note at the head of the answer. The hallucination note names the unsupported spans. |
| `on_unknown` | What a rule does when a response-stage signal it reads could not be read: the detector failed or timed out, the answer had no text, or there was no grounding context to check it against. `block`, `annotate` (the default) or `none`. |

Rules are read in order, and the first rule that blocks ends the read. A
response-stage signal is evaluated only when a rule reaches it, and once per
answer however many rules read it.

The warning codes a rule adds are `hallucination` (`unverified_factual` when
the answer could not be checked), `response_jailbreak`, and `response_guard`
for any other rule. A blocked answer is replaced with a message that names what
was found, never that a check failed.

The grounding context is the documents the [RAG plugin](./rag) retrieved and
the request's own tool results.

A streamed answer exists as a whole only once its bytes are with the client,
so its rules are read and recorded in [Router Replay](./router-replay) with
`enforcement: not_enforced_streaming`, and nothing is applied. Use
non-streaming responses where a rule must block.

`POST /api/v1/plugins/response_guard/preview` reads an active binding's guard
over a supplied question, context and answer; `mode: probe` asks the
configured models, without generation or persistence.

A configuration that still uses the `response_jailbreak` or `hallucination`
plugin fails at load with a one-line hint, and `vllm-sr config migrate`
rewrites it as `response_guard` rules over the signals those plugins checked.

See complete examples:
[`config/fragments/plugin/response-guard/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/strict.yaml)
and
[`config/fragments/plugin/response-guard/grounded-answers.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/grounded-answers.yaml).
