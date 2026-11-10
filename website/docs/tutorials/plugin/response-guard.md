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

Declare the signals under `recipes[].signals` and add the plugin under
`recipes[].decisions[].plugins`:

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: unsafe_completion
          stage: response
          predicate: {gte: 0.9}
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
| `rules[].when` | A condition tree over named signals, as in `decisions[].rules`. Jailbreak, safety and `systemone` rules with `stage: response`, and every hallucination and refusal rule, read the answer; every other signal reads the request. `on_error` and `on_unknown` are not accepted on its nodes: the guard's `on_unknown` decides. |
| `rules[].action` | `block` refuses the answer with HTTP 403, or with the stream's error event for a streamed answer; `annotate` delivers it with the rule's warning codes in `x-vsr-warnings`; `none` only records the outcome. |
| `rules[].details` | With `annotate`, also writes a note at the head of the answer. The hallucination note names the unsupported spans. |
| `on_unknown` | What a rule does when a response-stage signal it reads could not be read: the detector failed or timed out, the answer had no text, or there was no grounding context to check it against. `block`, `annotate` (the default) or `none`. |

Rules are read in order, and the first rule that blocks ends the read. A
response-stage signal is evaluated only when a rule reaches it, and once per
answer however many rules read it.

The warning codes a rule adds are `hallucination` (`unverified_factual` when
the answer could not be checked), `response_jailbreak`, and `response_guard`
for any other rule. A blocked answer is replaced with a message that names what
was found, never that a check failed.

The grounding context is the documents the [context plugin](./context.md) retrieved and
the request's own tool results.

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

## Streamed Answers

A guard that can refuse or rewrite an answer (a `block` rule, an `annotate`
rule with `details`, or `on_unknown: block`) applies to streamed answers too.
When the client of such a decision asks for a stream (`stream: true` on Chat
Completions, Responses or Messages), the Router holds the model's stream
until it ends and the guard has read the whole answer. The client then
receives one of:

- the events as the model streamed them, usage chunk and terminal event
  included, with the guard's codes in `x-vsr-warnings`;
- the answer's events with the guard's note at its head, for a rule with
  `details`;
- the refusal a buffered answer's 403 carries, as the protocol's stream
  error: a Chat `error` chunk, a Responses `response.failed` event or a
  Messages `error` event. When a deployment the guard asks takes the
  delegated credential and the request carries none, a buffered answer is
  refused with `401`, and the stream error carries that refusal instead.

Behind Envoy the Router asks for the response body buffered for these
requests only; standalone, it buffers the body itself. The cost is time to
first token: nothing reaches the client before the model has finished, so a
long answer must fit the idle timeouts of the client and of any proxy. A held
answer is bounded at 64 MiB, the size the Router accepts for any model
response, and Envoy's own buffer limit applies before it. A larger answer is
refused unread with the same stream error, and a stream that ends without a
complete answer sends only its failure: nothing the guard has not read
reaches the client.

A guard whose rules only `annotate` without `details` or record (`none`)
keeps true streaming. The client receives the answer as it is generated; the
guard reads it once the stream ends and records the outcome in
[Router Replay](./router-replay), but its codes cannot reach
`x-vsr-warnings`, which left with the first token.

Router Replay records the outcomes of a streamed answer with `stream:
buffered` or `stream: unbuffered`. `x-vsr-trace` carries the guard's outcome
as `response_guard`, with `buffered` for a held stream, and
`llm_response_guard_buffered_total` counts held streams by decision and
outcome: `passed`, `annotated`, `blocked`, `over_limit` or `incomplete`.

## Cached Answers

A [`response_cache`](./response-cache.md) on the same decision stores only the
answers the guard released: never a blocked answer, or one flagged only in
`x-vsr-warnings`. A cache hit is answered without calling the model, so the
guard does not run on it. An exact hit is the same request, so the guard's
verdict on it still holds. A semantic hit serves the answer the guard released
for a similar earlier request, without evaluating the guard's conditions on the
new request, such as a `keyword` or `fact_check` condition. Where that matters,
use `mode: exact` for that decision's cache.

Changing the guard starts the decision's cache fresh. Changing a signal it
reads, such as a jailbreak rule's `predicate`, does not: set a new
`revision.policy_revision` in the decision's `response_cache` configuration.
See [With a response guard](./response-cache.md#with-a-response-guard).
