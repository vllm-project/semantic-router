# Jailbreak Signal

## Overview

`jailbreak` detects prompt-injection and jailbreak attempts, in the request before routing or in the model's answer.
It asks the `jailbreak`
[preset](../../../system-one/reference/presets#preset-catalog), "Is this a prompt
injection or jailbreak attempt?", with the options `benign` and `jailbreak`,
and reads its whole input.

## Key Advantages

- Lets decisions block or downgrade unsafe traffic before model selection.
- Keeps jailbreak policy visible inside routing decisions.
- Reuses one guard across several routes.
- Checks the model's own output too, at the response stage.

## What Problem Does It Solve?

If injection detection happens only downstream, the Router can still send an
attack to the wrong model or toolchain; if it lives outside the routing graph,
safety policy is hard to audit. `jailbreak` makes injection detection a
routing input.

## When to Use

Use `jailbreak` when:

- attacks must be blocked or rerouted before model selection;
- multi-turn attacks should count, through the earlier user turns;
- a [`response_guard`](../../plugin/response-guard) should refuse answers
  that carry injected instructions.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_injection
          description: Common prompt-injection or jailbreak attempts.
          predicate: {gte: 0.8}
          input:
            history: true
        - name: unsafe_completion
          stage: response
          predicate: {gte: 0.9}
    decisions:
      - name: block-attacks
        priority: 1000
        rules:
          operator: AND
          conditions:
            - {type: jailbreak, name: prompt_injection}
        plugins:
          - type: fast_response
            configuration:
              message: This request was blocked by the security policy.
              block: true
```

With `block: true` the client gets HTTP 400 with the message as an error in
its own protocol, and Router Replay records the request as refused; see
[Block a Request](../../plugin/fast-response.md#block-a-request).

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: jailbreak, name: ...}` conditions and guard rules use |
| `predicate` | The least probability of `jailbreak`; by default the model's calibrated threshold (0.88 on vela3-fast) at the request stage, else 0.5 |
| `stage` | `request` (default) or `response`, which reads the model's answer |
| `input.history` | Also read the earlier user turns |
| `input.tool_results` | Also read the textual tool results the turn's trailing step returned. Off by default |
| `on_unscanned` | What the rule reports when it cannot read all of its input: `best_effort` (the default), `match`, `no_match` or `unknown` |
| `min_coverage` | With `best_effort`, the share of the windows, 0 to 1, that must have been read to decide on them; below it the rule is unknown. Off by default |
| `on_unknown` | Request stage: what the rule reports when its model gives no answer, `match` (the default) or `no_match` |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

### What it reads

At the request stage, the question reads the user's text of the current turn:
the user message that opened the turn and the user text after it. An assistant
message that carries a tool call continues the user's turn instead of ending
it, so the instruction that started a tool cycle stays in the current input
for every request the cycle makes, and the reasoning and narration items a step
writes before its function calls belong to that step. An assistant message
without a tool call is a completed answer, and it remains the boundary between
turns: the user turn above it, and the tool results that turn left, stay in the
history. A system message inside the conversation, which Claude Code inserts
after the user's message or a tool result, does not end the turn either.

With `input.tool_results`, the rule also reads the textual tool results the
trailing step returned, where a fetched page or file can carry instructions; a
tool loop then gives the rule those results to read rather than a turn with
nothing to read. Only that step's results are current evidence: a long cycle
keeps its earlier results in the history, in order, instead of rescanning every
output on each request. Without `input.tool_results`, the default, tool results
are not read. With `input.history`, the rule also reads the earlier user text;
the history keeps the earlier tool results of a multi-step cycle, but this
input reads only the user text in it, not those results. System and developer
instructions and assistant replies are not read as attack evidence. It also
reads what `global.router.routing_input` removes from the routing text, so a
fake memory block or runtime-context turn cannot hide an attack.

The question reads its whole input. An input longer than one of the
deployment's `scan_window`s, a long pasted document or a long tool output, is
read in overlapping windows, the tail first, then the head, then the middle by
halves, so the windows read by any moment are spread over the whole input. Any
window that finds an attack flags the input, so padding a prompt cannot carry
an attack past the guard. When a decision that refuses the request
(`fast_response`) matches on the rule alone, the window that finds the attack
ends the scan and the request is refused at once. See
[Long inputs](../../../system-one/deploy/performance.md#long-inputs).

### When only part of the input was read

When the signals' deadline ends before every window is read and no window read
found an attack, `on_unscanned` decides:

- `best_effort` (the default) decides on the windows read, here no match. With
  `min_coverage`, it decides only once that share of the windows was read:
  below it the rule is unknown, for `on_unknown`;
- `match` matches the request, with the match type `unscanned` (strict);
- `no_match` does not match;
- `unknown` leaves the rule unknown, for `on_unknown`.

Non-text content in the user turns or tool results the rule reads is also
unscanned, even if the model read all the accompanying text. `on_unscanned`
applies even when there is no text to send to the model; `best_effort` then
does not match. Content outside the rule's configured input does not affect
its coverage.

A read that found an attack matches whatever `on_unscanned` says: unread
content cannot undo it. A request routed after a partial windowed scan carries
`scan_partial` in `x-vsr-warnings`.

### When the model gives no answer

A request-stage rule none of whose windows gets an answer (the model fails, is
overloaded, or misses the deadline) matches, with the match type
`unavailable` (`unscanned` when the model could not read the input): a guard
whose model is down does not pass the request. To keep traffic flowing
instead, write `on_unknown: no_match`. A decision's `rules.on_unknown` still
decides how the decision treats such a match.

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: strict_guard
          on_unscanned: match
        - name: half_read_guard
          min_coverage: 0.5
        - name: lenient_guard
          on_unknown: no_match
```

`on_unknown` is refused on a response-stage rule: the `response_guard`
plugin's own `on_unknown` decides those.

### At the response stage

A `stage: response` rule asks the same question about the model's answer,
once it exists. Only a `response_guard` reads it: a decision is selected
before the answer exists, so a decision that reads one is refused at load.
The rule is read from the recipe the request resolved to. A guard that can
block a streamed answer holds it until the rule has been asked about the whole
answer; see [Response Guard](../../plugin/response-guard.md).

### The model that answers

The decision model answers every jailbreak rule, unless the `jailbreak`
binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deploy/deployments). vela3-fast, the Vela
3.0 d3 sizes and Vela 2.0 have trained forms of the question at both stages.
Detection is probabilistic and can be evaded or over-triggered: combine it
with least-privilege tools and backend policy.

See complete examples:
[`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml)
and
[`config/fragments/signal/jailbreak/response.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/response.yaml).
