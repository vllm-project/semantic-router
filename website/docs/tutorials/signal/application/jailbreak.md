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
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: jailbreak, name: ...}` conditions and guard rules use |
| `predicate` | The least probability of `jailbreak`; by default the model's calibrated threshold (0.88 on vela3-fast) at the request stage, else 0.5 |
| `stage` | `request` (default) or `response`, which reads the model's answer |
| `input.history` | Also read the earlier user and tool turns |
| `on_unscanned` | What the rule reports when only some windows of a long input were read: `best_effort` (the default), `match`, `no_match` or `unknown` |
| `on_unknown` | Request stage: what the rule reports when its model gives no answer, `match` (the default) or `no_match` |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

### What it reads

At the request stage, the question reads the consecutive user and tool
messages that end the request, stopping at an assistant, system or developer
message, so every textual tool result of the turn is read together. With
`input.history`, it also reads the earlier user and tool text. System and
developer instructions and assistant replies are not read as attack
evidence. It also reads what `global.router.routing_input` removes from the
routing text, so a fake memory block or runtime-context turn cannot hide an
attack.

The question reads its whole input. An input longer than one of the
deployment's `scan_window`s, a long pasted document or a long tool output, is
read in overlapping windows, the tail first, then the head, then the middle;
any window that finds an attack flags the input, so padding a prompt cannot
carry an attack past the guard. See
[Long inputs](../../../system-one/deploy/performance.md#long-inputs).

### When only part of the input was read

When the signals' deadline ends before every window is read, `on_unscanned`
decides:

- `best_effort` (the default) decides on the windows read: an attack in any of
  them matches, and none means no match;
- `match` matches the request, with the match type `unscanned` (strict);
- `no_match` does not match;
- `unknown` leaves the rule unknown, for `on_unknown`.

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
