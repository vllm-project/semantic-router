# PII Signal

## Overview

`pii` locates personal data in the request, with the exact characters of every span.
It asks the `pii` [preset](../../../system-one/reference/presets#preset-catalog), a
span question over the 11
[personal data labels](../../../system-one/reference/presets#personal-data-labels),
and reads its whole input. A rule matches when the model finds a span of a
label it counts; the [`masking`](../../plugin/masking) plugin replaces the
same spans with placeholders.

## Key Advantages

- Makes privacy-sensitive routing explicit: keep such requests on a private
  model, or refuse them.
- Returns positions, so masking replaces exactly the personal data.
- Ignores the labels a route accepts, such as email addresses.
- Reads tool results and earlier turns when told to.

## What Problem Does It Solve?

Without a personal-data signal, sensitive traffic can reach the wrong model
or plugin stack before anything notices. Ad hoc filters make the policy hard
to audit. `pii` turns personal-data detection into a reusable routing input.

## When to Use

Use `pii` when:

- requests may carry regulated or sensitive personal data;
- some labels are acceptable but others must take a safer route;
- a decision masks personal data before the provider sees it;
- tool results may carry personal data back into the conversation.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          predicate: {gte: 0.85}
          ignore_labels: [EMAIL_ADDRESS]
          input:
            history: true
        - name: tool_result_pii
          input:
            tool_results: true
            request: false
    decisions:
      - name: keep-private
        priority: 200
        rules:
          operator: OR
          conditions:
            - {type: pii, name: personal_data}
            - {type: pii, name: tool_result_pii}
        modelRefs:
          - model: private-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: pii, name: ...}` conditions and the masking plugin's `signal` use |
| `predicate` | The least probability of a span; by default the model's calibrated threshold (0.18 on vela3-fast), else 0.5 |
| `ignore_labels` | Labels the rule never counts |
| `input.history` | Also read the earlier user turns |
| `input.tool_results` | Also read the request's tool results |
| `input.request` | `false` leaves the current user turn out, so the rule reads only the earlier user turns or tool results it names |
| `on_unknown` | What the rule reports when its model gives no answer: `no_match` (the default) or `match` |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

A rule's `predicate` is policy: calibrate the value for the model you use,
or leave it out to use the model's calibrated threshold. vela3-fast's span
adapter applies its own per-length thresholds before it returns a span, so
its 0.18 accepts every span it returns.

### What it reads

The question reads the current user turn, the earlier user turns with
`input.history`, and the textual tool results with `input.tool_results`;
tool calls and non-text content are not read. It also reads what
`global.router.routing_input` removes from the routing text, so personal data
cannot hide in a fake memory block.

A coding agent's user turn carries its harness's own instructions: Claude Code
2.1.317, for example, puts a commit trailer with an email address into a
reminder in the first user message. A rule meant for what the agent's tools
bring back sets `input.request: false` with `input.tool_results: true`, as
`tool_result_pii` above does, so it reads the tool results alone, and leaves
out the current turn with what `routing_input` removed from it. Such a rule is
not asked, and does not match, before the request has a tool result.

It reads its whole input, in windows when the input is longer than one
forward. Text the model did not read (over its input or scan budget, or
truncated) counts as personal data, so text that could not be checked cannot
pass as clean. A scan that misses the signals' deadline gives no answer, which
matches only with `on_unknown: match`.

A model may find a value once and miss its later copies. Every other exact,
whole-word copy of a found value of two or more characters is reported too,
as the same label, so masking replaces every copy.

### When the model gives no answer

A rule whose model gives no answer (it fails, is overloaded or misses its
deadline) does not match. To fail closed instead, write `on_unknown: match`:
the rule then matches such a request, with the match type `unavailable`, and
a decision's `rules.on_unknown` still decides:

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          on_unknown: match
    decisions:
      - name: keep-private
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: pii, name: personal_data}
        modelRefs:
          - model: private-model
```

### The model that answers

The decision model answers every PII rule, unless the `pii` binding or the
rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deploy/deployments). The question needs
spans: a rule that resolves to a deployment without span answers, such as a
d3 model, fails to load with `span_required`, unless that deployment declares
the `pii` preset in its `capabilities`. vela3-fast (the default), the
Vela 3.0 d3 sizes and Vela 2.0 answer it with their own PII question, in the
same call as the request's other questions about the same text.

```yaml
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: SYSTEMONE_API_KEY}
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: primary
```

The signal is a routing control, not a substitute for redaction,
encryption, access control or data-loss prevention. See a complete example:
[`config/fragments/signal/pii/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/pii/strict.yaml).
