---
title: Model Reasoning
description: Inherit, reuse, or define a model reasoning contract and select its mode and effort from routing decisions.
---

# Model reasoning

A reasoning family tells the Router how a model represents thinking controls.
It describes the native parameter, supported modes, optional effort ladder,
default values, and disabled value. The selected Provider mapping then projects
that model contract onto the Provider's request protocol.

## Choose who owns the family

| Model kind | Configuration | Owner |
| --- | --- | --- |
| Catalog-backed | Set only `catalog`. | Built-in Model Card and Provider mapping. |
| Custom using a known family | Set `reasoning.family`. | Built-in reasoning-family catalog. |
| Custom using a new family | Set the inline fields under `reasoning`. | This model declaration. |
| Custom pass-through | Omit `reasoning`. | Upstream request and backend behavior. |

A catalog-backed Model cannot override reasoning. A custom Model must choose a
built-in family reference or an inline definition, never both.

## Reuse a built-in family

```yaml
providers:
  models:
    - name: private-qwen
      provider_model_id: Qwen3-32B
      reasoning:
        family: qwen3
      backend_refs:
        - provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
```

In the Dashboard, open **Models → Backend models**, choose **Connect model**,
select the Provider, and choose **Advanced settings → Reasoning family**. The
**Reasoning families** tab on the Backend models page shows the available IDs
and their native parameters.

## Define a custom family inline

When no built-in family matches, attach the complete contract to the custom
Model:

```yaml
providers:
  models:
    - name: private-reasoner
      provider_model_id: private-reasoner-v2
      reasoning:
        type: reasoning_effort
        parameter: reasoning_effort
        activation_parameter: enable_thinking
        levels: [low, medium, high]
        default: medium
        modes: [enabled, disabled]
        default_mode: enabled
        disabled: none
      backend_refs:
        - provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
```

Canonical v0.5 has no user-authored global family registry. The legacy
`providers.defaults.reasoning_families` plus per-model `reasoning_family`
combination migrates to `providers.models[].reasoning`: built-in IDs become a
`family` reference, while custom definitions are copied inline onto each custom
Model that uses them.

Use **Add manually** in the Dashboard to author an inline family. Leave
**Built-in Catalog Model** and **Reasoning Family** empty, then fill the
**Inline Reasoning** fields.

## Understand the inline fields

| Field | Required behavior |
| --- | --- |
| `type` | One of `chat_template_kwargs`, `reasoning_effort`, `reasoning_mode`, or `top_level_reasoning_effort`. |
| `parameter` | Native request parameter controlled by the family. Required with `type`. |
| `levels` | Allowed effort names. Required for effort-based types. |
| `default` | Default effort; when set, it must appear in `levels`. |
| `modes` | Any supported values from `enabled`, `disabled`, and `adaptive`. |
| `default_mode` | Required when `modes` is set and must appear in that list. |
| `disabled` | Native value used to disable reasoning; `disabled` mode must be supported. |
| `activation_parameter` | Separate on/off parameter for `reasoning_effort`; it must differ from `parameter`. |
| `effort_flags` | Advanced mapping from effort names to boolean parameters. It requires `activation_parameter`, and every key must appear in `levels`. |

For `top_level_reasoning_effort`, `parameter` must be `reasoning_effort`. A
mode-only family does not accept `reasoning_effort` in a decision. An always-on
family cannot be selected with `use_reasoning: false`.

## Select reasoning in a decision

After the Model has a family, configure its decision reference:

```yaml
recipes:
  - name: default
    decisions:
      - name: complex-request
        rules:
          operator: AND
          conditions:
            - type: complexity
              name: needs_reasoning
              label: hard
        modelRefs:
          - model: private-reasoner
            use_reasoning: true
            reasoning_mode: enabled
            reasoning_effort: high
```

The effort must be listed by the family and allowed by every Provider binding
for that Model. `reasoning_mode` must agree with `use_reasoning`. Omit an effort
to use the family default, or set `providers.defaults.reasoning_effort` when
one valid deployment-wide default should take precedence. The default applies
to every Model whose family lists it, so those Models' Provider bindings must
allow it too; a Model whose family does not list it keeps the family default.

A `modelRefs` entry without `use_reasoning` counts as `use_reasoning: false`.
For a Model with a family, the decision's choice replaces the client's: the
backend receives reasoning off when `use_reasoning` is false, even if the
client asked for thinking, and the decision's or the family's effort when it is
true, whatever effort the client sent. This holds on Chat, Responses and
Messages backends alike, with two exceptions. An Anthropic client that turns
thinking off explicitly keeps it off on a Chat or Responses backend. And a
`reasoning_mode` family on a Responses backend gets its mode only through the
`chat_template_kwargs` reasoning transport, the default for compatible
providers: the Responses request has no field for a named mode, so behind a
provider with any other transport the Router removes the client's reasoning
fields, sends no control, and the backend runs the model's default mode
whatever `use_reasoning` says. A pass-through Model keeps the client's
reasoning fields as sent, and the routing log, metrics and trace report no
effort for it: its `routing_decision` log line says `reasoning_projected: false`.
Its Router Replay record has no `reasoning_mode` and keeps an effort only when
the client sent one.

On an `api_format: anthropic` backend, `enabled` thinking needs a
`budget_tokens` value. When the client sent none, an `enabled` mode reaches the
backend as `thinking.type: adaptive` with `output_config.effort` set to the
selected effort.

A decision that turns thinking off on an `api_format: anthropic` backend sends
`thinking.type: disabled` and no effort. That covers the per-message effort
changes of a Messages client too: a `system` message in `messages` that carries
its own `output_config.effort`. The Router removes that field, and removes the
message when it holds nothing else. With thinking off the backend takes one
effort level for the whole request, and Claude Haiku 5.5 rejects a request that
changes it. When the decision keeps thinking on, per-message effort changes
reach the backend as the client sent them.

In the Dashboard decision editor, select the Model first. The UI derives the
mode and effort choices from its effective family and disables controls that
the Model or Provider cannot support. A custom Model without a family stays a
pass-through model and has no Dashboard reasoning controls.

## Turn reasoning off for small output limits

A backend counts reasoning against the request's output limit. When a decision
turns reasoning on, a request with a small limit can spend all of it thinking
and end with no answer: a Messages response holds only a thinking block, a
Chat response has `content: null`, and a Responses response holds only a
reasoning item. Set the smallest limit that keeps the decision's reasoning:

```yaml
providers:
  defaults:
    reasoning_min_output_tokens: 512
```

- **No preference and a small limit: reasoning off.** A request that states no
  reasoning preference and sets an output limit below the threshold runs with
  the model's reasoning off. The limit is Messages `max_tokens`, Chat
  `max_tokens` or `max_completion_tokens`, or Responses `max_output_tokens`.
- **What states a preference.** Messages `thinking` of any type; a Chat
  `reasoning_effort` (`none` and `minimal` included), `reasoning_budget_tokens`,
  or an `enable_thinking`, `thinking` or `reasoning_effort` member of
  `chat_template_kwargs`; a Responses `reasoning.effort` or
  `reasoning.summary`. A Messages `output_config.effort` alone does not, since
  it sets the effort of the whole answer. A request that states a preference
  is routed as before, whatever its limit.
- **Off through the model's own control.** The backend receives the
  reasoning-off control of the model's family, as with `use_reasoning: false`.
  A model whose family has none, such as an always-on family, keeps its
  reasoning. A fallback candidate follows the same rule.
- **Unchanged otherwise.** A request without an output limit, or with a limit
  at or above the threshold, keeps the decision's reasoning, and its thinking
  is returned as before. Unset or `0`, the default, turns the rule off.
- **Reported.** The `reasoning` member of `x-vsr-trace` reads
  `off;reason=output_budget`, and Router Replay records the reasoning mode as
  `off`.

## Turn reasoning off for requests with stop sequences

Backends such as vLLM match stop sequences against all the text the model
generates, its reasoning included. When a decision turns reasoning on for a
client that asked for none, a stop sequence the reasoning writes ends the
request before any answer: a Messages response holds only a thinking block
with the `stop_sequence` stop reason, and a Chat response has
`content: null`. The Router therefore runs such a request with the model's
reasoning off. There is nothing to configure.

- **Stop sequences and no preference: reasoning off.** A request that sets
  stop sequences, Chat `stop` or Messages `stop_sequences`, and states no
  reasoning preference runs with the model's reasoning off and keeps its stop
  sequences. Responses has no stop sequences.
- **A stated preference is routed as before.** The preferences are those
  listed under [small output limits](#turn-reasoning-off-for-small-output-limits).
  A request that states one keeps the reasoning its decision chooses.
- **Off through the model's own control.** As for small output limits, the
  backend receives the reasoning-off control of the model's family, a model
  whose family has none keeps its reasoning, and a fallback candidate follows
  the same rule. A decision whose `request_params` plugin blocks `stop` sends
  the backend no stop sequences and keeps its reasoning.
- **Reported.** The `reasoning` member of `x-vsr-trace` reads
  `off;reason=stop_sequences`, and Router Replay records the reasoning mode as
  `off`. When the request's output limit is also below
  `providers.defaults.reasoning_min_output_tokens`, the reason reads
  `output_budget`.
- **Not covered.** A request that states a reasoning preference and sets stop
  sequences reaches the backend with both when its decision turns reasoning
  on, and on such a backend a stop sequence the reasoning writes still ends
  the request inside the reasoning. The Router forwards stop sequences and
  does not apply them itself.
