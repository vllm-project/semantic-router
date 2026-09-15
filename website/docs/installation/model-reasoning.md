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

In the Dashboard, open **Build → Models → Add Model**, select the Provider, and
choose **Advanced settings → Reasoning family**. The **Built-in Reasoning
Families** table on the Models page shows the available IDs and their native
parameters.

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

Canonical v0.3 has no user-authored global family registry. The legacy
`providers.defaults.reasoning_families` plus per-model `reasoning_family`
combination migrates to `providers.models[].reasoning`: built-in IDs become a
`family` reference, while custom definitions are copied inline onto each custom
Model that uses them.

Use **Manual setup** in the Dashboard to author an inline family. Leave
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
routing:
  decisions:
    - name: complex-request
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: needs_reasoning:hard
      modelRefs:
        - model: private-reasoner
          use_reasoning: true
          reasoning_mode: enabled
          reasoning_effort: high
          max_completion_tokens: 2048
```

The effort must be listed by the family and allowed by every Provider binding
for that Model. `reasoning_mode` must agree with `use_reasoning`. Omit an effort
to use the family default, or set `providers.defaults.reasoning_effort` when
one valid deployment-wide default should take precedence.

`max_completion_tokens` is an optional per-model completion ceiling. When set,
provider dispatch takes the strictest (minimum) of the client request, the
decision `request_params.max_tokens_limit` plugin, this ModelRef value, any
Looper algorithm/stage bound, and a future request-wide ledger. Omission leaves
current behavior unchanged. The same ceiling applies to ordinary routes and to
Looper hops, including Confidence escalation, so each selected model is capped
independently at dispatch.

In the Dashboard decision editor, select the Model first. The UI derives the
mode and effort choices from its effective family and disables controls that
the Model or Provider cannot support. A custom Model without a family stays a
pass-through model and has no Dashboard reasoning controls.
