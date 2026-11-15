---
title: System One schemas
sidebar_label: Schemas
sidebar_position: 2
description: Which System One schema a listener speaks to its clients and an upstream is spoken to in, how the Router converts between them, and how to upgrade.
---

# System One schemas

A schema ID names whose specification and which version a System One API
follows. The Router speaks three:

| Schema | What it is |
| --- | --- |
| `systemone/0.2` | The published System One standard, version 0.2. The repository keeps its own description of the interface in `config/schemas/systemone`, with the source and the sha256 it was checked against. |
| `vllm-sr/5.0` | The standard plus the vllm-sr extension set 5.0: `roles`, `options`, question hints such as `preset` and `over`, the `set` and `span` question types, video assets, the batch endpoint, flat error fields and model cards. See the [extension reference](../reference/extension-reference.md). |
| `vllm-sr/4.0` | The earlier runtime contract 4.0.0, kept so servers and clients that have not upgraded keep working. |

Each side of the Router names its own schema, and the two are independent:
a listener can keep `vllm-sr/4.0` for its clients while every backend speaks
`vllm-sr/5.0`.

## The schema a listener speaks

A listener serves the System One API when its `apis` lists `systemone`:

```yaml
listeners:
  - name: decision-api
    port: 8899
    apis:
      systemone:
        schema: vllm-sr/4.0      # default vllm-sr/5.0
        models: [d3, vela3-fast]
        max_deadline_ms: 600000
```

| Schema | Requests | Answers |
| --- | --- | --- |
| `vllm-sr/5.0` (default) | Strict: any member 5.0 does not define is refused with 400, naming it and, for a 4.0 form, its replacement. An object state without `roles` gives no part a role. Media placed in front of the state is read there. | The standard `detail` (a list on a 422) with flat `code`, `question`, `state`, `reason` and `limit`; 529 for an overloaded server; batches under `responses`. |
| `vllm-sr/4.0` | Every form 4.0.0 accepted (`choices`, `levels`, `labels`, `images`, `videos`, `states` batches, preset-only questions, `options.long_context`) and every 5.0 form, on the same paths and with no header. An object state without `roles` keeps 4.0's implicit roles, and an asset is read where its placeholder is. | Today's 4.0 shapes; an error carries 4.0's `error` and the standard's `detail` with the flat fields, so either kind of client reads it. |

`/v1/decisions` is native on every listener: it takes the Router's internal
request form (below) and answers each failed question in its own answer of a
200, whatever the schema.

### When a listener can switch to 5.0

The Router counts what each listener's clients send in
`vsr_systemone_listener_diagnostics_total{listener, schema, field, action}`:

- `action="legacy_form"`, by `field`: the 4.0 forms a request used (`states`,
  `images`, `videos`, `choices`, `levels`, `labels`, `preset_only`,
  `long_context`), and two valid 5.0 requests a 5.0 listener reads
  differently: `implicit_roles`, an object state without `roles`, and
  `placed_assets`, a state of asset placeholders in front of one item;
- `field="request"` with `action="passed_through"` or `"translated"`: how each
  request went on.

When no client uses a legacy form any more, switch the listener to
`vllm-sr/5.0`, announcing it to its clients first. A listener can stay at
`vllm-sr/4.0` indefinitely.

## The schema an upstream speaks

A `provider: systemone` deployment, and a provider model with
`api_format: systemone`, name the schema their backend speaks. The default is
the published standard:

```yaml
global:
  model_catalog:
    deployments:
      serve-d3:                    # vllm-sr serve
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: vllm-sr/d3
        schema: vllm-sr/5.0
      gateway-d3:                  # a decision model behind an enterprise LLM gateway
        provider: systemone
        endpoint: https://gateway.example.com/v1/systemone
        model: d3
        schema: systemone/0.2      # the default
        deviations: [state_text]
        max_concurrency: 8         # the default
        timeout_ms: 6000
        auth: {header: X-Gateway-Key, prefix: "", api_key_env: GATEWAY_KEY}
providers:
  models:
    - name: pool-d3                # a vllm-sr/auto pool member
      api_format: systemone
      schema: vllm-sr/4.0          # a backend not yet upgraded
      provider_model_id: vllm-sr/d3
      backend_refs:
        - provider: systemone-compatible
          endpoint: https://old-decisions.example.com/v1
```

| Schema | What the Router sends |
| --- | --- |
| `vllm-sr/5.0` | The full 5.0 request: explicit `roles` (the roles the Router's part names have always had), presets as typed questions with a `preset` hint, media in front of the state, and a stage's states in one call to `<endpoint>/batch`. |
| `vllm-sr/4.0` | Today's 4.0 bodies: a stage as one `states` batch. Roles other than the ones a 4.0 server's state keys imply are refused. |
| `systemone/0.2` | Standard requests only: a stage split into one call per state, at most `max_concurrency` in flight (a sliding window, every call within the caller's deadline); presets expanded into their canonical `instructions` and `criteria`; `roles`, `options` and other hints stripped; no deadline field (the client's timeout applies). Span, set, video and full-coverage rules fail to load, naming the rule and the deployment. The standard has no model card, so the deployment's declared `capabilities` apply. |

A call that misses the deadline leaves only its own state's questions unknown,
so their `on_error` applies. The Router reads the standard's 400 `detail`, 422
`detail` list (asking once more without a question it names) and 429 or 529
`Retry-After` (waiting once when it fits the deadline).

### Deviations

`deviations` declare the ways a backend departs from its schema, each
documented and tested on its own:

| Deviation | Schema | What it does |
| --- | --- | --- |
| `state_text` | `systemone/0.2` | Sends an object or array state as its exact `json.dumps(state, ensure_ascii=False)` text: `", "` and `": "` separators, keys in order, no `\u` escaping of non-ASCII. Decision 3 renders an object state as exactly that text, so it reads both the same; Vela reads a JSON-object string as the object. For a backend that refuses any state but a string. |

## How the Router converts

Every conversion goes through one schema-neutral representation, the IR
(`systemoneprotocol`): requests, batches, questions, answers, errors and
listings, with JSON kept in order. A listener's codec reads a client's request
into the IR; an upstream's codec writes the IR in the upstream's schema.

The Router's own hops inside a deployment, to the model runtime's `/v1/bundle`
and `/v1/decisions`, carry the IR in the **internal form**: the 4.0 request
forms plus `roles`, where an absent `roles` is 4.0's implicit roles and
`roles: null` is none. The internal codec loses nothing: a request it reads
writes back unchanged, which a round-trip test checks over the model families'
own self-test requests and a fuzz corpus. So the model input of every
Router-built request, and of every request a 4.0 client sends to a 4.0
listener, is byte for byte what it was.

A listener decides how each request goes on, its **plan**:

- **Pass through.** A request already in the internal form (every 4.0 request,
  and every 5.0 request on `/v1/systemone` of a 4.0 listener) is decoded and
  checked, then forwarded with the client's own bytes.
- **Translate.** Anything else (a 5.0 listener's requests, the batch endpoint,
  a request with the long-context header) is written from the IR in the
  internal form.

Both plans are counted in the listener diagnostics above.

## Upgrading

`vllm-sr config migrate` moves the removed listener fields into `apis`:
`models` becomes each LLM API's `models`, and the `systemone` block becomes
`apis.systemone` with `schema: vllm-sr/4.0`, which its clients speak. It gives
a systemone deployment or provider model of a `vllm-sr/...` model
`schema: vllm-sr/5.0`. A runtime of this release speaks `vllm-sr/5.0` on
`/v1/systemone`; start it with `vllm-srun serve --systemone-schema vllm-sr/4.0`
to keep serving 4.0 clients directly.

A hosted Decision API built on the Router or the Engine upgrades in two steps:

1. Upgrade every component, backends included, with `vllm-sr/5.0` between the
   Router or Engine and the runtimes. Keep the public listeners at
   `apis.systemone.schema: vllm-sr/4.0`, so clients see no change; new clients
   can already use 5.0 forms there. Move tools that call the runtimes directly
   to 5.0 forms in the same release. A public endpoint a runtime serves without
   a frontend needs the Engine in front of it, or the same listener schema on
   the runtime.
2. Later, and optionally, move in-house callers to standard or 5.0 forms, and
   switch a listener to `vllm-sr/5.0` once its legacy-form counter shows that
   external 4.0 traffic is gone.
