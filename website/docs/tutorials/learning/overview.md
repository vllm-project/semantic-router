# Router Learning

## Overview

Router Learning is the router layer for cross-request routing intelligence. It
adjusts the model proposed by semantic decisions without making online state
part of `decision.algorithm`.

The public concepts are:

- `global.router.learning.adaptation`: online model-choice learning.
- `global.router.learning.protection`: session and conversation stability.
- `routing.decisions[].adaptations`: per-decision apply, observe, or bypass
  controls.
- Router Replay: durable diagnostics and outcomes for offline recipe learning.

Use Router Learning when a decision should remain semantic, but repeated
requests should consider current model, tool-loop state, prefix-cache evidence,
handoff cost, switch history, or runtime outcomes.

## Key Advantages

- Keeps semantic decisions readable and request-local.
- Gives online model-choice learning and stability protection one shared
  runtime pipeline.
- Lets hard policy decisions bypass learning without changing route rules.
- Records compact response headers and detailed Router Replay diagnostics.
- Feeds offline agent loops that can find routing problems and propose recipe
  patches.

## What Problem Does It Solve?

Semantic decisions are good at matching the current request, but they do not
remember whether a model was overprovisioned, underpowered, unstable, or expensive in
similar agent flows. Router Learning adds bounded online state and replay-linked
outcomes so the router can improve model choice while keeping recipes in
control.

## When to Use

- Your recipe has multiple candidate models and runtime evidence should improve
  the choice.
- Agent sessions need stability across tool loops, prefix cache, or provider
  state.
- Sensitive decisions need an explicit bypass from online learning.
- You want replay and outcomes to power offline recipe experiments.

## Configuration

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: true
        strategy: routing_sampling
        candidate_set: decision
      protection:
        enabled: true
        scope: conversation
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
        tuning:
          idle_timeout_seconds: 300
          switch_margin: 0.05
          stability_weight: 1.0
```

Decision-local controls are sparse. Most decisions inherit global behavior:

```yaml
adaptations:
  mode: bypass
```

Use `bypass` for privacy, security, local-only, compliance, or any other hard
policy route. Use component-level controls when one component should observe or
bypass independently:

```yaml
adaptations:
  adaptation:
    mode: observe
  protection:
    mode: apply
    stability_weight: 1.5
```

## Runtime Flow

```text
base selector
  -> protection preflight
  -> adaptation
  -> protection switch guard
  -> final model
```

Adaptation answers which model looks better from experience. Protection answers
whether exploration or switching is safe now.

## Header And Replay

The `x-vsr-learning-*` header family is intentionally compact:

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=propose_switch,protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=sampled_win,protection=switch_allowed
```

Detailed fields such as base model, proposal model, final model, cache warmth,
switch cost, candidate scores, sampling values, and hashed identity diagnostics
belong in Router Replay, keyed by `x-vsr-replay-id`.

## Related Pages

- [Adaptation](./adaptations) explains `routing_sampling` and candidate sets.
- [Protection](./protection) explains conversation and session stability.
- [Decision Adaptations](./decision-adaptations) explains decision-local
  controls.
- [Memory And Replay](./memory-and-replay) explains diagnostics and outcomes.

## Offline Recipe Learning

Router Learning does not rewrite deployed recipes on the request path. Use the
offline recipe-learning command to turn replay and outcomes into findings,
metrics, candidate recipe variants, experiment estimates, recipe patch
suggestions, and experience seed packs:

```bash
vllm-sr eval recipe-learning \
  --endpoint http://localhost:8080 \
  --recipe-file config.yaml \
  --output-dir ./router-learning-report
```

For air-gapped or CI workflows, export replay JSON first and pass it with
`--replay-file`. Add `--cases-file` when eval cases include expected decisions
or models.
