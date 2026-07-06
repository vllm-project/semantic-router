# Router Learning: Self-Improving Model Routing

## Status

This is the system design proposal and implementation contract for Router
Learning in vLLM Semantic Router. It is intended to be the source of truth for
follow-up implementation work: public API, runtime modules, state contracts,
replay diagnostics, feedback ingestion, offline recipe learning, and validation.

Router Learning turns a static routing recipe into a controlled self-improving
router. The recipe still defines policy. Learning only improves choices inside
the policy boundary, protects agent continuity, and produces offline recipe
patches that can be reviewed and deployed normally.

The design has three product responsibilities:

- **adaptation**: online model-choice learning from runtime experience
- **protection**: online stability control for sessions, conversations, cache,
  tool loops, and handoff cost
- **recipe learning**: offline agent-driven eval and experiment loops that
  improve recipes

These are the public product concepts. Mechanism names such as model
experience, session affinity, switch gates, online sampling, replay-derived
seeds, or route statistics may exist internally, but they should not be
exposed as top-level API concepts in the first Router Learning surface.
Adaptation is the online model-choice loop. Protection is the stability and
continuity loop. Recipe learning is the offline loop that evaluates evidence and
proposes recipe edits.

This is a clean-break design. The public learning API is defined only by the
shapes in this document. Historical public config names such as
`session_aware`, `bandit`, `elo`, `personalization`, `rl_driven`,
`gmtrouter`, and `lookup_tables` should not remain valid learning API.
Aliases should not be added for these names. If a deployed recipe uses an old
learning block, validation should fail with a clear validation error.

The public API should stay small and operator-readable. Internal implementations
may use posterior scoring, guarded stochastic sampling, online experience,
identity-scoped state, EWMA telemetry, or replay-derived seed packs, but
those implementation details should not become the primary product vocabulary.

The first implementation target is intentionally narrow and should be completed
before adding more learning families:

- one online adaptation strategy: `routing_sampling`
- one protection runtime with conversation and session scopes
- one replay-linked outcome API
- one offline recipe learning loop that produces findings, metrics, candidate
  recipes, recipe patches, and optional experience seed-pack artifacts

This proposal is the implementation handoff. Follow-up work should implement
the loop above first, then open separate follow-up issues for additional
strategies, multi-replica state, seed-pack runtime import, promotion workflows,
or dashboard artifact UX. Do not start by moving old algorithm families into
the new API or by adding a broad learning configuration surface.

The first implementation owns the seed-pack artifact shape and export path.
Public seed-pack import, warmup, retention, or storage configuration is a
follow-up. Do not add a public runtime config field just to consume seed packs
in the first learning API.

Future algorithms can extend this design through the same component contracts,
but they should not make the first operator-facing API larger than necessary.
The first implementation should be judged by whether the full loop is usable,
not by how many learning algorithms are present. A small working loop with
typed evidence, clear diagnostics, and stable agent behavior is preferred over
a broad surface of partially integrated algorithms.

The final first-version API commitments are:

- `global.router.learning.enabled` turns the system on.
- `global.router.learning.adaptation.strategy` selects the online adaptation
  algorithm. The day-0 and default value is `routing_sampling`.
- `routing_sampling` is the default online learning strategy when learning and
  adaptation are enabled. It is not a nested block, not a legacy `bandit`
  surface, and not an optional compatibility mode.
- `global.router.learning.adaptation.candidate_set` selects the model search
  scope: `decision`, `tier`, or `global`.
- `global.router.learning.protection` owns identity, scope, and switch-stability
  tuning.
- `routing.decisions[].adaptations` is the only decision-level learning control
  surface.
- `routing.decisions[].adaptations.mode: bypass` prevents both adaptation and
  protection from changing that decision.
- Experience is automatic strategy-owned state. There is no public
  `experience.enabled`, `experience.source`, or `router.learning.memory`
  switch.
- The first API does not expose `goals`. Cost, latency, reliability, overuse,
  and cache pressure are fixed routing semantics in `routing_sampling`, not a
  public weighted-goals DSL.
- The first API does not expose per-decision learning weights. A decision tunes
  the adaptation/protection trade-off through `adaptation.candidate_set`,
  component `mode`, `protection.stability_weight`, and
  `protection.switch_margin`.

The final runtime contract is:

```text
match recipe decision
  -> honor decision learning bypass
  -> select base model from recipe policy
  -> protection preflight decides whether sampling is safe
  -> adaptation scores the allowed candidate set with routing_sampling
  -> protection decides hold, allow, or rescue
  -> emit compact learning headers
  -> write typed replay diagnostics
  -> accept typed outcomes
  -> let offline recipe learning propose reviewed recipe patches
```

This ordering is fixed. Adaptation and protection are independent components:
adaptation proposes a model from the allowed candidate set, and protection
decides whether sampling or switching is safe. They share typed pipeline facts;
they do not call into each other and they do not mutate recipe policy.

The first implementation is not a passive metrics-only system. When learning is
enabled, the router should be able to make bounded online model-choice
improvements from local experience, accept explicit feedback through outcomes,
and generate offline recipe patches from replay/eval evidence. The safety
boundary is that online learning may change only the model choice inside the
configured candidate set; recipe structure changes remain offline artifacts.

The phrase "self-improving" means two different but connected capabilities:

- online learning improves the current model choice from local, typed
  experience while respecting the recipe boundary
- offline recipe learning evaluates replay and outcomes, runs experiments, and
  proposes reviewable recipe patches

It does not mean request-time classifier mutation, hidden policy edits, or
automatic production recipe rewrites.

There is no separate public `algorithm`, `memory`, `experience`, `model_pool`,
or `goals` control plane in the first API. Those are either recipe policy,
internal state, or offline artifacts. The public control plane is deliberately
small:

| Scope | Public Control | Purpose |
| --- | --- | --- |
| Global | `global.router.learning.adaptation` | Choose the online model-choice strategy and default candidate boundary. |
| Global | `global.router.learning.protection` | Choose conversation/session stability behavior and identity headers. |
| Decision | `routing.decisions[].adaptations` | Allow, observe, or bypass learning for the matched decision. |
| Offline | recipe learning artifacts | Propose recipe patches and optional experience seed packs after eval. |

This is the boundary follow-up agents should implement. If an implementation
needs a new field, it should first prove the behavior cannot be expressed with
recipe policy, candidate scope, component mode, protection tuning, typed
outcomes, or offline recipe learning.

The first version also avoids product terms that sound like independent
features but are really implementation details. Do not expose
`model_experience`, `session_aware`, `model_pool`, `memory_backend`,
`experience.source`, or strategy-specific nested config in the public API.
Those concepts map to the narrower product surface:

| Internal Concern | Public Surface |
| --- | --- |
| model performance memory | `learning.adaptation.strategy: routing_sampling` |
| session or conversation stability | `learning.protection.scope` |
| model search space | `learning.adaptation.candidate_set` |
| hard learning boundary | `routing.decisions[].adaptations.mode: bypass` |
| offline classifier or recipe improvement | recipe learning artifacts |

The most important product constraint is that online learning changes only the
model selected for the current request, and only inside the configured
candidate boundary. Changes to signals, examples, thresholds, priorities,
tiers, and `modelRefs` are recipe changes and must be produced by offline
recipe learning as reviewable artifacts.

This document describes the target design and task boundary. Follow-up
implementation work should close the phases in this document and avoid adding
legacy shims or experimental public fields that are not listed here.

The implementation should optimize for one thing first: a router that can learn
from real routing evidence while remaining predictable enough for agent
sessions. More advanced learning methods are useful only after the day-0 loop
is typed, replay-explainable, testable, and easy for recipe authors to control.

## Goals

- Keep recipes as the explicit operator control plane.
- Let deployed recipes improve model choice from runtime experience.
- Keep agent sessions stable while still allowing useful model changes.
- Let users decide which decisions can be adjusted and which decisions must be
  bypassed.
- Provide one typed outcome path for users, agents, evals, providers, routers,
  and operators.
- Make replay the durable evidence log for audit, debugging, eval, and offline
  recipe learning.
- Make offline agent loops capable of finding routing problems, running
  experiments, and proposing recipe patches.
- Avoid request-time dependence on synchronous external storage reads.

## Non-Goals

- Do not silently rewrite deployed recipes on the request path.
- Do not hard-code privacy, security, local-only, or compliance concepts into
  learning. Users express those boundaries through decisions.
- Do not expose a broad `goals` weighted map in the first public API.
- Do not expose storage backends, experience internals, or replay
  implementation details as first-version learning config.
- Do not preserve old learning config names as aliases.
- Do not add request-path recipe mutation, automatic recipe promotion, or hidden
  edits to route classifiers in the first implementation.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Recipe policy | Static route policy: signals, decisions, priority, tier, modelRefs, and base selection algorithms. |
| Adaptation | Online model-choice learning that proposes a better model within an allowed candidate set. |
| Protection | Online stability control that decides whether exploration or switching is safe now. |
| `routing_sampling` | The day-0 adaptation strategy. It scores candidates from experience and can sample when protection allows. |
| Outcome | Typed feedback linked to replay, used by online experience and offline recipe learning. |
| Replay | Durable event log for route decisions, responses, outcomes, and diagnostics. |
| Experience | Hot-path, algorithm-owned state used by adaptation to score candidate models. |
| Experience seed pack | Offline artifact that captures cold-start quality evidence without becoming request-path policy. |
| Recipe learning | Offline agent loop that uses replay, outcomes, and evals to propose recipe changes. |
| Candidate set | The model search space that adaptation is allowed to consider for the matched request. |
| Proposal model | The model proposed by adaptation before protection decides whether it is safe to use. |
| Final model | The model actually sent to the backend after protection has allowed, held, or rescued a switch. |

## Design Principles

- Recipes remain the explicit policy layer. Learning can adjust choices allowed
  by the recipe, but it does not create hidden policy.
- The request path reads and writes local bounded state only. Durable storage is
  used for replay, outcomes, export, and offline learning.
- Adaptation and protection are independent components composed by the learning
  pipeline. They do not call each other directly.
- A matched decision is the policy boundary. If that decision bypasses learning,
  neither adaptation nor protection can change the route.
- Feedback enters through typed outcomes. Runtime telemetry can update latency,
  cache, cost, and reliability evidence, but HTTP success alone is not a model
  quality reward.
- Recipe edits happen offline through recipe learning. There are no hidden
  online edits to signals, examples, thresholds, priorities, tiers, or
  `modelRefs`.
- Public config should describe operator intent. Posterior parameters, feature
  vectors, experience record layout, storage layout, and implementation-specific
  score details belong in typed runtime code and replay diagnostics.

Learning should not become a second recipe language. If a concept is policy,
put it in the recipe. If it is runtime memory, keep it in typed state and
replay diagnostics. If it changes recipe policy, make it an offline recipe
learning artifact.

The implementation should be clean-break. A config block should either be part
of the API in this document or fail validation. There should be no silent
rewrites from old algorithm names into the new learning API.

## Learning Boundaries

Router Learning has two different loops. They share replay and outcomes, but
they are allowed to change different things.

| Loop | Runs | May Change | Must Not Change |
| --- | --- | --- | --- |
| Online adaptation | request path | `proposal_model` within the configured candidate set | signals, thresholds, decisions, priority, tier, or modelRefs |
| Online protection | request path | whether to sample, hold, allow, or rescue a proposed switch | model quality scores, decision matching, or recipe policy |
| Outcome ingestion | API path | bounded online experience for model-targeted outcomes | recipe policy or route classifier behavior |
| Offline recipe learning | explicit eval/agent command | candidate recipe patches and optional experience seed packs | production routing unless the patch is reviewed and deployed |

The matched decision is always the policy boundary. If that decision bypasses
learning, online adaptation and protection cannot change the route. If the
decision allows learning, online adaptation may search the configured candidate
set and protection may decide whether the resulting proposal is safe to use.

Hard boundaries are operator policy, not built-in learning categories. A recipe
that needs privacy, security, local-only, compliance, tenant, or data-residency
protection should express that through decision matching, allowed `modelRefs`,
and `adaptations.mode: bypass` when learning must not adjust the result.

This keeps the product contract simple:

```text
recipe policy decides what is allowed
online learning chooses or protects within that allowance
offline recipe learning proposes recipe edits
```

## Data Layers

Router Learning has three runtime data layers plus one offline artifact layer.
They are related, but they should not be exposed as one public configuration
object.

| Layer | Request Path | Purpose |
| --- | --- | --- |
| Replay | Write-only from the hot path | Durable evidence log for route choices, responses, learning diagnostics, and outcomes. |
| States | Local hot-path reads and writes | Mutable in-process state used by learning components during routing. |
| Experience | Part of states | Adaptation-owned model performance summaries keyed by decision, tier, and model. |
| Recipe artifacts | Offline only | Findings, candidate recipes, recipe patches, and optional experience seed packs. |

Replay is the source of truth for audit and offline learning. States are the
request-time cache of what the router has learned so far. Experience is a
specific state family owned by adaptation. Recipe artifacts are generated
offline and applied through normal recipe review or deployment flows.

The first implementation keeps request-time states in process. It must not make
synchronous external storage reads during routing. Durable stores are used for
replay and offline recipe learning, and may later be used to rebuild or warm
local states outside the request path.

The product-level model is:

```text
replay = durable evidence log
states = local request-path memory
experience = adaptation-owned model-choice state
recipe artifacts = offline learning output
```

This keeps latency predictable. A request can update replay and local state, but
it should not block on an external replay query or storage lookup before
choosing a model. If later deployments need multi-replica state, they should
add explicit rebuild, warmup, sticky-session, or shared-state semantics after
the single-router contract is stable.

## System Model

```text
request
  -> recipe policy
  -> base model selection
  -> protection preflight
  -> adaptation proposal
  -> protection switch decision
  -> final model
  -> response + compact learning headers
  -> replay record
  -> outcome ingestion
  -> online experience update
  -> offline recipe learning
```

The key separation is:

```text
adaptation changes model choice within the active recipe
recipe learning changes the recipe offline
```

The runtime should expose this as one narrow pipeline, not as separate request
hooks that can drift independently:

```text
policy boundary:
  matched decision + base selection

online learning boundary:
  protection preflight + adaptation proposal + protection switch guard

offline learning boundary:
  replay/outcomes/evals -> candidate recipe patch + optional seed pack
```

This boundary matters for follow-up implementation work:

- policy matching remains deterministic for a request before learning runs
- adaptation can choose from the configured candidate set but cannot rematch the
  request
- protection can hold, allow, or rescue the proposed model but cannot create a
  new policy exception
- offline recipe learning can propose policy edits, but the live router does
  not apply them implicitly

## Module Responsibilities

| Module | Hot Path | Responsibility |
| --- | --- | --- |
| Decision engine | Yes | Match request signals to a decision and candidate modelRefs. |
| Base selector | Yes | Produce the recipe-controlled base model. |
| Learning pipeline | Yes | Run protection and adaptation in a fixed order and return the final model. |
| Protection runtime | Yes | Maintain identity-scoped stability state and guard sampling/switching. |
| Adaptation runtime | Yes | Maintain experience and score candidate models with `routing_sampling`. |
| Outcome API | No | Accept typed feedback linked to replay. |
| Router Replay | No | Persist route, response, learning, and outcome events. |
| Offline recipe learner | No | Run evals and experiments, then propose findings, recipe patches, and experience seed packs. |

The online modules should be replaceable behind typed contracts. Adding a new
adaptation strategy should not require changing protection, headers, replay
storage, or outcome ingestion. Adding a new protection policy should not require
rewriting adaptation scoring. The learning pipeline is the only component that
knows the composition order.

Request-time routing must use in-process online state. Durable storage is used
for replay and offline work, not synchronous request-path lookups.

## Public API

The public API has two levels:

- global learning components under `global.router.learning`
- per-decision controls under `routing.decisions[].adaptations`

There are only two request-path learning components in the first API:

- `adaptation`: choose a better model candidate inside the configured boundary.
- `protection`: decide whether exploration or switching is safe for the active
  session or conversation.

There is no separate public `memory`, `experience`, `bandit`, `session_aware`,
`model_pool`, or `goals` object. Those concepts are either implementation state,
strategy internals, recipe policy, or future proposals.

The shape is:

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
          min_turns_before_switch: 1
          switch_margin: 0.05
          stability_weight: 1.0

routing:
  decisions:
    - id: example
      adaptations:
        mode: apply
        adaptation:
          mode: apply
          candidate_set: tier
        protection:
          mode: apply
          stability_weight: 1.0
          switch_margin: 0.05
```

The operator-facing reading is intentionally short:

| Question | Field |
| --- | --- |
| Should the router learn on the request path? | `global.router.learning.enabled` |
| How should online model choice learn? | `learning.adaptation.strategy` |
| How far may adaptation search for models? | `learning.adaptation.candidate_set` |
| How long should routing stability be protected? | `learning.protection.scope` |
| Should a matched decision allow, observe, or bypass learning? | `routing.decisions[].adaptations.mode` |

Most recipes should not need the per-decision block. It exists for bypassing
sensitive decisions, observing learning without changing routes, or tuning the
adaptation/protection trade-off for a specific route. Global learning config is
the normal path; decision-level config is an override for routes whose risk,
candidate scope, or stability requirements differ from the recipe default.

Decision-level controls are not a second global learning configuration. They
are matched-decision controls. The router must first match a decision using
recipe policy, then apply only that matched decision's `adaptations` block.
Other decisions in the recipe do not contribute learning policy, even when
`candidate_set: tier` lets adaptation search their `modelRefs`.

Mode inheritance must be explicit and conservative:

1. `adaptations.mode` is the decision-level learning permission.
2. If omitted, the effective decision mode is `apply`.
3. If set to `bypass`, both adaptation and protection are bypassed for the
   matched decision; component-level overrides must not make either component
   active.
4. If set to `observe`, both components are capped at observe-only behavior;
   component-level overrides must not change the final route.
5. If set to `apply`, component-level modes may independently apply, observe,
   or bypass their component.
6. If no component-level mode is set, the component inherits the effective
   decision mode.

This makes the top-level decision mode easy to reason about: `bypass` means no
learning adjustment, `observe` means diagnostics only, and `apply` means the
component blocks may tune behavior. A recipe that needs one component to apply
and another to observe should leave `adaptations.mode` omitted or set to
`apply`, then configure the component modes directly.

The first public API deliberately does not expose:

- `global.router.learning.adaptations`
- `router.learning.memory`
- `experience.enabled`
- `experience.source`
- `goals`
- `model_pool`
- `model_experience`
- per-evidence score weights such as cache, handoff, latency, or reliability
  weights
- nested strategy blocks such as `routing_sampling:`
- old algorithm-family blocks such as `session_aware`, `bandit`, `elo`,
  `personalization`, `rl_driven`, `gmtrouter`, or `lookup_tables`

Adaptation and protection are the product concepts. Strategy names and state
shape are implementation details unless they affect operator behavior. Experience
is created and maintained by the active adaptation strategy; users do not enable
or configure it separately.

### Defaults and Validation

The first-version API should be strict:

| Field | Default | Validation |
| --- | --- | --- |
| `global.router.learning.enabled` | `false` | If `false` or omitted, learning does not adjust routes. |
| `learning.adaptation.enabled` | enabled when `learning.enabled: true` | Explicit `false` disables adaptation only. |
| `learning.adaptation.strategy` | `routing_sampling` | Unknown strategies are rejected. |
| `learning.adaptation.candidate_set` | `decision` | Must be `decision`, `tier`, or `global`. |
| `learning.protection.enabled` | enabled when `learning.enabled: true` | Explicit `false` disables protection only. |
| `learning.protection.scope` | `conversation` | Must be `conversation` or `session`. |
| `learning.protection.identity.headers.session` | `x-session-id` | Empty values are rejected when the header block is present. |
| `learning.protection.identity.headers.conversation` | `x-conversation-id` | Empty values are rejected when the header block is present. |
| `learning.protection.tuning.idle_timeout_seconds` | `300` | Must be a non-negative integer. |
| `learning.protection.tuning.min_turns_before_switch` | `1` | Must be a non-negative integer. |
| `learning.protection.tuning.stability_weight` | `1.0` | Must be non-negative. Higher values favor stability. |
| `learning.protection.tuning.switch_margin` | `0.05` | Must be non-negative. |
| `decision.adaptations.mode` | `apply` | Must be `apply`, `observe`, or `bypass`. |
| `decision.adaptations.adaptation.mode` | inherit `decision.adaptations.mode` | Must be `apply`, `observe`, or `bypass`; cannot broaden a decision-level `observe` or `bypass`. |
| `decision.adaptations.adaptation.candidate_set` | global candidate set | Must be `decision`, `tier`, or `global`. |
| `decision.adaptations.protection.mode` | inherit `decision.adaptations.mode` | Must be `apply`, `observe`, or `bypass`; cannot broaden a decision-level `observe` or `bypass`. |
| `decision.adaptations.protection.stability_weight` | global protection stability weight | Must be non-negative. |
| `decision.adaptations.protection.switch_margin` | global switch margin | Must be non-negative. |

Validation should reject unknown old learning families and misplaced fields. In
particular, `candidate_set` is an adaptation control and must not be accepted
under `adaptations.protection`. `strategy` is global-only in the first API and
must not be accepted under a decision-level adaptation block.

### Reference Configuration Examples

These examples are the intended operator-facing shapes. They are not migration
examples and they should not be expanded with legacy aliases.

Minimal Router Learning:

```yaml
global:
  router:
    learning:
      enabled: true
```

This enables `routing_sampling` adaptation and conversation-scoped protection
with default identity headers.

Conversation-level protection with an explicit candidate boundary:

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        strategy: routing_sampling
        candidate_set: decision
      protection:
        scope: conversation
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
```

Session-level protection:

```yaml
global:
  router:
    learning:
      enabled: true
      protection:
        scope: session
```

With `scope: session`, the first final model after a new or idle-reset session
becomes the protected baseline for that session. New conversations inside the
same session do not automatically re-route unless the switch guard or bypass
rules allow it.

Sensitive decision bypass:

```yaml
routing:
  decisions:
    - id: private_local
      tier: 2
      modelRefs:
        - local-secure-model
      adaptations:
        mode: bypass
```

This prevents both adaptation and protection from changing the matched
decision. The decision's own recipe policy remains the only route source.

Tier-level adaptation for a specific decision:

```yaml
routing:
  decisions:
    - id: code_reasoning
      tier: 3
      modelRefs:
        - balanced-code-model
      adaptations:
        adaptation:
          mode: apply
          candidate_set: tier
        protection:
          mode: apply
```

This allows adaptation to search the union of `modelRefs` from decisions with
the same `tier` while still using the matched decision's protection controls.

Observe-only rollout:

```yaml
routing:
  decisions:
    - id: high_risk_route
      adaptations:
        mode: observe
```

This records learning diagnostics without changing the final route. It is the
recommended rollout mode for routes where operators want replay evidence before
allowing online adjustment.

### Global Learning

`global.router.learning.enabled: true` enables the learning system. When enabled
without component overrides, both adaptation and protection are enabled with
conservative defaults:

- `adaptation.strategy: routing_sampling`
- `adaptation.candidate_set: decision`
- `protection.scope: conversation`
- protection identity headers default to `x-session-id` and `x-conversation-id`

In other words, `routing_sampling` is the default online learning strategy once
Router Learning is enabled. The router does not learn by default when
`global.router.learning.enabled` is omitted or `false`.

Enablement uses clean tri-state semantics:

| Config Shape | Runtime Meaning |
| --- | --- |
| `learning.enabled` omitted or `false` | Learning components do not adjust routing. |
| `learning.enabled: true` with no component blocks | Adaptation and protection are both enabled with defaults. |
| `learning.enabled: true` and `adaptation.enabled: false` | Adaptation is disabled; protection can still guard base-route switches. |
| `learning.enabled: true` and `protection.enabled: false` | Protection is disabled; adaptation can still score candidates, but session/cache stability guards are not active. |

The minimal valid learning config is therefore:

```yaml
global:
  router:
    learning:
      enabled: true
```

Recipes should usually set these values explicitly for readability:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres

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
          min_turns_before_switch: 1
          switch_margin: 0.05
          stability_weight: 1.0
```

`global.services.router_replay` remains the durable event-log configuration.
Learning attaches diagnostics and outcomes to replay. It does not define a
second replay backend.

There is no public `router.learning.memory` block in this design. Replay,
states, and experience are implementation contracts under Router Learning, not
separate operator-managed services in the first API.

### Component Independence

Adaptation and protection are independent components with a fixed composition
order. Either component can be disabled globally or bypassed per decision.

The router should treat the model entering the switch guard as the
`proposal_model`:

- if adaptation applies, `proposal_model` is the selected candidate
- if adaptation observes only, `proposal_model` remains the base model
- if adaptation is disabled, `proposal_model` is the base model

Protection compares the protected model with the `proposal_model`. This means
protection can still provide session or conversation stability even when
adaptation is disabled. Conversely, when protection is disabled, adaptation can
still select a model within its candidate set, but the router provides no
session, conversation, cache, tool-loop, or handoff-cost protection beyond
adaptation's static candidate, cost, and reliability limits.

The components communicate only through typed pipeline inputs and outputs:

```text
base selection -> protection preflight -> adaptation decision -> protection decision
```

Protection does not call adaptation, and adaptation does not call protection.
The learning pipeline owns ordering and passes bounded facts between them.

Component modes have fixed semantics:

| Decision/Component Mode | Adaptation Behavior | Protection Behavior | Final Route Effect |
| --- | --- | --- | --- |
| inherited `apply` | may score candidates and propose a model | may suppress sampling, hold, allow, or rescue | learning may change the final model |
| decision `observe` | computes diagnostics but cannot change the proposal | computes diagnostics but cannot hold, switch, or rescue | recipe base selection is final |
| adaptation `observe` | computes diagnostics but returns the base model as proposal | still applies unless protection is also observe/bypass | adaptation alone cannot change the model |
| protection `observe` | adaptation may propose normally | computes diagnostics but cannot hold, switch, or rescue | final model is whatever entered protection |
| component `bypass` | that component does not adjust this decision | that component does not adjust this decision | only the non-bypassed component can affect routing |
| decision `adaptations.mode: bypass` | disabled for the matched decision | disabled for the matched decision | recipe base selection is final |

This matrix is part of the implementation contract. In particular,
`protection.mode: observe` must not act as a hidden switch guard. It may record
what it would have done, but it cannot change the model selected by base
routing or adaptation.

### Adaptation Config

Adaptation has one public strategy on day 0:

```yaml
adaptation:
  enabled: true
  strategy: routing_sampling
  candidate_set: decision
```

`routing_sampling` is the default online learning strategy. It reads experience,
scores candidates, and may draw bounded random samples when protection is not
participating or when protection explicitly allows exploration. It is enabled by
default when adaptation is enabled.

Sampling is part of the day-0 strategy, not a separate public toggle. Operators
enable or constrain it through the existing public controls:

- `learning.adaptation.enabled` turns adaptation on or off.
- `learning.adaptation.candidate_set` limits where sampling can search.
- `learning.protection.scope` and protection tuning decide when sampling is
  safe.
- decision `adaptations.mode`, `adaptation.mode`, and `protection.mode` can
  apply, observe, or bypass the behavior for the matched decision.

The default behavior under `learning.enabled: true` is therefore:

```text
routing_sampling scores candidates
  -> protection may suppress stochastic sampling for this request
  -> deterministic scoring still runs when sampling is suppressed
```

`strategy` is the field that selects the online adaptation algorithm.
`routing_sampling` is a strategy value, not a separate nested component and not
a second enable switch. The first API deliberately avoids exposing algorithm
internals such as experience seed weights, reward weights, feature vectors, or
posterior parameters.

There should be no public block like this:

```yaml
adaptation:
  routing_sampling:
    enabled: true
```

The clean shape is:

```yaml
adaptation:
  enabled: true
  strategy: routing_sampling
```

`strategy` may be omitted only when the default behavior is acceptable. The
effective strategy is still `routing_sampling`; implementations should expose
that effective value in validation, replay, and debug output so operators do
not have to infer which algorithm ran.

Future online strategies can be added as new `strategy` values only when they
represent distinct operator-facing behavior. They must reuse the same outcome,
experience, protection, header, and replay contracts.

The strategy is configured globally. A decision can override candidate scope,
or bypass/observe the component, but it should not redefine the adaptation
algorithm inline.

### Candidate Sets

`candidate_set` controls where adaptation may search for models. The global
value is the recipe default. A decision may override it with
`routing.decisions[].adaptations.adaptation.candidate_set` when a specific route
needs a narrower or broader search space.

| Value | Candidate Models |
| --- | --- |
| `decision` | Models from the matched decision's `modelRefs`. |
| `tier` | Union of `modelRefs` from all decisions with the same `decision.tier` as the matched decision. |
| `global` | All deployed models in the recipe's model/provider inventory. |

`tier` uses the existing decision `tier` field. It is a route tier, not a model
quality label. If `candidate_set: tier` is configured and the matched decision
has no positive tier, the router falls back to `decision`.

`global` is the broadest mode. It can propose any deployed model that passes
static eligibility, including a model that does not appear in any decision's
`modelRefs`. It should use stricter cost and reliability guards because it can
propose a model outside the matched decision's local candidate set.

`candidate_set` is not named `model_pool` because it does not create a new pool
object. It is an adaptation search boundary derived from the recipe:
`decision` from the matched decision, `tier` from decisions with the same tier,
and `global` from deployed model/provider inventory.

Candidate set construction must be deterministic:

- de-duplicate models while preserving the recipe's stable decision/model order
- include only deployed models with configured provider/backend metadata
- apply model health, capability, provider, and static cost eligibility before
  scoring
- never use another decision's `adaptations` block as policy for the matched
  request
- respect the matched decision's `adaptations.mode: bypass` before candidate
  construction
- apply the matched decision's component modes before scoring or switching

`candidate_set: global` and `candidate_set: tier` expand where adaptation may
search. They do not turn learning into an unrestricted model router and they do
not bypass the matched decision's policy boundary. `global` means the operator
has explicitly allowed adaptation to search the deployed model inventory for
that request; it does not allow the router to use undeployed models, rematch the
request to another decision, or ignore a matched decision's `adaptations.mode:
bypass`.

Candidate scope is the main day-0 operator control for how much adaptation can
move:

| Scope | Best For | Risk |
| --- | --- | --- |
| `decision` | Stable production recipes, sensitive routes, and first rollout. | May miss a better model that appears in a nearby route. |
| `tier` | Routes that share a task class or difficulty tier. | Requires route tiers to be meaningful. |
| `global` | Broad model optimization experiments and agents that can tolerate wider model search. | Can cross route boundaries, so protection and static eligibility must be stricter. |

Decision-level `adaptations.mode: bypass` still wins for all three scopes.

### Protection Config

Protection controls exploration and switching:

```yaml
protection:
  enabled: true
  scope: conversation
  identity:
    headers:
      session: x-session-id
      conversation: x-conversation-id
  tuning:
    idle_timeout_seconds: 300
    min_turns_before_switch: 1
    switch_margin: 0.05
    stability_weight: 1.0
```

Protection supports two scopes:

| Scope | Meaning |
| --- | --- |
| `conversation` | Strong protection within one conversation id. A new conversation can re-route while still using session history as soft evidence. |
| `session` | Strong protection across the full session. The first final model after idle release becomes the protected baseline until the session idles again or a matched decision bypasses learning. |

Identity names are configured under protection because they are used by the
protection state machine. Future components can add their own identity needs
without changing the global header names.

If the configured identity header for the active scope is missing, protection
must fail open for cross-request stability: it should not error the request, it
should not create a synthetic long-lived identity, and it should record a
bounded diagnostic reason such as `missing_identity`. Request-local static
guards can still apply.

The two scopes are not different products. They are the same protection runtime
with different identity boundaries:

| Scope | Protected Epoch Starts | Strongly Protected Until | Typical Use |
| --- | --- | --- | --- |
| `conversation` | first final model for the conversation id | conversation idle timeout or bypassed decision | multi-run agent sessions where each run can re-evaluate routing |
| `session` | first final model for the session id | session idle timeout or bypassed decision | assistants that should keep one model across many runs |

Both scopes can still allow deterministic rescue switches when the current
model appears underpowered. The difference is how long the protected baseline
is remembered.

The first public API exposes one stability trade-off knob:
`stability_weight`. Cache preservation, tool/protocol continuity, handoff cost,
switch history, and provider reliability are internal switch-cost evidence.
They should be recorded in replay diagnostics, but they should not become
separate public weights until real deployments prove that one combined
stability knob is insufficient.

The first request in a protected scope should not be reported as an "initial
route" special case. It should be reported as protection establishing a
baseline:

```text
method=protection
action=establish_baseline
scope=conversation|session
reason=new_conversation|new_session|idle_reset
```

This keeps user-facing diagnostics consistent. Every route should be explained
with the same shape: component method, action, scope, and reason.

### Decision Controls

Decision controls live directly under `routing.decisions[].adaptations`.
Learning is globally enabled, but each decision can control whether learning may
adjust it.

Most decisions can omit this block and inherit global behavior.

To bypass all learning adjustments for a sensitive decision:

```yaml
adaptations:
  mode: bypass
```

This is the policy boundary mechanism for privacy, security, local-only,
compliance, or any other operator-defined sensitive route. The router does not
hard-code which decisions are sensitive.

To control adaptation and protection separately:

```yaml
adaptations:
  adaptation:
    mode: apply
    candidate_set: tier
  protection:
    mode: observe
```

Allowed modes:

| Mode | Meaning |
| --- | --- |
| `apply` | The component may affect the final route. |
| `observe` | The component runs and records diagnostics but cannot change the final route. |
| `bypass` | The component does not adjust this decision. |

`adaptations.mode: bypass` has precedence over component-level modes. Component
modes are otherwise independent.

Decision controls are intentionally small in the first API:

- `mode: bypass` blocks all learning adjustment for the matched decision
- `adaptation.mode` controls whether online model-choice learning can apply
- `adaptation.candidate_set` optionally overrides the global candidate set
- `protection.mode` controls whether stability protection can apply
- `protection.stability_weight` and `protection.switch_margin` optionally tune the
  adaptation/protection trade-off for this decision

Do not add per-decision algorithm internals unless they solve a concrete recipe
authoring problem that cannot be expressed by these controls. Per-decision
candidate scope is allowed because it is policy surface, not algorithm internals.

Decision-level protection tuning is the only first-version trade-off knob
between adaptation and protection:

```yaml
adaptations:
  protection:
    stability_weight: 1.5
    switch_margin: 0.05
```

Higher `protection.stability_weight` favors continuity and cache preservation.
Lower `protection.stability_weight` makes it easier for adaptation to switch
models. `switch_margin` is the minimum advantage required before any switch for
this decision.

This tuning is intentionally small. It adjusts how strongly protection pushes
back on an adaptation proposal for this decision. It does not configure the
adaptation algorithm, outcome schema, replay format, or storage behavior.

The global protection tuning remains the default. Decision-level protection
tuning only changes the local trade-off for the matched decision. It should be
omitted unless a route has a clear need to favor stability or adaptation more
strongly than the recipe default.

## Runtime Flow

The ordering is fixed:

```text
base selector
  -> protection preflight
  -> adaptation
  -> protection switch guard
  -> final model
```

1. The decision engine chooses a matched decision.
2. The base selector chooses the base model from the recipe.
3. Protection preflight decides whether stochastic sampling is allowed now.
4. Adaptation builds a candidate set and may produce a `proposal_model`.
5. Protection switch guard decides whether to hold, allow, or rescue the
   `proposal_model`.
6. The router sends the request to the final model.
7. Compact headers summarize active methods, actions, scopes, and reasons.
8. Replay records base, proposal, final, decision, tier, and diagnostics.
9. Outcomes update experience and feed offline recipe learning.

Protection is intentionally split into two guard points:

- **preflight**: suppress or allow random exploration before adaptation scores.
- **switch guard**: allow, hold, or rescue the `proposal_model` after
  scoring.

Adaptation answers "which model looks better from experience?" Protection
answers "is it safe and worthwhile to explore or switch at this point in the
agent flow?" If adaptation is disabled or observing only, protection still
answers that question for the base selector's proposed model.

## Adaptation

Adaptation improves model selection within the configured candidate set while
the recipe remains deployed.

The day-0 strategy is `routing_sampling`. It combines:

- offline or default quality seeds
- `good_fit`, `underpowered`, `overprovisioned`, and `failed` outcomes
- latency EWMA
- cache hit/write EWMA
- effective input-cost evidence
- reliability evidence
- bounded stochastic sampling when protection permits exploration

Adaptation does not rewrite signals, thresholds, decisions, priorities, tiers,
or modelRefs. Those are offline recipe learning outputs.

`routing_sampling` is not exposed as a generic "bandit" block. It is the first
operator-facing adaptation strategy. Internally it can use posterior scoring and
bounded stochastic samples, but the public contract is simpler: score eligible
candidate models from experience, optionally sample when protection allows it,
and send one `proposal_model` to the switch guard.

The strategy is intentionally not a full high-dimensional contextual learner in
the first implementation. It does not learn an embedding classifier, train a
request feature model on the hot path, or mutate decision matching. It learns
model choice experience for the matched decision/tier/model context and uses
offline recipe learning for classifier-like recipe changes.

### Cold Start and Local Posterior

`routing_sampling` should combine two kinds of evidence:

- a cold-start quality seed from model metadata, offline evals, or imported
  seed packs
- local posterior evidence from outcomes and runtime telemetry observed after
  deployment

The cold-start seed prevents the router from treating every model as identical
on day 0. Local posterior evidence lets the router adapt when the deployed
traffic differs from the offline benchmark or eval set. The online request path
must not train a classifier, mutate route rules, or rewrite recipe policy from
that evidence.

The initial posterior should be conservative:

```text
prior quality -> local good_fit/underpowered outcomes -> bounded posterior score
```

Telemetry adjusts cost, latency, cache, and reliability. It should not be
mistaken for task quality unless it arrives as a typed model outcome.

This shape gives Router Learning a stable first online algorithm without
turning the recipe into a hidden model-ranking program.

### Evidence Channels

Adaptation should learn from bounded evidence channels:

| Channel | Source | Online Use |
| --- | --- | --- |
| Model outcomes | `POST /v1/router/outcomes` with `target: model` | Update model fit, overuse, or failure evidence. |
| Router telemetry | response status, latency, cache, token, retry, and provider evidence | Update reliability, latency, cache, and effective cost summaries. |
| Experience seed packs | recipe learning output | Export cold-start quality estimates and support offline experiments. Runtime import is a follow-up unless it can be done without public config. |

Route, policy, and stability outcomes are preserved for replay and offline
recipe learning. They should not directly mutate model-fit counters, because a
wrong decision is not the same failure as a weak model.

Feedback producers should normalize their local language into the router's typed
outcome schema. A chat client, eval harness, agent, provider adapter, or
operator tool can use its own UI labels, but the router receives bounded
sources, targets, verdicts, reasons, and scores.

| Source | Typical Producer | Primary Use |
| --- | --- | --- |
| `user` | End-user feedback or product UI | Preserve product feedback and optional model fit evidence. |
| `agent` | Agent harness after a run, tool loop, or verifier step | Explain underpowered models, overuse, failed tool progress, or stability problems. |
| `eval` | Offline route/model/cost eval | Compare recipes and create candidate patches. |
| `provider` | Model gateway or backend adapter | Reliability, retry, status, and latency evidence. |
| `router` | Router runtime | Cache, token, switch, fallback, and diagnostic evidence. |
| `operator` | Manual review or SRE workflow | Annotate policy misses, local-only boundaries, or incident findings. |

Only model-targeted outcomes can update online model-fit experience. Route,
policy, stability, provider, and router evidence is still valuable, but it is
used for diagnostics, protection, reliability, and offline recipe learning
unless a later strategy defines a typed online consumer for it.

### Reward Semantics

The first online reward model is deliberately small and typed:

| Evidence | Experience Update |
| --- | --- |
| `target: model`, `verdict: good_fit` | Increase model fit for the matched decision/tier/model context. |
| `target: model`, `verdict: underpowered` | Decrease model fit because the model lacked capability for the task. |
| `target: model`, `verdict: overprovisioned` | Increase overuse or cost pressure without reducing model quality. |
| `target: model`, `verdict: failed` | Increase reliability penalty for the model/provider context. |
| Router latency telemetry | Update latency EWMA only. |
| Cache hit/write telemetry | Update cache benefit or cache pressure only. |
| Token and effective cost telemetry | Update effective cost evidence only. |
| Provider errors and retries | Update reliability evidence only. |
| Route, policy, or stability outcomes | Preserve for replay and offline recipe learning; do not directly update model quality. |

This avoids conflating three different problems:

- the model was weak for the task
- the route or decision was wrong
- the model worked, but was more expensive or disruptive than necessary

Outcome labels are router API labels, not UI labels. A chat UI, eval harness, or
agent can present any product language it wants, but it should normalize that
feedback into these bounded verdicts before sending it to the router.

### Experience Key

Experience is keyed by matched decision, decision tier, and model:

```text
decision_id + decision_tier + model
```

Fallback order:

```text
decision_id + decision_tier + model
  -> decision_tier + model
  -> model
```

When `candidate_set: tier` or `candidate_set: global` proposes a model outside
the matched decision's `modelRefs`, the primary key still uses the matched
decision as context, then falls back to tier-level and model-level evidence.

### Experience State

Each experience record should be typed.

| Field | Meaning |
| --- | --- |
| `quality_seed` | Offline or imported quality estimate, defaulting to neutral. |
| `seed_weight` | Pseudo-count strength for the quality seed. |
| `good_fit_count` | Outcomes where the model fit the task. |
| `underpowered_count` | Outcomes where the model lacked capability for the task. |
| `overprovisioned_count` | Outcomes where the model worked but was unnecessarily expensive or strong. |
| `failed_count` | Provider or execution failures. |
| `latency_ewma` | Observed latency score. |
| `cache_hit_ewma` | Observed cache reuse. |
| `cache_write_ewma` | Observed cache write pressure. |
| `input_cost_multiplier_ewma` | Effective input cost multiplier. |
| `last_updated` | Freshness and decay input. |

`routing_sampling` needs experience for four concrete runtime decisions:

| Runtime Need | Experience Used |
| --- | --- |
| Cold start | `quality_seed` and `seed_weight` prevent all candidates from starting as identical. |
| Model fit | `good_fit_count` and `underpowered_count` estimate whether the model is capable for this decision/tier/model context. |
| Model overuse | `overprovisioned_count`, token/cost evidence, and latency evidence penalize unnecessarily strong or slow models without treating them as low quality. |
| Safety and reliability | `failed_count`, provider errors, cache evidence, and freshness prevent unhealthy or disruptive candidates from winning cheaply. |

Those are the day-0 state requirements. Do not add generic experience fields
unless a strategy can explain which runtime decision consumes them, how they
are updated, and how replay will make their effect visible.

The first strategy consumes experience as follows:

| Evidence | Runtime Use |
| --- | --- |
| `quality_seed`, `seed_weight` | Cold-start posterior before enough local outcomes exist. |
| `good_fit_count`, `underpowered_count` | Posterior model-fit estimate for the current decision/tier/model context. |
| `overprovisioned_count` | Overuse and cost penalty when a cheaper or smaller model is likely sufficient. |
| `failed_count` | Reliability penalty and guardrail against sampling into unhealthy models. |
| `latency_ewma` | Latency adjustment in the candidate score. |
| `cache_hit_ewma`, `cache_write_ewma` | Cache benefit or cache pressure in candidate scoring and switch-cost evidence. |
| `input_cost_multiplier_ewma` | Effective cost penalty, including prefix-cache misses and provider-side cost multipliers. |
| `last_updated` | Decay and freshness handling for stale experience. |

HTTP success can update reliability and latency evidence. It is not a quality
reward by itself.

There is no public `experience.enabled` switch in the first API. If adaptation
is enabled, experience is part of the strategy implementation. If
adaptation is disabled or bypassed for a decision, experience may still be
observed for diagnostics, but it cannot change the route.

Experience lifecycle is automatic:

| Adaptation State | Experience Behavior |
| --- | --- |
| globally disabled | no route adjustment; replay can still record telemetry. |
| globally enabled, decision applies | read and update local experience for scoring. |
| decision `observe` | compute and record diagnostics, but final route is unchanged. |
| decision `bypass` | do not let adaptation use experience to adjust the route. |

Users configure adaptation, not experience storage internals. The active
strategy owns which experience fields it needs.

Experience is not a replacement for replay. Replay is the durable evidence log.
Experience is the local hot-path summary that adaptation can read without an
external storage call. Offline recipe learning can rebuild, inspect, or seed
experience from replay, but the request path must treat experience as local
strategy state.

Online experience should update from bounded evidence only:

- router telemetry can update latency, cache, cost, and provider failure fields
- model-targeted outcomes can update fit, overprovisioning, and failure fields
- route, policy, and stability outcomes are preserved for offline recipe
  learning and should not directly mutate model-fit counters
- offline recipe learning can export experience seed packs; a future controlled
  warmup path may import them by setting quality seeds and seed weights

### Routing Sampling

For each request, `routing_sampling` runs these steps:

1. Build the candidate set from `decision`, `tier`, or `global`.
2. Apply static eligibility filters.
3. Load local experience with fallback from decision/tier/model to tier/model
   to model.
4. Compute a posterior quality estimate for each candidate.
5. Apply cost, overuse, reliability, latency, and cache adjustments.
6. Use posterior means when protection explicitly suppresses sampling.
7. Draw bounded posterior samples when protection is absent or allows
   exploration.
8. Select a winner and emit either the base model or a `proposal_model`.
9. Send the proposal to protection for the final switch decision.

For each candidate model, adaptation computes the posterior quality estimate:

```text
alpha = seed_weight * quality_seed + good_fit_count + smoothing
beta  = seed_weight * (1 - quality_seed) + underpowered_count + smoothing
mean  = alpha / (alpha + beta)
```

When protection is absent or allows exploration:

```text
predicted_quality = sample_beta(alpha, beta)
```

When protection suppresses exploration:

```text
predicted_quality = mean
```

The final candidate score uses fixed router semantics rather than a public
weighted-goals map:

```text
score =
  predicted_quality
  - cost_penalty
  - overuse_penalty
  - reliability_penalty
  + latency_bonus
  + cache_bonus
```

`overprovisioned_count` should not reduce quality. It increases the cost or
overuse penalty. `failed_count` affects reliability, not task quality.

Sampling is part of the online strategy, but it is not uncontrolled request-time
randomness. A sampled decision must be bounded by protection, static
eligibility, cost and reliability guards, and replay diagnostics. Tests should
be able to inject or replay randomness so a route can be explained after the
fact.

The score formula is an implementation contract, not a public tuning surface in
the first API. Operators should tune recipe policy, candidate set, component
modes, and protection tuning before asking for new score knobs.

There is no public `goals` weighted map in the first version. The route author
does not have to tune quality/cost/latency weights to make the router usable.
`routing_sampling` owns the fixed day-0 scoring semantics, and offline recipe
learning reports whether those semantics are producing useful cost, quality,
latency, cache, and switch outcomes. If future deployments prove that
operator-tuned objectives are necessary, they should be introduced through a
separate design with eval evidence, not as a default field in this API.

The strategy should use a small typed action set in replay and compact headers:

| Action | Meaning |
| --- | --- |
| `keep_base` | The base model remained the best candidate. |
| `propose_switch` | Adaptation proposed a different `proposal_model` for the switch guard. |
| `observe` | Adaptation ran diagnostics but could not change the route. |
| `bypass` | The matched decision disabled adaptation. |

Recommended adaptation reasons:

| Reason | Meaning |
| --- | --- |
| `cold_start` | Experience was mostly seed/default evidence. |
| `posterior_win` | Posterior score selected the proposal. |
| `sampled_win` | A bounded posterior sample selected the proposal. |
| `base_best` | The base model remained best after scoring. |
| `candidate_ineligible` | Static eligibility removed one or more candidates. |
| `observe_only` | Adaptation was in observe mode. |
| `decision_bypass` | The matched decision bypassed learning. |

Sampling is bounded by guardrails:

- decision `adaptations.mode: bypass` blocks adaptation changes
- adaptation `observe` simulates and records but cannot change the final model
- protection preflight can suppress sampling for this request
- tool/protocol continuation steps suppress sampling by default
- routine healthy traffic suppresses sampling when there is no capability
  pressure
- expensive candidates need a larger gain before they can win
- unreliable candidates cannot win through sampling after enough failure
  evidence exists
- `global` candidate sets use stricter margins than `tier`, and `tier` uses
  stricter margins than `decision`

Replay diagnostics should record whether the winner used posterior mean or a
sampled value, plus enough bounded metadata to explain the route.

The compact header may expose only the most user-relevant adaptation action for
the request. Replay should preserve the fuller trace, including whether
sampling was allowed, whether a sample was used, candidate scores, and the
reason the proposal did or did not differ from the base model.

`routing_sampling` is the first online strategy, and it is intentionally
smaller than a full contextual online learner. It learns per decision, tier, and
model from outcomes and telemetry. It does not yet fit or update a
high-dimensional request feature model on the hot path.

This is a deliberate stability choice for agent routing. Day-0 online learning
should improve choices from local experience and bounded sampling, while recipe
structure and classifier-like behavior continue to improve through offline
recipe learning.

### Routing Sampling Contract

`routing_sampling` must have a narrow typed contract:

| Contract | Requirement |
| --- | --- |
| Input | base model, matched decision id, decision tier, candidate set, static model metadata, local experience, protection preflight decision, request evidence. |
| Output | one typed adaptation decision: action, candidate set, base model, proposal model, score summary, sampling metadata, and reason. |
| State | local experience records only; no synchronous replay/storage lookup on the request path. |
| Randomness | injectable or replay-recorded enough for deterministic tests and post-hoc explanation. |
| Failure Mode | fail open to the base model with diagnostics if adaptation cannot score safely. |
| Forbidden | recipe mutation, decision rematching, hidden classifier edits, or public strategy-specific config blocks. |

The router should be able to answer these questions from replay after each
request:

- Which candidate set was used?
- Which candidates were eligible and why?
- Was sampling allowed or suppressed?
- Did the selected value come from posterior mean or a bounded sample?
- Why did adaptation keep the base model or propose a switch?
- What did protection do with the proposal?

Future adaptation strategies can add contextual behavior after the feature,
diagnostic, and eval contracts are stable. Candidate follow-ups include:

- contextual posterior sampling over a fixed request and decision feature
  vector
- confidence-bound scoring for low-data candidate sets
- preference-based ranking when outcomes can represent comparative model
  evidence
- reliability-aware pruning when provider and model health evidence is strong
  enough to remove candidates before scoring

These follow-ups should appear as new `adaptation.strategy` values. They should
not add new top-level learning product concepts and should not bypass
protection.

## Protection

Protection keeps agent sessions coherent and protects prefix cache, tool-loop
continuity, and handoff cost.

Protection is not a policy classifier. It does not decide that privacy,
security, or local-only traffic is special. Those boundaries are expressed by
matched decisions through `adaptations.mode: bypass` or component-level
`bypass`.

Protection reads:

- session id
- conversation id
- protected model
- turn count
- tool-loop state
- protocol state, such as tool requests and tool results
- cache evidence
- model switch history
- handoff cost
- provider failure and retry evidence
- agent pressure, such as repeated tool failures or failed verification

Protection owns identity-scoped state. The first implementation should keep the
state small and typed:

| Field | Meaning |
| --- | --- |
| `scope` | `conversation` or `session`. |
| `state_key_hash` | Bounded hash of the configured identity key. |
| `protected_model` | Current model that protection is trying to preserve. |
| `last_decision_id` | Last matched decision id observed in the protected scope. |
| `last_tier` | Last matched decision tier observed in the protected scope. |
| `turn_count` | Number of protected turns since the current epoch started. |
| `last_seen` | Idle timeout and epoch reset input. |
| `switch_count` | Recent switch history used as switch-cost evidence. |
| `tool_or_protocol_active` | Whether tool-call, tool-result, or protocol continuation state is active. |
| `cache_evidence` | Bounded cache hit/write evidence for switch-cost calculation. |

Protection state is not public recipe policy. It is local runtime state used to
decide whether a proposed switch is safe now.

### Protection Contract

Protection must also have a narrow typed contract:

| Contract | Requirement |
| --- | --- |
| Preflight Input | matched decision, component modes, active identity, tool/protocol evidence, current protected state, cache evidence, provider/retry evidence. |
| Preflight Output | sampling policy, action, scope, state key hash, and reason. |
| Switch Input | base model, proposal model, proposal gain, switch cost evidence, protected state, rescue evidence, component modes. |
| Switch Output | final model, action, scope, switch cost, rescue flag, updated state, and reason. |
| State | identity-scoped local state only. |
| Failure Mode | fail open with diagnostics when identity is missing or state cannot be loaded. |
| Forbidden | model quality scoring, recipe mutation, decision rematching, or overriding decision `bypass`. |

Protection should never make a weak model permanent. It preserves continuity
when the current model is good enough, and it allows deterministic rescue when
runtime evidence shows the agent is stuck.

### Preflight Sampling Guard

Before adaptation samples, protection returns a sampling policy:

| Policy | Meaning |
| --- | --- |
| `allowed` | Adaptation may draw stochastic samples for eligible candidates. |
| `suppressed` | Adaptation must use deterministic posterior means for this request. |

Sampling should be suppressed by default for:

- agent steps that carry tool-call or tool-result state
- protocol continuation steps where a model change would be hard to interpret
- routine low-risk traffic with a healthy current model
- decisions whose adaptation mode is `observe` or `bypass`

Suppressing sampling is not the same as hard-locking the model. The switch guard
can still allow a deterministic switch if the proposal model has a large enough
advantage.

### Switch Guard

The switch rule is:

```text
switch if proposal_gain >= switch_margin + stability_weight * switch_cost
```

Where:

- `proposal_gain` is the proposed model's advantage over the protected, current,
  or base model.
- `switch_cost` is derived from cache, handoff, tool-loop, session, and switch
  history evidence.
- `switch_margin` is the minimum gain required before any switch.
- `stability_weight` controls how much continuity, cache preservation, and
  handoff cost matter.

Expensive candidates pay a larger effective switch cost unless there is strong
quality, reliability, or rescue evidence.

### Rescue Guard

Protection should avoid trapping an agent on a weak model. When request evidence
shows that the current model is likely underpowered, protection may allow a
bounded rescue switch even when cache or handoff cost is high.

Initial rescue evidence:

- repeated tool failures
- repeated retries after weak or incomplete answers
- failed verification
- high agent pressure after enough agent steps
- a capability requirement that the protected model cannot satisfy

Rescue is bounded:

- it is per request or a short window, not a permanent session upgrade
- it still respects decision-level `bypass`
- it records an explicit replay reason
- it does not enable random exploration during tool/protocol steps unless
  preflight allows it

Initial rescue should be deterministic. It should use route evidence, outcomes,
retries, tool failures, and capability pressure; it should not depend on random
sampling to escape a weak model.

### Protection Actions

Protection should use a small typed action set in replay and compact headers.

| Action | Meaning |
| --- | --- |
| `establish_baseline` | The first final model for the active scope became the protected model. |
| `allow_sampling` | Stochastic adaptation is allowed for this request. |
| `suppress_sampling` | Adaptation must score deterministically for this request. |
| `hold_current` | The proposal model was held back by the switch guard. |
| `allow_switch` | The proposal model cleared the switch guard. |
| `rescue_switch` | The switch was allowed because rescue evidence outweighed stability cost. |
| `bypass` | The matched decision disabled the protection component. |

Recommended protection reasons:

| Reason | Meaning |
| --- | --- |
| `new_conversation` | A conversation-scoped baseline was established. |
| `new_session` | A session-scoped baseline was established. |
| `idle_reset` | The old protected epoch expired and a new baseline was established. |
| `missing_identity` | The configured identity header was absent, so protection failed open. |
| `tool_or_protocol_state` | Sampling or switching was suppressed because a tool/protocol continuation was active. |
| `cache_cost_high` | The switch cost was dominated by cache preservation. |
| `handoff_cost_high` | The switch cost was dominated by model handoff cost. |
| `switch_margin_not_met` | The proposal did not clear the required margin. |
| `switch_allowed` | The proposal cleared the guard. |
| `rescue_evidence` | Rescue evidence allowed a bounded switch. |

## Outcome Ingestion

Outcomes are the shared feedback path for online adaptation and offline recipe
learning.

First-party endpoint:

```http
POST /v1/router/outcomes
```

Outcome shape:

```json
{
  "replay_id": "replay_123",
  "source": "agent",
  "target": "model",
  "target_ref": "google/gemini-2.5-flash-lite",
  "verdict": "underpowered",
  "reason": "insufficient_capability",
  "score": 0.2,
  "metadata": {
    "run_id": "run_42"
  }
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `replay_id` | Strong link to the routed request. |
| `source` | `user`, `agent`, `eval`, `operator`, `provider`, or `router`. |
| `target` | `model`, `route`, `policy`, `stability`, `provider`, or `router`. |
| `target_ref` | Optional model, decision, route, policy, provider, or router id. For `target: model`, omitted means the final routed model from replay. |
| `verdict` | `good_fit`, `underpowered`, `overprovisioned`, or `failed`. |
| `reason` | Optional structured reason string. |
| `score` | Optional normalized evidence strength. |
| `metadata` | Optional bounded metadata for run ids, eval ids, or provider ids. |

The outcome API is the explicit feedback entrypoint. Product UIs, chat clients,
agent harnesses, eval jobs, provider adapters, and operator tools can use any
local labels they want, but they should normalize feedback into this bounded
schema before sending it to the router. The response header `x-vsr-replay-id`
is the stable link from a routed request to later feedback.

Outcome semantics:

| Target | Verdict | Runtime Update |
| --- | --- | --- |
| `model` | `good_fit` | Increase model fit for the decision/tier/model key. |
| `model` | `underpowered` | Decrease model fit because capability was insufficient. |
| `model` | `overprovisioned` | Increase overuse or cost penalty without reducing quality. |
| `model` | `failed` | Update reliability or provider failure state. |
| `route` | any | Feed offline recipe learning; do not directly update model quality. |
| `policy` | any | Feed offline recipe learning; do not let adaptation override the boundary. |
| `stability` | any | Feed protection tuning and offline analysis. |
| `provider` | any | Update reliability evidence or feed offline incident analysis; do not update model fit directly. |
| `router` | any | Feed diagnostics, cache/cost/switch analysis, and offline recipe learning. |

Online adaptation should update experience only from `target: model`
outcomes. Route, policy, stability, provider, and router outcomes are preserved
for offline recipe learning and analysis unless a typed online consumer exists.
This keeps "wrong decision", "bad model", and "unhealthy provider" feedback
separate.

The first implementation should synchronously validate outcomes and record them
for replay-linked analysis. Online state updates are best effort; a transient
online-state update failure should not make a valid outcome unusable offline.

Outcome ingestion is also the bridge between online adaptation and offline
recipe learning:

- model-targeted outcomes update adaptation experience
- route, policy, and stability outcomes are replay evidence for offline recipe
  learning
- provider and router outcomes can update reliability evidence and explain
  failed routes
- eval and agent outcomes can be replayed to compare recipe variants without
  changing production routing

Feedback is not an eval-only feature. The same endpoint is used by interactive
clients, agent harnesses, offline eval jobs, provider adapters, and operator
tools. Producers may use their own UI language, but they must normalize it into
the typed outcome schema before sending it to the router. This keeps online
adaptation state and offline recipe learning on the same evidence model.

## Headers and Replay Diagnostics

Response headers should stay compact. Detailed explanation belongs in Router
Replay.

Recommended compact headers:

```text
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=propose_switch,protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=sampled_win,protection=switch_allowed
x-vsr-replay-id: replay_...
```

Do not encode a mini language in one learning header. Headers identify active
methods and final high-level actions only. Detailed guard inputs, scores,
samples, costs, state hashes, and candidate traces belong in replay.

Header values should be comma-separated lowercase tokens or `method=value`
pairs. They should not use semicolon-delimited parameter strings.

Inactive components should be omitted from component headers. For example, if
only protection is active:

```text
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=hold_current
x-vsr-learning-scopes: protection=session
x-vsr-learning-reasons: protection=cache_cost_high
```

For a new protected epoch, use the same header family rather than a special
route status:

```text
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=establish_baseline
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: protection=new_conversation
```

For a model switch, expose both the adaptation proposal and the protection
decision when both components were active:

```text
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=propose_switch,protection=allow_switch
x-vsr-learning-scopes: adaptation=tier,protection=conversation
x-vsr-learning-reasons: adaptation=posterior_win,protection=switch_allowed
```

User interfaces should render these as short labeled fields, not as one opaque
string. A compact status bar can show model, decision, action, scope, and
reason; a detailed debug view should read replay for candidate scores, cache
cost, sampled values, state hashes, and switch math.

Client and CLI displays should not surface old internal status names such as
`initial route`, `learn select`, `new conv`, or semicolon-delimited
`method;action=...` strings. They should map the compact headers into stable
labels. Recommended display labels:

| Header Value | Display Label |
| --- | --- |
| `adaptation=keep_base` | adaptation: kept base model |
| `adaptation=propose_switch` | adaptation: proposed switch |
| `adaptation=observe` | adaptation: observe only |
| `adaptation=bypass` | adaptation: bypassed |
| `protection=establish_baseline` | protection: baseline established |
| `protection=suppress_sampling` | protection: sampling suppressed |
| `protection=allow_sampling` | protection: sampling allowed |
| `protection=hold_current` | protection: held current model |
| `protection=allow_switch` | protection: switch allowed |
| `protection=rescue_switch` | protection: rescue switch |
| `protection=bypass` | protection: bypassed |

Model changes should be rendered as model transitions, separate from the
learning action labels. A row or status line should not need to overload one
column with model, decision, scope, action, and reason. The compact header
family is intentionally split so clients can show these fields separately.

Compact header reasons are normalized for UI and CLI use. Replay may also keep
raw protection-trace fields such as `hard_lock_reason` or `decision_reason`
under the detailed learning and route diagnostics blocks; those fields explain
the low-level guard math and should not be treated as the public compact reason
vocabulary.

Replay diagnostics should include typed learning blocks:

```json
{
  "learning": {
    "protection_preflight": {
      "enabled": true,
      "mode": "apply",
      "scope": "conversation",
      "sampling_policy": "allowed",
      "action": "allow_sampling",
      "reason": "no_tool_or_protocol_state"
    },
    "adaptation": {
      "enabled": true,
      "mode": "apply",
      "candidate_set": "tier",
      "strategy": "routing_sampling",
      "base_model": "simple-model",
      "proposal_model": "frontier-model",
      "decision": "complex_code",
      "decision_tier": 3,
      "sampling": {
        "used": true,
        "posterior_mean": 0.71,
        "sampled_quality": 0.83
      },
      "reason": "posterior_win"
    },
    "protection": {
      "enabled": true,
      "mode": "apply",
      "scope": "conversation",
      "action": "allow_switch",
      "base_model": "simple-model",
      "proposal_model": "frontier-model",
      "final_model": "frontier-model",
      "switch_margin": 0.05,
      "stability_weight": 1.0,
      "switch_cost": 0.03,
      "rescue": {
        "active": false
      },
      "reason": "switch_allowed"
    }
  }
}
```

Raw session, conversation, user, tenant, and workspace identifiers should not be
stored in learning diagnostics. Store source, status, and bounded hashes.

## Offline Recipe Learning

Offline recipe learning uses replay and outcomes to improve the recipe. It can
run inside an agent loop:

```text
collect replay + outcomes
  -> run evals
  -> generate findings
  -> generate candidate recipes
  -> run experiments
  -> compare metrics
  -> output recipe patch and experience seed pack
```

The offline loop may propose changes to:

- signals
- examples
- thresholds
- decision rules
- decision `tier`
- decision `priority`
- decision `modelRefs`
- adaptation config
- protection config
- experience seed packs

First-version outputs:

| Artifact | Purpose |
| --- | --- |
| `findings` | Explain what looks wrong or suboptimal. |
| `metrics` | Summarize correctness, cost, latency, cache, switch rate, and outcome quality. |
| `candidate_recipes` | Alternative recipe variants produced by the agent. |
| `experiment_results` | Metrics for candidate recipes. |
| `recipe_patch` | Proposed config changes. |
| `experience_seed_pack` | Offline quality seeds for adaptation cold start. |

Offline recipe learning is where recipe edits happen. It uses replay, outcomes,
eval cases, and experiment metrics to produce findings and candidate patches.
It does not silently apply those patches on the request path.

The offline loop is the place where classifier-like recipe improvement happens.
It can inspect replay, outcomes, failed eval cases, and successful counterfactual
experiments, then propose edits to signals, examples, thresholds, decisions,
tiers, priorities, and modelRefs. Those edits are recipe patches, not hidden
runtime mutations.

The first offline loop should be agent-usable even before a dashboard exists.
Artifacts must be structured enough for an external agent to inspect evidence,
edit a recipe, run the experiment again, and decide whether to keep the patch.
A prose-only report is not enough.

The first version should make the loop usable before adding promotion gates.
That means it should produce clear findings, metrics, experiments, and patches;
human or external agent review can decide whether to apply those patches.

The first loop should optimize for useful evidence:

- identify routing decisions that look wrong
- identify decisions where a cheaper or local model is usually sufficient
- identify decisions where the selected model is often underpowered
- identify protection settings that cause excessive holds or excessive switches
- identify candidate-set choices that waste cost or miss useful alternatives
- generate candidate recipe patches and run experiments against them

The loop should be useful without an automatic promotion gate. A first version
can run as an explicit command, produce artifacts, and leave application of the
recipe patch to a human or an external agent.

The first implementation should be structured as explicit offline modules, not
as one opaque eval script:

| Module | Responsibility |
| --- | --- |
| Replay loader | Read replay routes, responses, diagnostics, and outcomes from an endpoint or exported file. |
| Case loader | Load optional eval cases with expected decision/model/cost behavior. |
| Metric engine | Compute per-decision and per-tier correctness, cost, latency, cache, switch, rescue, and outcome metrics. |
| Finding detector | Convert metrics and replay evidence into actionable findings with supporting replay ids. |
| Candidate generator | Produce complete candidate recipes that change only recipe-level policy or learning config. |
| Experiment runner | Compare baseline and candidate recipes against replay/eval cases. |
| Patch writer | Emit the minimal recipe patch and explain why each edit was proposed. |
| Seed-pack builder | Optionally emit quality seeds and seed weights for adaptation cold start. |

This loop is also where classifier-like recipe improvement belongs. For example,
if route outcomes show that a decision rule is too broad or too narrow, the
offline loop should propose changes to examples, thresholds, priorities, tiers,
or modelRefs. The online router should not mutate those policy fields from
feedback on the request path.

### Experience Seed Packs

Experience seed packs are optional offline artifacts. They are not public
request-path policy and they are not a replacement for the recipe.

The first implementation should define, validate, and export seed packs from
offline recipe learning. It may use seed packs inside offline experiments. It
should not expose a public runtime `seed_pack`, `experience.source`, or
`router.learning.memory` field to import them. Runtime import and warmup can be
added later as a separate design once retention, validation, freshness, and
multi-replica behavior are specified.

A seed pack should be shaped around adaptation experience, not generic lookup
tables:

| Field | Meaning |
| --- | --- |
| `decision_id` | Optional decision-specific context. |
| `decision_tier` | Optional tier context. |
| `model` | Model the seed applies to. |
| `quality_seed` | Normalized cold-start quality estimate. |
| `seed_weight` | Pseudo-count strength of the estimate. |
| `source_metric` | Metric or eval source that produced the seed. |
| `support` | Bounded replay ids, eval ids, or aggregate counts. |

If a future runtime import path initializes experience records from a seed pack,
it must still apply decision bypass, candidate scope, protection, reliability,
and cost guards. A seed pack should make cold start better; it must not create
an unreviewed routing policy.

### Offline Loop Contract

The offline loop should consume:

| Input | Purpose |
| --- | --- |
| Replay routes | Reconstruct base, proposal, final model, decision, tier, headers, and learning diagnostics. |
| Outcomes | Explain model fit, overuse, route mistakes, policy misses, and stability problems. |
| Eval cases | Provide expected route/model behavior and cost/latency expectations. |
| Candidate recipe | The current recipe plus generated recipe variants. |
| Optional experience seed packs | Quality seed estimates for offline cold-start experiments. Runtime import is future work. |

The loop should produce:

| Output | Required Contents |
| --- | --- |
| Findings | Per-decision explanation of observed routing problems and supporting replay ids. |
| Metrics | Per-decision and per-tier correctness, cost, latency, cache, switch, and outcome metrics. |
| Candidate recipes | Complete recipe variants that can be replayed or served in a test environment. |
| Experiment results | Side-by-side metrics for baseline and candidate recipes. |
| Recipe patch | Minimal proposed config diff. |
| Experience seed pack | Optional quality seeds and seed weights for adaptation cold start. |

The first loop should not train or mutate the online router directly. If an
agent wants to improve classifier-like recipe behavior, it should do so by
running evals, generating candidate recipes, comparing metrics, and outputting a
recipe patch.

## Evaluation Metrics

The eval loop should report:

- route correctness
- model fit
- overuse rate
- underpowered-model rate
- provider failure rate
- policy-boundary misses
- unnecessary switch rate
- cache preservation
- latency overhead
- cost savings against simple baselines
- adaptation win rate
- sampling suppression rate
- protection stay/switch reasons
- rescue switch rate and rescue success rate
- outcome coverage
- replay explainability coverage

Metrics should be reported per decision and per decision tier.

## Data Structures

The implementation should use typed configs, enums, policies, and result
structs. Maps are acceptable only at serialization boundaries such as replay
JSON, response headers, and API metadata.

Core contracts:

| Contract | Required Shape |
| --- | --- |
| `RouterLearningConfig` | `enabled`, `adaptation`, `protection`. |
| `LearningAdaptationConfig` | `enabled`, `strategy`, `candidate_set`. |
| `LearningProtectionConfig` | `enabled`, `scope`, `identity`, `tuning`. |
| `DecisionAdaptationsConfig` | `mode`, `adaptation.mode`, optional `adaptation.candidate_set`, `protection.mode`, optional `protection.stability_weight`, optional `protection.switch_margin`. |
| `LearningPipelineInput` | base selection, matched decision, tier, request evidence, identity, replay ids. |
| `AdaptationDecision` | action, candidate set, strategy, base model, proposal model, score diagnostics. |
| `ProtectionDecision` | action, scope, protected model, proposal model, final model, switch cost, reason diagnostics. |
| `RouterOutcome` | replay id, source, target, target ref, verdict, reason, score, metadata. |
| `ExperienceRecord` | typed online state keyed by decision, tier, and model, including quality seeds, outcome counts, latency, cache, cost, reliability, and freshness fields. |

The learning pipeline owns ordering and composition. Adaptation and protection
components should not call each other directly.

The implementation should keep these as real structs or equivalent typed
classes, not anonymous nested maps passed between components. Serialization can
flatten them for headers or replay, but component boundaries should remain
typed so additional strategies can be added without copying ad hoc field
mutation patterns.

Global component `enabled` fields should preserve the tri-state semantics from
the public API: omitted means inherit the default-on behavior under
`learning.enabled: true`; explicit `false` disables that component.

### Engineering Contracts

The implementation should avoid weakly structured policy bags. Each learning
component should expose a small typed interface and typed diagnostics:

| Area | Contract |
| --- | --- |
| Config | Typed structs and enums for strategies, scopes, modes, actions, targets, verdicts, and candidate sets. |
| Pipeline | A small orchestrator that calls protection preflight, adaptation, and protection switch guard in order. |
| Strategy factory | A registry from `adaptation.strategy` enum values to typed strategy implementations. The first value is `routing_sampling`; future values plug into the same contract. |
| Adaptation | Returns `AdaptationDecision`, not a generic map. |
| Protection | Returns `ProtectionDecision`, not a generic map. |
| Headers | Render compact headers from typed decisions. |
| Replay | Serialize typed diagnostic blocks; do not construct policy by setting arbitrary string keys. |
| Outcomes | Validate into a typed `RouterOutcome` before storage or online updates. |
| Experience | Update through an adaptation-owned interface; do not expose generic state mutation to other modules. |

Maps remain acceptable for API metadata, replay JSON serialization, and header
key/value rendering. They should not be the internal contract between learning
components.

The minimum internal interfaces should be shaped around typed decisions, not
loosely shared maps:

```text
Protection.Preflight(input) -> ProtectionPreflightDecision
AdaptationStrategy.Select(input, preflight) -> AdaptationDecision
Protection.DecideSwitch(input, adaptation) -> ProtectionDecision
LearningPipeline.Route(input) -> LearningResult
```

Each return type should carry typed action, reason, scope, model, score, and
diagnostic fields. Header and replay renderers can convert those structs into
compact strings or JSON at the edge.

Strategy selection should happen once through a small factory or registry. Do
not scatter `if strategy == ...` blocks across config loading, runtime routing,
replay rendering, and dashboard code. Runtime components should depend on the
typed `AdaptationStrategy` contract, not on strategy-specific structs.

## Implementation Invariants

- Recipe policy chooses the matched decision and base candidate set.
- Adaptation can propose a model but cannot rewrite recipe policy.
- Protection can hold, allow, or rescue a switch but does not score model
  quality by itself.
- Decision `adaptations.mode: bypass` wins before adaptation or protection can
  change the route.
- Request-time routing must not depend on synchronous external storage reads.
- Replay is the durable evidence log; in-process state is the hot-path cache of
  learned experience and protection state.
- Headers stay compact and only expose active methods, actions, scopes, and
  reasons.
- Detailed scores, samples, costs, state hashes, and candidate traces belong in
  typed replay diagnostics.
- Internal configs, policies, actions, scopes, outcomes, and diagnostics should
  use typed structs and enums. Maps are acceptable at serialization boundaries
  only.

## Implementation Plan

The plan below is the follow-up work breakdown. Each phase should land with
focused tests and without accepting removed public API names.

The phases are ordered by dependency. A later phase may be developed in
parallel only when it consumes the typed contract produced by the earlier
phase, not by reaching into temporary maps or legacy runtime state.

Agent execution rules for this proposal:

- Treat this document as the implementation contract, not as background notes.
- Implement the public API exactly as written before adding runtime behavior.
- Do not add compatibility rewrites for removed learning config names.
- Keep adaptation and protection independently testable behind typed structs.
- Keep request-path state local and bounded; do not add synchronous storage
  reads to make a test pass.
- Put detailed explanation in replay and keep headers compact.
- Do not add new public concepts such as memory backends, lookup-table priors,
  goals, or legacy algorithm family names while implementing the first loop.
- If implementation pressure suggests a new config field, first prove it cannot
  be expressed through recipe policy, candidate set, component mode,
  protection tuning, outcome ingestion, or offline recipe learning.
- Do not mention external reference projects or borrowing language in code,
  docs, headers, replay diagnostics, or issue text. The public design stands on
  vLLM Semantic Router's recipe, replay, outcome, adaptation, and protection
  contracts.

For follow-up agents, the completion rule is:

```text
public config validates
  -> typed pipeline routes
  -> replay explains every learning action
  -> outcomes update only the allowed online state
  -> offline recipe learning produces artifacts that can be reviewed
```

If an implementation detail is not needed to satisfy that chain, it should stay
out of the first PR.

### First Implementation Scope

The implementation task should close the first usable self-improving router
loop, not the entire long-term roadmap.

First implementation scope:

| Area | Must Land |
| --- | --- |
| Public config | `global.router.learning`, `learning.adaptation`, `learning.protection`, and decision `adaptations` exactly as described above. |
| Config cleanup | Old public learning names are rejected, not translated. |
| Runtime composition | One typed learning pipeline that composes adaptation and protection in the fixed order. |
| Strategy registration | A small typed strategy factory for `learning.adaptation.strategy`, initially registering only `routing_sampling`. |
| Adaptation | `routing_sampling` with `decision`, `tier`, and `global` candidate sets. |
| Protection | Conversation and session scope with sampling preflight, switch guard, and deterministic rescue. |
| State | Local typed protection state and local typed model experience. |
| Outcomes | Replay-linked `POST /v1/router/outcomes` with model-targeted updates to experience. |
| Observability | Compact learning headers plus typed replay diagnostics. |
| Offline loop | A runnable recipe-learning path that produces metrics, findings, candidate recipes or patches, and optional seed-pack artifacts. |
| Validation | Unit, config, replay, local smoke, and AMD agentic recipe validation for the first loop. |

Explicitly out of current scope:

- automatic recipe promotion
- multi-replica state sharing
- synchronous external hot-path state lookup
- dashboard UX for recipe-learning artifacts
- contextual feature-model training on the request path
- additional adaptation strategies beyond `routing_sampling`
- compatibility aliases for old learning config names

If a follow-up implementation hits an ambiguity, prefer the narrower contract:
typed local state, no hidden recipe edits, compact headers, replay for detail,
and explicit offline recipe patches.

### Phase 0: Remove the Old Learning Surface

- Delete old public learning config blocks instead of translating them.
- Replace old runtime method names with `adaptation` and `protection`.
- Replace old single-value learning header values with the compact header set.
- Replace generic replay learning maps with typed `protection_preflight`,
  `adaptation`, and `protection` diagnostics.
- Keep old docs and tutorials from presenting removed API shapes.
- Reject old global and decision-level fields with validation errors, including
  old selector blocks and old method names.
- Remove parsing paths that translate old names into the new API.
- Delete dead old learning implementations when they are no longer used by the
  new pipeline. If a historical selector remains for a non-learning recipe path,
  keep it isolated from `global.router.learning` and do not document it as
  Router Learning.

Done signal:

- old learning names fail validation with path-specific errors
- examples, tutorials, and proposals show only the new API
- no runtime compatibility path silently rewrites old learning blocks into the
  new shape

### Phase 1: Runtime Interfaces and State Contracts

- Add a learning pipeline that receives the base selection result and returns a
  final selection result.
- Add typed component interfaces for protection and adaptation.
- Add typed online protection state keyed by session or conversation identity.
- Add typed experience records keyed by decision, tier, and model.
- Add a typed outcome model shared by API handlers, replay storage, and online
  experience updates.

Done signal:

- component interfaces compile without map-based policy mutation
- config, pipeline inputs, decisions, replay diagnostics, and outcomes all use
  typed structs/enums internally
- serialization tests prove replay/header output remains compact and stable

Non-goal for this phase:

- do not implement a second online strategy
- do not add external storage reads to the request path
- do not add a public memory or experience config block

### Phase 2: Protection Runtime

- Implement conversation and session scope.
- Implement identity resolution from configured headers.
- Implement preflight sampling suppression for agent, tool, protocol, routine,
  and decision-bypass cases.
- Compute switch cost from cache, tool-loop, handoff, turn, and switch history
  evidence.
- Apply `proposal_gain >= switch_margin + stability_weight * switch_cost`.
- Implement deterministic bounded rescue switching.
- Record stay/switch diagnostics in replay and compact headers.

Done signal:

- conversation scope protects within one `x-conversation-id`
- session scope protects across conversations in one `x-session-id` until idle
- missing identity fails open with diagnostics
- tool/protocol states suppress sampling without permanently trapping the model
- rescue switch is deterministic and replay-explainable

### Phase 3: Adaptation Runtime

- Build candidate sets for `decision`, `tier`, and `global`.
- Implement experience fallback lookup:
  `decision + tier + model -> tier + model -> model`.
- Implement `routing_sampling` with posterior mean and guarded sampling.
- Apply cost, overuse, reliability, latency, and cache adjustments.
- Respect protection preflight, decision bypass, component `observe`, and
  component `bypass`.
- Record candidate scores, sampling metadata, and selected proposals in replay.
- Keep strategy state and score diagnostics typed. Do not reintroduce generic
  policy maps as the component contract.
- Make online sampling bounded, explainable, and suppressible by protection.

Done signal:

- `decision`, `tier`, and `global` candidate sets are deterministic
- decision-level `adaptations.mode: bypass` prevents adaptation changes
- `observe` records diagnostics without changing the final model
- posterior mean is used when protection suppresses sampling
- bounded stochastic sampling is used only when protection allows exploration
- replay explains candidate scores, sampled values, proposal, and final result

Implementation note:

`routing_sampling` is the first online model-choice algorithm. It should be
implemented as a typed strategy selected by `learning.adaptation.strategy`, not
as a nested config family. Its state is adaptation-owned experience; its public
operator controls are the global strategy field, candidate set, decision
component mode, and protection interaction.

### Deferred Strategy Roadmap

This section is not part of the first implementation boundary. It is the
follow-up strategy roadmap after the day-0 loop is working end to end.

Do not reintroduce old algorithm-family config blocks. Future online model
choice algorithms should be added as `adaptation.strategy` values only after
they can use the same contracts as `routing_sampling`.

Known follow-up tracks are:

- contextual posterior sampling that uses request, decision, tier, cache, and
  tool-state features without adding synchronous external reads
- confidence-bound scoring once low-data exploration can be explained in replay
  and measured in eval
- preference-based ranking once outcome ingestion can represent pairwise or
  comparative model evidence cleanly
- reliability-aware candidate pruning when provider and model health evidence
  is strong enough to remove candidates before scoring
- deterministic optimizer strategies for deployments that want no online
  stochastic exploration and move exploration entirely to offline experiments

Future strategies should:

- consume typed replay, outcomes, experience, and protection state
- preserve decision-level bypass and component modes
- keep request-time reads local and bounded
- emit the same compact headers and typed replay diagnostics
- prove value with route correctness, cost, latency, cache, and switch metrics

Base routing algorithms that are not online self-improvement mechanisms remain
under the recipe's routing algorithm surface, not under learning.

Future strategy candidates should be tracked as separate issues only after the
day-0 strategy works end to end. A follow-up agent closing the first
implementation should not implement these strategies unless the issue or PR
scope explicitly says so:

| Strategy Family | Why It Might Matter | Must Prove Before Implementation |
| --- | --- | --- |
| Contextual posterior sampling | Use request and route features when model quality differs by subtask. | Stable feature contract, replay explainability, and eval lift over `routing_sampling`. |
| Confidence-bound scoring | Explore low-data candidates without random posterior draws. | Lower regret or safer exploration in sparse routes. |
| Preference ranking | Learn from comparative model outcomes instead of single-model verdicts. | Outcome API can express pairwise evidence cleanly. |
| Reliability-aware pruning | Remove unhealthy providers or models before scoring. | Provider evidence is strong enough and false-positive pruning is controlled. |
| Deterministic optimizer | Disable online stochastic exploration and rely on offline experiments. | Comparable cost/quality improvements without online sampling. |

These are adaptation strategies, not separate global learning concepts.

Existing experimental or historical learning algorithms should not be surfaced
through their old public config names. If an algorithm remains useful, it must
be reintroduced through the same strategy contract: typed inputs, typed
decisions, local request-path state, compact headers, replay diagnostics,
decision bypass, and protection composition.

### Phase 4: Outcome Ingestion

- Add `POST /v1/router/outcomes`.
- Validate source, target, target ref, verdict, score, and bounded metadata.
- Accept only the first-version source values:
  `user`, `agent`, `eval`, `operator`, `provider`, and `router`.
- Accept only the first-version target values:
  `model`, `route`, `policy`, `stability`, `provider`, and `router`.
- Accept only the first-version verdict values:
  `good_fit`, `underpowered`, `overprovisioned`, and `failed`.
- Store outcomes linked by `replay_id`.
- Update online experience for `target: model` outcomes.
- Preserve route, policy, stability, provider, and router outcomes for offline
  recipe learning and diagnostics.

Done signal:

- invalid source, target, verdict, score, or metadata is rejected
- valid outcomes are replay-linked
- `target: model` updates online experience when possible
- `target: model` updates only the matching fit, overprovisioning, or failure
  state implied by the verdict
- route/policy/stability/provider/router outcomes do not mutate model quality
  directly
- online-state update failure does not drop a valid offline outcome record

### Phase 5: Offline Recipe Learning Loop

- Build replay/outcome loading from both a live router endpoint and exported
  replay files.
- Load optional eval cases that describe expected decision, model, local-only,
  cost, latency, or stability behavior.
- Run route correctness, model fit, overuse, underpowered-model, provider
  failure, cost, latency, cache, switch-rate, and rescue-rate evals.
- Generate findings for wrong decisions, overuse, underpowered selections,
  excessive switching, excessive holds, missing protection, and overly broad
  candidate sets.
- Generate complete candidate recipes, not only textual suggestions.
- Run experiments against baseline and candidate recipes and report
  side-by-side metrics.
- Output a minimal recipe patch with supporting replay ids and metric deltas.
- Optionally output an experience seed pack for cold-start quality seeds.
- Keep the offline loop capable of being driven by an agent: every finding must
  carry enough replay ids, metrics, and candidate recipe diffs for the next
  agent step to inspect, rerun, or apply.

Phase 5 is not complete if it only summarizes replay. It must close the offline
learning loop by producing candidate recipes or a recipe patch that an agent can
inspect, test, and apply through the normal recipe deployment path.

Recipe learning is the offline counterpart to online adaptation. It is where
classifier-like improvements happen. If feedback shows that a signal is too
broad, a threshold is too weak, a decision priority is wrong, or a tier/modelRef
assignment is stale, the offline loop should produce a recipe patch and
experiment evidence. It should not push those edits into the live router as a
hidden runtime mutation.

Done signal:

- the loop can consume replay/outcome exports and optional eval cases
- findings include supporting replay ids and per-decision metrics
- at least one candidate recipe path can be generated for overuse,
  underpowered-model, wrong-route, and excessive-switch findings
- candidate recipes are experimentally compared against baseline
- output includes metrics, findings, experiment results, and a minimal patch

Agent usability signal:

- every finding has a stable id, severity, affected decisions, supporting
  replay ids, metric deltas, and a concrete next action
- every patch explains which finding it addresses
- the loop can run in report-only mode and patch-generating mode
- generated candidate recipes remain complete recipes that can be validated by
  the same config validator as production recipes

Non-goal for this phase:

- do not auto-apply generated recipe patches
- do not train or mutate online route classifiers from the request path
- do not require a dashboard workflow before the command-line artifact loop is
  useful

The agent-facing loop should be complete enough that an external agent can:

1. run the offline learner against replay/outcome data
2. inspect findings and supporting replay ids
3. generate candidate recipe variants
4. run experiments and compare metrics
5. apply or reject the proposed patch through normal config review

### First Implementation Boundary

The first implementation is complete when these pieces work together:

- clean config API for global learning and decision `adaptations`
- typed runtime pipeline with adaptation and protection as separate components
- `routing_sampling` over `decision`, `tier`, and `global` candidate sets
- conversation and session protection scopes
- compact learning headers with no semicolon mini-language
- replay diagnostics for protection preflight, adaptation, protection, and
  outcomes
- outcome ingestion that records replay-linked feedback and updates local
  model experience for `target: model`
- an offline recipe learning command or harness that can export replay/outcome
  data, run experiments, produce metrics, and output a recipe patch
- docs and examples that use only the new public API

The first boundary does not require a full contextual online learner, automatic
recipe promotion, distributed learning state, or a UI dashboard for eval
artifacts.
Those are follow-ups after the single-router loop is usable and explainable.

Do not add contextual feature learning, external hot-path storage, automatic
recipe promotion, or multi-replica shared state before this boundary is closed.

### Blocking Questions

There are no blocking product questions left for the first implementation.
Follow-up agents should proceed with the confirmed API and boundary decisions in
this document.

If an implementation detail is underspecified, use these defaults:

- prefer the narrower candidate set
- prefer deterministic behavior when protection suppresses sampling
- prefer fail-open routing with diagnostics over request failure
- prefer replay diagnostics over new response headers
- prefer offline recipe artifacts over request-path recipe mutation
- prefer typed structs/enums over maps inside component boundaries

The remaining questions are intentionally deferred and should not block the
first implementation:

| Deferred Question | Where It Belongs |
| --- | --- |
| Should future strategies use contextual features, confidence bounds, or pairwise preferences? | Separate `adaptation.strategy` follow-up issues after `routing_sampling` is stable. |
| Should online state be shared across replicas? | Storage and multi-replica follow-up after single-router semantics are validated. |
| Should recipe patches have automatic promotion gates? | Recipe promotion workflow follow-up after offline artifacts are useful. |
| Should dashboards visualize recipe-learning artifacts? | Evaluation dashboard follow-up after CLI/agent artifacts are stable. |
| Should operators tune multi-objective score weights? | Future API proposal only if eval shows fixed semantics are insufficient. |

### Follow-Up Tracks

The tracks below are explicit follow-ups, not blockers for the first
implementation. They should be opened as concrete issues only after the day-0
`routing_sampling` + protection + outcomes + offline artifact loop is working
end to end.

Each follow-up issue should state:

- the public API impact, if any
- the runtime state it reads or writes
- the replay diagnostics it needs
- the eval metric that proves value
- the clean-break stance, unless explicitly changed

Recommended issue split after the first implementation:

| Issue Track | Build | First Useful Acceptance Evidence | Keep Out Of Scope |
| --- | --- | --- | --- |
| Contextual adaptation strategy | A new `adaptation.strategy` value that consumes a fixed request, decision, tier, cache, and tool-state feature vector and emits the same typed adaptation diagnostics. | Replay can explain feature values, score components, candidate choice, and metric lift over `routing_sampling`. | Hot-path classifier training, recipe mutation, or new top-level learning concepts. |
| Low-data exploration strategy | A deterministic or confidence-bound strategy for sparse candidate sets. | Eval shows safer sparse-route exploration or lower regret than bounded posterior sampling. | Changing the `routing_sampling` public shape or bypassing protection. |
| Comparative outcome support | Pairwise or preference-style outcome schema plus offline metrics for comparative model evidence. | A replay-linked pairwise outcome can affect offline metrics without breaking existing model outcomes. | Replacing the first `good_fit/underpowered/overprovisioned/failed` outcome path. |
| Reliability-aware candidate pruning | Typed provider/model health evidence that can remove unhealthy candidates before adaptation scoring. | Provider failure evidence removes a candidate with replay diagnostics and no false quality penalty. | Treating transient HTTP success as model quality reward. |
| Runtime seed-pack import/warmup | Optional validation, import, freshness, and cold-start warmup path for seed packs produced by offline recipe learning. | A seed pack can initialize local experience before traffic while still respecting bypass, candidate scope, protection, cost, and reliability guards. | Turning seed packs into hidden policy or synchronous request-path storage reads. |
| Multi-replica learning state | Sticky-session guidance, replay-to-state rebuild semantics, and optional shared-state strategy. | A multi-replica deployment has documented fail-open behavior, timeout semantics, and latency budget before any external request-path read is added. | External hot-path reads before latency, consistency, and fail-open semantics are defined. |
| Recipe promotion workflow | Review, gate, and apply flows for offline recipe patches. | A generated patch can be evaluated, approved, and deployed through explicit review steps. | Automatic production mutation without explicit review or deployment. |
| Evaluation dashboard | UI for findings, metrics, replay links, patches, and experiment diffs. | A user can inspect why a patch was proposed and which replay ids support it. | Blocking the CLI/agent artifact loop on dashboard availability. |

Storage and multi-replica work is a follow-up track, not a hidden first-version
requirement. The first implementation keeps request-time state local. A later
storage design must first define sticky-session behavior, shared-state timeout,
replay-to-state rebuild, consistency expectations, and fail-open semantics.

The meta issue should state that the first completed PR owns the day-0
`routing_sampling` + protection + outcomes + offline artifact loop. The tracks
above are follow-ups that extend that loop without changing the first public
API vocabulary.

## Validation Plan

- Config validation rejects old learning shapes and accepts only the new API.
- Config validation proves `learning.enabled: true` defaults both components
  on, while explicit component `enabled: false` disables only that component.
- CLI schema and recipe validation match the Go config contract.
- Unit tests cover decision bypass, component observe/apply/bypass, candidate
  sets, protection scopes, sampling suppression, switch guard, rescue guard,
  headers, replay diagnostics, and outcome ingestion.
- Unit tests cover adaptation-only and protection-only operation.
- Unit tests cover missing identity headers as fail-open diagnostics, not
  request errors.
- Replay tests prove learning diagnostics are typed and replay-linked.
- Local smoke proves headers stay compact and replay carries the detailed
  explanation.
- AMD regression proves the agentic recipe can route simple, complex, privacy,
  and domain tasks while preserving conversation/session stability.
- Agent/eval harness integration proves the offline loop can compute
  correctness, switch, cache, latency, and cost metrics from replay and
  outcomes.

## Confirmed Decisions

- This is a breaking public API change; old learning config names are removed.
- Public runtime concepts are `adaptation` and `protection`.
- Decision-level controls live under `routing.decisions[].adaptations`.
- `learning.enabled: true` defaults adaptation and protection on unless each
  component is explicitly disabled.
- `routing_sampling` is the day-0 online adaptation strategy and is the default
  when adaptation is enabled.
- `candidate_set` values are `decision`, `tier`, and `global`.
- A decision may override the global candidate set with
  `adaptations.adaptation.candidate_set`.
- `tier` candidate set is the union of `modelRefs` from decisions with the same
  `decision.tier`.
- `candidate_set: tier` falls back to `decision` when the matched decision has
  no positive tier.
- `global` candidate set is the deployed model/provider inventory, not only the
  union of models referenced by decisions.
- Adaptation may use stochastic sampling online when protection allows it.
- `routing_sampling` is allowed to use cold-start quality seeds plus local
  posterior evidence, but it does not train a hot-path classifier or mutate
  recipe policy.
- Protection has a preflight guard before adaptation sampling and a switch
  guard after adaptation scoring.
- Protection can suppress stochastic exploration without hard-locking the
  current model.
- Protection can guard base-route switches even when adaptation is disabled.
- Protection can allow bounded rescue switches when an agent appears stuck or
  underpowered.
- The first route in a protected identity scope is reported as
  `protection=establish_baseline`, not as an opaque "initial route" status.
- Compact learning headers use comma-separated tokens and `method=value` pairs,
  not semicolon parameter strings.
- Decision `adaptations.mode: bypass` is how users express sensitive policy
  boundaries.
- `goals` is not part of the first public API.
- Decision-level learning control does not expose a generic weight map; use
  component `mode`, `adaptation.candidate_set`,
  `protection.stability_weight`, and `protection.switch_margin`.
- The first implementation keeps request-time state in-process and uses replay
  as the durable event log.
- Outcome ingestion is the shared feedback path for users, agents, evals,
  providers, routers, and operators.
- First-version outcome verdicts are `good_fit`, `underpowered`,
  `overprovisioned`, and `failed`.
- Offline recipe learning can propose changes to rules, tier, priority,
  modelRefs, adaptation config, protection config, and experience seed packs.
- Future online strategies extend `adaptation.strategy`; they do not add new
  top-level learning product concepts unless the operator-facing behavior is
  genuinely different from adaptation or protection.

## Success Criteria

- A recipe can enable adaptation and protection once globally.
- Decisions can bypass learning with one small block.
- Adaptation can search `decision`, `tier`, or `global` candidate sets.
- Tier candidate search uses the existing decision `tier` field.
- Routing sampling can explore better models without ignoring reliability,
  cost, decision bypasses, or protection.
- Protection can suppress sampling on agent/tool/protocol steps without forcing
  a hard model lock.
- Protection prevents unnecessary switches during agent conversations.
- Protection can allow bounded rescue switches for stuck or underpowered agent
  flows.
- Outcomes update online experience and feed offline recipe learning.
- Offline agent loops can produce findings, metrics, candidate recipes, recipe
  patches, and experience seed packs.
- Request-time routing does not require synchronous external storage reads.
