# Router Learning: Memory and Adaptations

## Status

Accepted. The first implementation covers session-aware Router Learning with
conversation/session scopes, decision-level `apply` / `observe` / `bypass`,
generic learning headers, replay diagnostics, and the AMD agentic routing
recipe. Broader Router Memory features such as public memory configuration,
replay-derived priors, multi-adaptation composition, and migration of other
learning-style algorithms remain future work.

## Summary

The router needs one product concept for cross-request routing intelligence:
**Router Learning**.

Router Learning has two parts:

- **Memory** records and learns from routing traffic.
- **Learning adaptations** consume memory to adjust a base routing decision.

Router Learning memory should unify three existing concepts that are currently described
separately:

- replay records as the event log
- session and conversation memory as online state
- lookup tables as replay-derived materialized priors

The first learning adaptation is session-aware routing stability. It decides when
to keep the current model and when to switch based on conversation continuity,
session continuity, prefix-cache evidence, handoff cost, and model-switch
history.

This proposal moves session-aware behavior out of decision-local
`algorithm.type: session_aware` and into a global router learning layer:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
      ttl_seconds: 2592000

  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
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
            cache_weight: 0.20
            handoff_penalty: 0.05
            handoff_penalty_weight: 1.0
            switch_history_weight: 0.04
            max_cache_cost_multiplier: 2.5
```

Decisions remain semantic. A decision can opt out of a learning adaptation when
it represents a hard policy boundary:

```yaml
adaptations:
  session_aware:
    mode: bypass
```

The default decision behavior is `mode: apply`.

## Implementation Status

| Area | Status | Notes |
| --- | --- | --- |
| Global `router.learning.adaptations.session_aware` API | Implemented | Session-aware routing moved out of `algorithm.type=session_aware`. |
| Decision-level `adaptations.session_aware` | Implemented | Supports `apply`, `observe`, `bypass`, sparse scope overrides, and tuning overrides. |
| Conversation and session identity scopes | Implemented | `conversation` protects one run; `session` protects across runs until idle release. |
| Online session/conversation state | Implemented | First implementation uses low-latency in-process router state and single-replica assumptions. |
| Generic learning response headers | Implemented | Uses `x-vsr-learning-methods`, `x-vsr-learning-actions`, `x-vsr-learning-scopes`, `x-vsr-learning-reasons`, and `x-vsr-learning-modes`. |
| Replay learning diagnostics | Implemented, partial | Records method/action/reason/scope, base/final model evidence, cache/cost evidence, and hashed identity diagnostics. The exact replay schema can evolve. |
| Clean break from old public session-aware algorithm config | Implemented | Old `algorithm.type=session_aware`, `algorithm.session_aware`, `model_switch_gate`, and public `lookup_tables` config are rejected instead of rewritten. |
| AMD agentic routing recipe and guide | Implemented | Recipe uses session-aware Router Learning plus decision bypasses for hard privacy/security boundaries. |
| Public `router.learning.memory` config | Future work | Online state is internal today; no public memory backend, TTL, or storage controls are exposed. |
| Replay-derived materialized priors | Future work | Lookup tables remain a roadmap concept under Router Learning memory, not a supported public API. |
| Additional adaptations and composition | Future work | Only `session_aware` is implemented; bandit, personalization, provider health, and cost optimizer need separate designs. |
| Elo / RL / GMT migration | Future work | These remain decision algorithms in this implementation. |
| Distributed memory consistency | Future work | Multi-replica memory semantics, storage hot-path policy, and sticky routing are out of scope. |
| Full eval harness | Future work | Unit and recipe validation exist; broader cost/cache/latency/replay-quality eval belongs in the external eval harness. |

## Background

The AMD agentic routing recipe demonstrates a common production pattern:

- simple requests should use simple, low-cost models
- complex requests should use stronger models
- private or sensitive requests should stay on local models
- domain requests should use domain-specialized models
- agent conversations should remain stable during tool loops and cache-heavy
  continuations

An earlier recipe draft used a synthetic `agentic_session_route` decision and
`algorithm.type: session_aware` to attach this stability behavior. That worked
for a narrow demonstration, but it had the wrong abstraction boundary.

An agentic conversation can move through multiple semantic decisions. A first
turn may match `complex_code`, a follow-up tool result may match
`agentic_workflow`, and a later clarification may match `simple_general`.
If session-aware behavior is tied to a single decision-local selector, the
protection breaks whenever the next request matches a different decision with a
different `modelRefs` set.

Session-aware behavior is not itself a semantic route. It is a runtime learning
algorithm over the selected route.

## Goals

- Make session-aware routing a global learning adaptation under
  `global.router.learning.adaptations.session_aware`.
- Keep `algorithm` focused on base model selection only.
- Let learning adaptations apply by default to all decisions.
- Let individual decisions opt out when they are hard policy boundaries.
- Distinguish session and conversation identity:
  - `x-session-id` identifies the long-lived agent session or workspace.
  - `x-conversation-id` identifies one user-initiated agent conversation.
- Support both common stability modes:
  - conversation-level protection
  - session-level protection
- Define Router Learning memory as one product surface with event-log, online-state, and
  materialized-prior layers.
- Keep the user-facing API small while preserving advanced tuning knobs.
- Provide a common extension point for future router-memory learning adaptations.

## Non-Goals

- This proposal does not redesign semantic signals or projections.
- This proposal does not introduce prompt-visible user memory.
- The first implementation does not require a distributed memory backend or
  multi-replica online-state consistency.
- This proposal does not require every future learning adaptation to use
  session and conversation identity.
- This proposal does not let learning silently override decisions that opt out
  with `mode: bypass`.

## Mental Model

The final routing pipeline should be:

```text
request
  -> signals and projections
  -> decision rules
  -> base selection algorithm
  -> router learning memory read
  -> router learning adaptations
  -> final model
  -> router learning memory write
```

The ownership boundary is:

| Layer | Responsibility |
| --- | --- |
| `decision.rules` | Decide which semantic scenario matched. |
| `decision.modelRefs` | Define the base candidate set for the matched scenario. |
| `decision.algorithm` | Produce the proposed model for the matched scenario. |
| `router.learning.memory` | Store event logs, online state, and materialized priors. |
| `router.learning.adaptations` | Adjust the proposed model using router learning memory. |
| `decision.adaptations` | Decide whether this decision allows each learning adaptation to affect the final result. |

This keeps the recipe explainable:

```text
decision = complex_code
base_model = frontier-model
learning.session_aware.action = stay
learning.session_aware.reason = active_tool_loop
final_model = previous-frontier-model
```

## Global API

Router Learning is configured under `global.router.learning`. Memory and
learning adaptations live under the same product entry because adaptations are
defined by how they use cross-request memory.

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
      ttl_seconds: 2592000

  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
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
            cache_weight: 0.20
            handoff_penalty: 0.05
            handoff_penalty_weight: 1.0
            switch_history_weight: 0.04
            max_cache_cost_multiplier: 2.5
```

`adaptations` is intentionally plural because it is a collection of
cross-request learning adaptations. This avoids confusing
`router.learning.adaptations.session_aware` with decision-local
`decision.algorithm`, which remains the base selector for one request.

### Defaults and Memory Dependency

`router.learning.enabled` is the global gate for Router Learning.

If `router.learning.enabled: false`, the router ignores learning memory and
learning adaptations. Decisions route using their semantic rules and base
`algorithm` only.

If any `router.learning.adaptations.<name>.enabled: true`, Router Learning needs
memory. The router creates the low-latency online state required by enabled
adaptations. That online state is runtime state, not a new public storage
backend knob.

Online state is not a public toggle. If an enabled adaptation needs session,
conversation, provider, or feedback state, the router owns that state
internally.

Router Learning event-log data should reuse the existing Router Replay service:
`global.services.router_replay`. Its `store_backend`, TTL, and async-write
settings remain the deployment controls for durable replay records. The
route-local `router_replay` plugin remains the capture policy surface for a
decision.

Decision-level `adaptations.<name>.mode` defaults to `apply`.

### First Implementation Scope

The first implementation should focus on moving session-aware routing into
Router Learning:

- single router replica only
- no required external storage reads on the request path
- session-aware online state stored in process memory
- Router Replay used only as the event log and eval/debug source
- replay-derived priors documented as the roadmap, not required for the first
  session-aware migration
- future adaptation ordering left as an extension point, since the first
  implementation has only `session_aware`

If the required identity is missing, session-aware learning should no-op rather
than fail the request. The base routing decision remains valid, and diagnostics
can record `identity_missing` when replay/debug evidence is available.

### `session_aware.enabled`

Turns the session-aware learning adaptation on or off.

```yaml
enabled: true
```

When disabled, routing behaves like the base decision and base selection
algorithm, though other enabled learning adaptations may still run.

### `scope`

Controls the protection lifecycle.

```yaml
scope: conversation
```

Allowed values:

| Value | Meaning |
| --- | --- |
| `conversation` | Protect strongly within one conversation. Across conversations in the same session, re-evaluate routing but account for cache, history, and handoff cost. |
| `session` | Protect strongly across the whole session. The first selected model becomes the session model and later conversations prefer to keep it until the session idles out or a decision bypasses learning. |

The recommended default for agentic routing is `conversation`.

`conversation` mode gives the desired default behavior:

- a tool loop does not switch models mid-conversation
- a new conversation in the same session can route to a simpler or more
  specialized model
- cross-conversation cache and handoff cost still influence switching

`session` mode gives the stricter behavior:

- the session becomes the stability boundary
- the router strongly prefers the established session model across
  conversations
- policy boundaries can still bypass the learning layer

### `identity`

Defines how session-aware learning reads request identity.

```yaml
identity:
  headers:
    session: x-session-id
    conversation: x-conversation-id
```

Defaults:

| Field | Default |
| --- | --- |
| `headers.session` | `x-session-id` |
| `headers.conversation` | `x-conversation-id` |

The map form keeps the API extensible. `session_aware` needs `session` and
`conversation`; future adaptations can add keys such as `tenant`, `workspace`,
`user`, or `provider` without adding a new top-level header field each time.

If the conversation header is missing, the router may use explicit
missing-header resilience such as request-shape inference, but diagnostics
should mark the identity source as inferred. Production agent clients should
send both headers.

### `tuning`

The tuning block preserves the useful parts of the existing `session_aware`
selector without making them required in every recipe.

```yaml
tuning:
  idle_timeout_seconds: 300
  min_turns_before_switch: 1
  switch_margin: 0.05
  cache_weight: 0.20
  handoff_penalty: 0.05
  handoff_penalty_weight: 1.0
  switch_history_weight: 0.04
  max_cache_cost_multiplier: 2.5
```

| Field | Meaning |
| --- | --- |
| `idle_timeout_seconds` | Expire protection after inactivity. |
| `min_turns_before_switch` | Require a short warm-up period before switching. |
| `switch_margin` | Require the proposed model to be better by this margin before switching. |
| `cache_weight` | Increase stay preference when prefix-cache evidence is warm. |
| `handoff_penalty` | Default model-to-model switch cost when no learned penalty exists. |
| `handoff_penalty_weight` | Weight applied to handoff cost. |
| `switch_history_weight` | Penalize repeated model switching. |
| `max_cache_cost_multiplier` | Prevent cache preservation from justifying unbounded cost increases. |

The following existing `session_aware` knobs should not be part of the primary
user-facing API:

| Existing Field | Proposed Handling |
| --- | --- |
| `stay_bias` | Fold into `switch_margin` and internal defaults. |
| `tool_loop_stay_bias` | Keep internal; tool-loop protection should be explicit behavior, not routine tuning. |
| `quality_gap_multiplier` | Keep internal; it is tied to selector score normalization. |
| `remaining_turn_prior_weight` | Keep internal or future advanced setting. |
| `remaining_turn_prior_horizon` | Keep internal or future advanced setting. |
| `min_remaining_turn_prior_samples` | Keep internal or future advanced setting. |

Two protections should default to enabled and can later be exposed as advanced
settings if operators need them:

- tool-loop protection
- context-portability protection

## Decision API

Learning adaptation behavior can be controlled per decision.

Decision-level config uses `adaptations` directly. `router.learning` remains the
global management namespace for memory, adaptation registration, and future
learning-wide settings. A decision is narrower: it only declares how the matched
semantic route interacts with globally registered adaptations.

The default is:

```yaml
adaptations:
  session_aware:
    mode: apply
```

Most decisions do not need to write anything.

### Apply

```yaml
adaptations:
  session_aware:
    mode: apply
```

`apply` allows session-aware learning to adjust the base selection.

In `apply` mode, the learning layer may keep the current protected model even
if the current matched decision has a different `modelRefs` set. This is
intentional. It is what allows a conversation to remain stable when follow-up
turns match different semantic decisions.

The protected carry-over model must still be a configured backend model and must
not be blocked by the current decision's adaptation setting.

### Bypass

```yaml
adaptations:
  session_aware:
    mode: bypass
```

`bypass` makes the base decision result final. Session-aware learning can
record diagnostics, but it cannot change the selected model.

Use `bypass` for hard boundaries:

- privacy containment
- security containment
- explicit local-only policy
- compliance routes
- probes or operational routes where model stability should not interfere

### Observe

```yaml
adaptations:
  session_aware:
    mode: observe
```

`observe` computes what session-aware learning would have done, records it in
diagnostics and the event log, but does not change the final model.

Use `observe` for rollout, debugging, and evaluation.

### Local Overrides

Decision-level overrides should be sparse. They exist for exceptional cases:

```yaml
adaptations:
  session_aware:
    mode: apply
    scope: session
    tuning:
      switch_margin: 0.10
```

Local overrides merge with the global configuration. Unset fields inherit from
`global.router.learning.adaptations.session_aware`.

## Base Algorithm vs Learning Adaptation Boundary

The clean API should draw a hard line between request-time selection algorithms
and cross-request learning adaptations.

```text
decision.algorithm         = score or select using the current request
router.learning.adaptations = use cross-request memory, feedback, cache, cost,
                             health, or history to adjust the proposed model
```

The practical rule is:

> If it persists state across requests and that state can influence later
> routing, it belongs under `router.learning`, not under decision-local
> `algorithm`.

Algorithms can still consume read-only learning evidence. They should not own
the durable memory themselves.

### Existing Algorithm Classification

| Current Capability | Final Home | Reason |
| --- | --- | --- |
| `static` | `algorithm` | Request-local deterministic selection. |
| `hybrid` | `algorithm` | Request-local score composition. It may read learning evidence later, but should not own memory. |
| `multi_factor` | `algorithm` | Request-local quality, latency, cost, and load scoring. |
| `latency_aware` | `algorithm` | Request-local runtime metric scoring, unless it starts owning cross-request health state. |
| `router_dc` | `algorithm` | Query/model contrastive matching is a request-time selector. Learned affinity state, if added later, should be learning evidence. |
| `automix` | mostly `algorithm` | Escalation and verification are request-time selection behavior. Durable verifier success priors should move to Router Memory. |
| `knn`, `kmeans`, `svm`, `mlp` | `algorithm` | Trained model artifacts are offline selector assets, not online router memory. |
| `session_aware` | `router.learning.adaptations.session_aware` | It is cross-request continuity and stay/switch behavior. |
| `model_switch_gate` | `router.learning.adaptations.session_aware` | It is also stay/switch behavior and overlaps with session-aware routing. |
| `lookup_tables` | `router.learning.memory.priors` | They are replay-derived memory views, not selector configuration. |
| `elo` | split; ratings in Router Memory, scoring policy optional | Elo ratings are cross-request learned state. A future Elo scorer can read Router Memory. |
| `rl_driven` / Thompson | split; posterior in Router Learning memory, scoring policy optional | Bandit posterior and feedback updates are cross-request learning. |
| `gmtrouter` | split; interaction graph in Router Learning memory, scoring policy optional | User/model/task interaction history is cross-request personalization memory. |

This means `router_dc` can remain an algorithm, as long as it is doing
request-time semantic matching. If it later learns query/model affinity from
traffic, that learned affinity becomes Router Memory that `router_dc` may read.

The final public algorithm surface should keep request-time selectors and remove
learning-owned selectors:

```text
keep as algorithm:
  static, hybrid, multi_factor, latency_aware, router_dc, automix,
  knn, kmeans, svm, mlp

move out of algorithm:
  session_aware, elo, rl_driven, gmtrouter
```

`hybrid` can still combine memory-backed evidence, such as Elo ratings or
bandit priors, but those facts should be read from Router Memory. The hybrid
algorithm should not own the rating store, posterior, interaction graph, or
feedback update loop.

### Split Pattern for Learning Adaptations

Learning-capable selectors should be decomposed into:

```text
durable state/update loop -> router.learning.memory
read-time learning policy -> router.learning.adaptations.<name>
base request-time selector -> decision.algorithm
```

For example, Thompson Sampling should not keep posterior state inside
`algorithm.rl_driven`. The posterior belongs to Router Learning memory:

```yaml
global:
  router:
    learning:
      adaptations:
        bandit:
          enabled: true
          method: thompson
          reward: quality_cost_latency
          memory_scope: decision
```

Most decisions can omit `algorithm`. The bandit learning adaptation then
adjusts the model proposed by the router's normal default selector. A decision
can still use a normal base selector when it has a specific selection policy,
but the adaptation is not a reason to force every decision to spell out a base
algorithm.

Elo follows the same model:

```yaml
global:
  router:
    learning:
      adaptations:
        elo:
          enabled: true
          scope: decision
          initial_rating: 1200
          k_factor: 32
```

The ratings are router memory. A read-time Elo scoring policy can consume those
ratings, but config should not imply that a decision-local algorithm owns the
rating store.

## Router Memory Product Model

Router Memory should be the product-level umbrella for replay, live state, and
derived priors. Users should not need to reason about three independent systems.

```text
Router Memory
  event_log
    replay records
    immutable or append-oriented history for audit, debug, eval, and replay

  online_state
    session state, conversation state, provider health, bandit posterior,
    Elo ratings, personalization graph

  priors
    materialized views derived from event_log and optional overrides:
    quality_gap, handoff_penalty, remaining_turn_prior
```

The product relationship is:

```text
requests and responses
  -> Router Memory event_log
  -> aggregation / outcome scoring
  -> Router Memory online_state and priors
  -> Router Learning consumers
```

Replay is the event-log layer. Session and conversation memory are online-state
views. The old lookup tables become materialized priors. They are all Router
Memory.

### Latency Model

Router Learning must not turn every routed request into an external storage
round trip. The memory layers have different latency contracts:

| Layer | Hot Request Path | Storage Path | Examples |
| --- | --- | --- | --- |
| `online_state` | Read and update locally during routing. This is latency-sensitive. | Optional shared backend or replay-derived rebuild, but not required for every request. | current model, tool-loop state, turn count, cache warmth, switch history |
| `priors` | Read from an in-process or locally cached snapshot. | Recomputed from replay by background jobs or offline commands. | quality gaps, handoff penalties, remaining-turn priors |
| `event_log` | Append after the routing decision, preferably async or fail-open. Reads are not required for the current decision. | Durable Router Replay backend. | replay records, diagnostics, eval traces |

The hot path should therefore be:

```text
request
  -> read online_state from local memory
  -> read priors from local snapshot
  -> compute adaptations
  -> update online_state locally
  -> append replay event asynchronously or fail-open
```

External storage calls are appropriate for:

- replay writes and replay APIs
- background prior materialization
- cold-start seeding of online state or prior snapshots
- optional shared online-state synchronization when an operator chooses that
  deployment mode

External storage calls should not be required to decide every request. If a
shared online-state backend is introduced, it needs a small timeout,
fail-open behavior, and local fallback so storage latency or transient outages
do not become routing latency spikes.

### Memory API

Router Memory is the product model, but the public API should reuse existing
storage seams instead of introducing a second replay backend.

The existing Router Replay implementation already provides the event-log
building blocks:

- `global.services.router_replay.enabled` is the router-wide replay gate.
- `store_backend` supports `postgres`, `redis`, `milvus`, `qdrant`, and
  `memory`.
- Shared backends use one shared recorder/store; `memory` is local development
  storage and is lost on restart.
- The route-local `router_replay` plugin can disable capture or tune capture
  policy for one decision.
- Replay APIs already expose list, aggregate, record lookup, and trajectory
  views.

Router Learning should therefore attach learning diagnostics to Router Replay
records instead of inventing a parallel event-log service. Replay remains the
append-oriented history used for audit, eval, and prior materialization; it is
not the low-latency online state that decides the next request.

The event-log layer is Router Replay:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
      ttl_seconds: 2592000

  router:
    learning:
      enabled: true
      memory:
        priors:
          source: router_replay
          populate_interval: 15m
          overrides:
            quality_gaps: []
            handoff_penalties: []
            remaining_turn_priors: []
```

The default recipe can keep Router Learning itself simple:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres

  router:
    learning:
      enabled: true
```

`global.services.router_replay` already owns the replay storage backend. It
supports `postgres`, `redis`, `milvus`, `qdrant`, and `memory`. Production
recipes should use a shared durable backend such as `postgres` or `redis`;
`memory` remains a local development backend and loses records on restart.

Router Learning should attach structured learning diagnostics to replay records
when Router Replay captures the request. It should not expose a second
`router.learning.memory.replay.enabled` flag, because that would duplicate the
existing router-wide service and route-local plugin.

The existing `router_replay` plugin remains the per-decision capture policy.
Learning does not need raw payload fields, but Router Learning should not add a
second privacy policy on top of Router Replay. If operators enable payload
capture, payloads may be persisted according to the existing Router Replay
configuration. Learning evals should ignore those payload fields and use
structured routing metadata instead.

Privacy-sensitive recipes can still use the existing plugin flags when they
want metadata-only replay records:

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      capture_request_body: false
      capture_response_body: false
      max_tool_trace_steps: 100
```

Learning diagnostics, selected model, decision, signal names, token counts,
cache counts, and cost estimates are enough for the first eval path.

`online_state` is intentionally not configurable. It is required runtime state
for enabled adaptations, so exposing `online_state.enabled` would let users
create invalid or confusing configs.

`priors` is optional and advanced. It exists for replay-derived materialized
views and manual overrides. `source: router_replay` means the priors are built
from the same event log exposed by `/v1/router_replay`. If omitted,
adaptations still work from live online state and built-in defaults.

When memory is configured, learning adaptations can consume the available layers
by default.
For example, `session_aware` reads conversation state, session state, cache
accounting, and materialized priors when those views exist.

### Memory Views

Initial Router Memory should expose at least these views:

| View | Layer | Source | Consumers |
| --- | --- | --- | --- |
| `replay_record` | `event_log` | request/response records | audit, eval, replay, priors |
| `session_state` | `online_state` | request/response telemetry | `session_aware` |
| `conversation_state` | `online_state` | request/response telemetry | `session_aware` |
| `provider_health` | `online_state` | transport outcomes | `provider_health`, `multi_factor`, `hybrid` |
| `elo_rating` | `online_state` | feedback/outcome updates | `elo`, `hybrid` |
| `bandit_posterior` | `online_state` | feedback/outcome updates | `bandit`, `hybrid` |
| `interaction_graph` | `online_state` | feedback, user/session history | `personalization`, future GMTRouter policy |
| `quality_gap` | `priors` | replay aggregation, outcomes, overrides | `session_aware`, `hybrid`, `bandit` |
| `handoff_penalty` | `priors` | replay aggregation, overrides | `session_aware` |
| `remaining_turn_prior` | `priors` | replay aggregation, overrides | `session_aware` |

The important part is that these are shared memory views. They should not be
private fields inside individual algorithms.

### Lookup Table Migration

The current `lookup_tables` block is useful conceptually but sits in the wrong
place:

```yaml
global:
  router:
    model_selection:
      lookup_tables:
        enabled: true
```

It is not a model-selection algorithm. It is the old name for materialized
Router Memory priors.

| Old Field | New Field |
| --- | --- |
| `model_selection.lookup_tables.enabled` | `router.learning.memory.priors.enabled` |
| `model_selection.lookup_tables.storage_path` | Implementation detail removed from public config. |
| `model_selection.lookup_tables.auto_save_interval` | Implementation detail removed from public config. |
| `model_selection.lookup_tables.populate_from_replay` | `router.learning.memory.priors.source: router_replay` |
| `model_selection.lookup_tables.populate_interval` | `router.learning.memory.priors.populate_interval` |
| `model_selection.lookup_tables.quality_gaps` | `router.learning.memory.priors.overrides.quality_gaps` |
| `model_selection.lookup_tables.handoff_penalties` | `router.learning.memory.priors.overrides.handoff_penalties` |
| `model_selection.lookup_tables.remaining_turn_priors` | `router.learning.memory.priors.overrides.remaining_turn_priors` |

Manual overrides remain useful, but they become overrides for Router Memory
priors, not selector-local tables.

### Online State vs Priors

Online state and priors are different layers of Router Memory.

| Memory Layer | Scope | Purpose |
| --- | --- | --- |
| `online_state` | live session, conversation, provider, user, or decision | Protect current continuity, cache, tool loops, health, preference, and switch history. |
| `priors` | aggregate traffic history | Provide learned quality gaps, handoff penalties, and expected remaining turns. |

Session-aware learning should read both:

```text
conversation_state says: active tool loop, keep current model
session_state says: cache is warm, switching has been frequent
priors say: switching from local to frontier has handoff penalty 0.05
```

This lets the runtime make an auditable stay/switch decision without putting
memory ownership inside `algorithm.type: session_aware`.

## Example Recipe Shape

### Global Learning

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres

  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
          scope: conversation
          identity:
            headers:
              session: x-session-id
              conversation: x-conversation-id
```

### Simple Decision

```yaml
- name: simple_general
  description: Simple public requests route to the low-cost model by default.
  rules: ...
  modelRefs:
    - model: simple-model
    - model: frontier-model
```

This decision inherits `mode: apply` and uses the router's normal default
selector before adaptation.

### Complex Decision

```yaml
- name: complex_code
  description: Complex coding requests route to stronger models.
  rules: ...
  modelRefs:
    - model: frontier-model
    - model: local-rocm-model
```

This decision also inherits `mode: apply` and does not need an explicit
`algorithm` block.

If the current conversation is in a tool loop, the learning layer can keep the
current model even if the new request matches `simple_general`.

### Privacy Decision

```yaml
- name: privacy_sensitive
  description: Sensitive content stays on the local model.
  rules: ...
  modelRefs:
    - model: local-rocm-model
  algorithm:
    type: static
  adaptations:
    session_aware:
      mode: bypass
```

This decision cannot be overridden by session-aware stability. If privacy
matches, the final model remains local.

## Runtime Semantics

### Conversation Scope

With:

```yaml
scope: conversation
```

the router maintains two related memories:

```text
conversation key = session_id + conversation_id
session key      = session_id
```

Strong protections use the conversation key:

- active tool loop
- non-portable provider state
- current conversation model
- short-turn continuity

Soft trade-offs can use the session key:

- previous session model
- cross-conversation cache evidence
- switch history
- handoff cost

This means a new `x-conversation-id` releases the previous conversation's hard
locks, while still letting session-level evidence influence the next selection.

### Session Scope

With:

```yaml
scope: session
```

the session key becomes the strong protection boundary. A model established in
the session is preferred across conversations. The session model is the first
selected model after the session state is created or after the session has
expired.

The router can reselect the session model when one of these happens:

- the session is idle past `idle_timeout_seconds`
- the current decision uses `mode: bypass`
- an explicit future reset mechanism clears the session state

This mode is for users who want a stable model throughout an IDE session,
workspace session, or long agent session.

### Policy Boundaries

`mode: bypass` always wins over session-aware learning.

For example, if the current session model is a cloud frontier model and a later
request matches privacy containment, the privacy decision routes to the local
model and session-aware learning cannot keep the cloud model.

## Diagnostics and Event Log

Response headers should stay compact. Detailed evidence belongs in Router
Memory's event-log layer. The current replay record is the concrete event-log
implementation.

Recommended response headers:

```text
x-vsr-learning-methods: session_aware
x-vsr-learning-actions: session_aware=stay
x-vsr-learning-scopes: session_aware=conversation
x-vsr-learning-reasons: session_aware=active_tool_loop
x-vsr-learning-modes: session_aware=apply
x-vsr-replay-id: replay_...
```

The `x-vsr-learning-*` header family is method-keyed so future adaptations can
share the same contract without adaptation-specific header names. Detailed
scoring evidence, cache math, identity source, multi-adaptation traces, and
alternative-model candidates should remain in the replay record pointed to by
`x-vsr-replay-id`.

The event log should include structured diagnostics:

```json
{
  "learning": {
    "adaptations": {
      "session_aware": {
        "enabled": true,
        "mode": "apply",
        "scope": "conversation",
        "identity": {
          "scope": "conversation",
          "headers": {
            "session": "x-session-id",
            "conversation": "x-conversation-id"
          },
          "session": {
            "source": "header:x-session-id",
            "required": true,
            "status": "present",
            "hash": "4f2a8c0e9b7d3411"
          },
          "conversation": {
            "source": "header:x-conversation-id",
            "required": true,
            "status": "present",
            "hash": "0bb97f4a3c812efe"
          }
        },
        "base_model": "simple-model",
        "final_model": "frontier-model",
        "action": "stay",
        "reason": "active_tool_loop",
        "cache": {
          "prompt_tokens": 12000,
          "cached_tokens": 8200,
          "cache_weight": 0.2
        },
        "cost": {
          "handoff_penalty": 0.05,
          "handoff_penalty_weight": 1.0
        }
      }
    }
  }
}
```

Raw session and conversation identifiers are operationally sensitive. Learning
diagnostics should store source and status plus a bounded hash, not the raw
identity values.

## Breaking Change and Migration

This proposal intentionally uses a clean breaking change. The old session-aware
configuration should not remain as a compatibility alias in the final API.
Keeping both spellings would make it unclear whether session-aware behavior is a
selector, a post-selection gate, or a Router Learning adaptation.

### Removed API

Old recipes may currently use:

```yaml
algorithm:
  type: session_aware
  session_aware:
    base_method: hybrid
```

The final API should reject this shape.

These old blocks should also be rejected:

```yaml
global:
  router:
    model_selection:
      session_aware: ...
      model_switch_gate: ...
      lookup_tables: ...
      elo: ...
```

`model_selection.session_aware` and `model_switch_gate` are older ways to
express stay-vs-switch behavior. In the final API, that behavior belongs under
`global.router.learning.adaptations.session_aware`.

`model_selection.lookup_tables` is a replay-derived memory view. It belongs
under `global.router.learning.memory.priors`.

Learning-owned algorithm types should also be rejected as final public
adaptations:

```yaml
algorithm:
  type: elo
```

```yaml
algorithm:
  type: rl_driven
```

```yaml
algorithm:
  type: gmtrouter
```

Their durable state and feedback loops should move to `router.learning.adaptations.elo`,
`router.learning.adaptations.bandit`, and `router.learning.adaptations.personalization`
respectively.

### Replacement API

The target shape is a normal decision selector plus global learning. Most
decisions can omit `algorithm` and use the router's normal default selector:

```yaml
- name: simple_general
  rules: ...
  modelRefs:
    - model: simple-model
    - model: frontier-model
```

A decision can keep an explicit base algorithm only when it needs a specific
selector. The important part is that `session_aware` is no longer an algorithm
wrapper.

Session-aware behavior is configured globally:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres

  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
          scope: conversation
```

If an old `session_aware` block had `base_method`, that value can become the
decision's explicit base `algorithm.type` only when the recipe wants to preserve
that explicit selector. If the decision can rely on the default selector, it can
omit `algorithm` entirely.

### Migration Map

| Old Field | New Field |
| --- | --- |
| `global.router.model_selection.session_aware` | `global.router.learning.adaptations.session_aware` |
| `algorithm.type: session_aware` | remove; use a normal base `algorithm` only if needed |
| `algorithm.session_aware.base_method` | optional explicit base `algorithm.type`; otherwise omit `algorithm` |
| `algorithm.session_aware.idle_timeout_seconds` | `router.learning.adaptations.session_aware.tuning.idle_timeout_seconds` |
| `algorithm.session_aware.min_turns_before_switch` | `router.learning.adaptations.session_aware.tuning.min_turns_before_switch` |
| `algorithm.session_aware.switch_margin` | `router.learning.adaptations.session_aware.tuning.switch_margin` |
| `algorithm.session_aware.prefix_cache_weight` | `router.learning.adaptations.session_aware.tuning.cache_weight` |
| `algorithm.session_aware.default_handoff_penalty` | `router.learning.adaptations.session_aware.tuning.handoff_penalty` |
| `algorithm.session_aware.handoff_penalty_weight` | `router.learning.adaptations.session_aware.tuning.handoff_penalty_weight` |
| `algorithm.session_aware.switch_history_weight` | `router.learning.adaptations.session_aware.tuning.switch_history_weight` |
| `algorithm.session_aware.max_cache_cost_multiplier` | `router.learning.adaptations.session_aware.tuning.max_cache_cost_multiplier` |
| `model_switch_gate.min_switch_advantage` | `router.learning.adaptations.session_aware.tuning.switch_margin` |
| `model_switch_gate.cache_warmth_weight` | `router.learning.adaptations.session_aware.tuning.cache_weight` |
| `model_switch_gate.default_handoff_penalty` | `router.learning.adaptations.session_aware.tuning.handoff_penalty` |
| `model_switch_gate.mode: shadow` | `decision.adaptations.session_aware.mode: observe` for rollout decisions |
| `model_switch_gate.mode: enforce` | `decision.adaptations.session_aware.mode: apply` |
| `model_selection.lookup_tables` | `router.learning.memory.priors` |
| `algorithm.type: elo` | `router.learning.adaptations.elo`; add a base `algorithm` only if the decision needs one |
| `model_selection.elo` | `router.learning.adaptations.elo` |
| `algorithm.type: rl_driven` | `router.learning.adaptations.bandit`; add a base `algorithm` only if the decision needs one |
| `algorithm.rl_driven.use_thompson_sampling` | `router.learning.adaptations.bandit.method: thompson` |
| `algorithm.type: gmtrouter` | `router.learning.adaptations.personalization`; add a base `algorithm` only if the decision needs one |
| `algorithm.gmtrouter.storage_path` | `router.learning.memory` backend configuration |

These old fields should not have public replacements in the clean API:

| Old Field | Handling |
| --- | --- |
| `stay_bias` | Fold into `switch_margin` and internal defaults. |
| `tool_loop_stay_bias` | Internal behavior; tool-loop protection should not require score tuning. |
| `tool_loop_hard_lock` | Default enabled in session-aware learning. |
| `context_portability_hard_lock` | Default enabled in session-aware learning. |
| `decision_drift_reset` | Replaced by explicit conversation identity. |
| `quality_gap_multiplier` | Internal selector-normalization detail. |
| `remaining_turn_prior_weight` | Internal or future advanced setting. |
| `remaining_turn_prior_horizon` | Internal or future advanced setting. |
| `min_remaining_turn_prior_samples` | Internal or future advanced setting. |

### Validation Behavior

The router should fail fast with actionable validation errors when it sees old
configuration:

```text
algorithm.type=session_aware has moved to global.router.learning.adaptations.session_aware.
Remove algorithm.type=session_aware. If the recipe needs an explicit base
selector, set algorithm.type to the old session_aware.base_method; otherwise
omit algorithm and enable global router.learning.adaptations.session_aware.
```

```text
global.router.model_selection.session_aware has moved to
global.router.learning.adaptations.session_aware.
```

```text
global.router.model_selection.model_switch_gate has been folded into
global.router.learning.adaptations.session_aware.tuning.
```

```text
algorithm.type=elo has moved to router learning. Enable
global.router.learning.adaptations.elo and choose a request-time base algorithm.
```

```text
algorithm.type=rl_driven has moved to router learning. Enable
global.router.learning.adaptations.bandit and choose a request-time base algorithm.
```

```text
algorithm.type=gmtrouter has moved to router learning. Enable
global.router.learning.adaptations.personalization and choose a request-time base
algorithm.
```

```text
global.router.model_selection.lookup_tables has moved to
global.router.learning.memory.priors.
```

Runtime config loading should not silently rewrite old config, and the first
implementation should not provide an automatic config rewrite command. Migration
is a deliberate manual edit from the old API shape to the new one. Silent or
automatic rewrites would preserve ambiguity in the API.

## Implementation Plan

### 1. Config Contract

- Add `global.router.learning`.
- Add `global.router.learning.adaptations.session_aware`.
- Add decision-level `adaptations.session_aware`.
- Remove `global.router.model_selection.session_aware` from the final public
  config contract.
- Remove `algorithm.type: session_aware` from the final public config contract.
- Fold `model_switch_gate` into session-aware learning and reject the old
  standalone block.
- Reject `lookup_tables` from `global.router.model_selection`; replay-derived
  priors move to the Router Learning roadmap.
- Reuse `global.services.router_replay` as the Router Memory event-log
  implementation.
- Add `global.router.learning.memory.priors` only for replay-derived materialized
  priors and manual overrides in the roadmap; the first implementation can omit
  prior materialization.
- Implement required online state internally for enabled adaptations.
- Validate `scope` values: `conversation`, `session`.
- Validate `mode` values: `apply`, `bypass`, `observe`.
- Emit hard validation errors for old memory, learning, and session-aware config
  shapes with explicit migration messages.
- Add defaults:
  - `router.learning.enabled: false` unless a recipe enables it
  - enabled learning adaptations create required online state automatically
  - Router Learning diagnostics attach to Router Replay records when
    `global.services.router_replay.enabled: true`
  - Router Replay storage defaults and backends remain owned by
    `global.services.router_replay`
  - `scope: conversation`
  - `identity.headers.session: x-session-id`
  - `identity.headers.conversation: x-conversation-id`
  - decision mode defaults to `apply`

The final architecture still classifies `elo`, `rl_driven`, and `gmtrouter` as
learning-owned capabilities. Moving them requires new feedback, posterior, and
personalization adaptation APIs, so that migration is future Router Learning
work rather than part of this session-aware implementation.

### 2. Request Identity

- Parse identity values from `identity.headers`.
- For `session_aware`, require the `session` key and use the `conversation` key
  when present.
- Treat configured identity header names as the source of truth.
- If identity is missing, use only explicit fallback behavior and mark the
  identity source as missing in diagnostics when diagnostics are captured.
- If the required identity is missing, session-aware learning should no-op and
  allow the base routing decision to proceed.
- Store identity source in diagnostics.
- Treat conversation identity as distinct from session identity throughout
  request context, selection context, learning trace, and event-log records.

### 3. Learning Runtime

- Run decision matching as today.
- Run the decision's base `algorithm` to produce a proposed model.
- Apply enabled Router Learning adaptations after base selection.
- For session-aware learning:
  - read Router Memory `online_state`
  - read `priors` when a local snapshot exists
  - read conversation state and session state from `online_state`
  - evaluate the decision's adaptation mode
  - compute keep/switch/observe
  - produce a final model and structured trace
- Allow protected carry-over of the previous model across decision modelRef
  boundaries only when mode is `apply`.
- Never override a decision with `mode: bypass`.

### 4. Router Memory

- Keep the request hot path free of required external storage reads.
- Read `online_state` from in-process state or a local fast cache.
- Read `priors` from an in-process snapshot refreshed outside the request path.
- Treat shared online-state backends as optional synchronization layers with
  timeout and fail-open behavior, not mandatory per-request dependencies.
- Use existing Router Replay records as the durable event stream for audit,
  debugging, eval, and prior generation.
- Extend replay records with generic learning diagnostics instead of adding a
  parallel learning event store.
- Reuse existing Router Replay storage backends and API surfaces:
  - `store_backend: postgres|redis|milvus|qdrant|memory`
  - `/v1/router_replay`
  - `/v1/router_replay/aggregate`
  - `/v1/router_replay/trajectory`
  - route-local `router_replay` plugin capture policy
- Do not make learning depend on replay payload fields. Request bodies,
  response bodies, prompts, tool schemas, tool arguments, and tool outputs are
  controlled by the existing Router Replay capture policy.
- Keep replay writes off the critical path where possible. Async writes should
  have bounded queues, backpressure metrics, and safe drop behavior for
  diagnostics when the replay backend is unhealthy.
- Implement first-version `online_state` as single-replica in-process state:
  - conversation-scoped state keyed by session ID and conversation ID
  - session-scoped state keyed by session ID
- Record session-aware online state:
  - previous model
  - turn count
  - active tool-loop state
  - non-portable context state
  - switch history
  - cache accounting
  - last decision and last learning reason
- Leave `priors` as roadmap materialized views derived from `event_log`:
  - quality gaps
  - handoff penalties
  - remaining-turn priors
- Future implementations can refresh prior snapshots through a background
  materializer or offline command.
- Expire online state by idle timeout or configured TTL.

### 5. Event Log and Headers

- Add structured learning diagnostics to Router Replay records.
- Add compact generic response headers for bounded adaptation summaries:
  `x-vsr-learning-methods`, `x-vsr-learning-actions`,
  `x-vsr-learning-scopes`, `x-vsr-learning-reasons`, and
  `x-vsr-learning-modes`.
- Keep `x-vsr-replay-id` as the stable pointer to full diagnostics.
- Include cache details needed to debug cache protection:
  - prompt tokens
  - cached tokens
  - cache source
  - estimated cache gap where exact backend accounting is unavailable

### 6. Recipe Migration

- Remove the synthetic `agentic_session_route`.
- Configure `global.services.router_replay` when eval/debug event logs are
  required.
- Configure global `router.learning.adaptations.session_aware`.
- Keep simple, complex, privacy, and domain decisions as semantic decisions.
- Mark privacy, security, and local-only policy decisions with
  `mode: bypass`.
- Leave normal simple, complex, and domain decisions on the default
  `mode: apply`.

### 7. Evaluation

- Update the agentic routing eval profile to send configured identity headers
  for session and conversation.
- Capture `x-vsr-learning-*` and `x-vsr-replay-id` from router responses.
- Validate semantic routing:
  - simple requests route to the simple model
  - complex requests route to the stronger model
  - privacy-sensitive requests route to the local model
  - domain requests route to the domain model
- Validate conversation scope:
  - same session and same conversation protects tool-loop continuity
  - same session and new conversation releases tool-loop hard locks
  - same session and new conversation still accounts for cache and handoff cost
- Validate session scope:
  - same session strongly preserves the established model across conversations
  - bypass decisions still override session protection
- Validate Router Memory event log:
  - every learning adaptation has action, scope, reason, base model, and
    final model
  - cache accounting appears when available
  - observe mode records the hypothetical result without changing the final
    model
- Report cache and cost evidence:
  - cached tokens
  - cache gap
  - estimated cost delta from auto routing and cache preservation
- Report learning latency overhead at p50 and p95.

### 8. Documentation

- Add a dedicated tutorial section under `docs/tutorials/learning/`.
- Use the section as the user-facing home for Router Learning, Router Memory,
  and learning adaptations.
- Start with these pages:
  - `docs/tutorials/learning/overview.md`
  - `docs/tutorials/learning/session-aware.md`
  - `docs/tutorials/learning/memory-and-replay.md`
  - `docs/tutorials/learning/decision-adaptations.md`
  - `docs/tutorials/learning/priors.md`
- Keep future pages in the same section as new adaptations graduate:
  - `docs/tutorials/learning/bandit.md`
  - `docs/tutorials/learning/personalization.md`
  - `docs/tutorials/learning/provider-health.md`
- Cover these concepts in the tutorials:
  - why Router Learning is separate from `decision.algorithm`
  - how `router.learning.memory` works at a product level
  - how `router.learning.adaptations.<name>` registers global behavior
  - how `decision.adaptations.<name>.mode` controls hard boundaries
  - session vs conversation identity and headers
  - `apply`, `bypass`, and `observe`
  - replay records, diagnostics, and cache evidence
  - how to evaluate a learning adaptation before enabling it broadly
- Cross-link the tutorials from:
  - the AMD agentic routing recipe
  - the router config reference
  - any agentic routing blog post
  - future adaptation-specific proposal or reference pages

## Future Learning Extensions

`router.learning.adaptations` is intentionally broader than session-aware routing. Future
router-memory adaptations should use the same Router Memory plus
global learning plus decision override model.

Example:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres

  router:
    learning:
      enabled: true
      memory:
        priors:
          source: router_replay
          populate_interval: 15m

      adaptations:
        session_aware:
          enabled: true
          scope: conversation

        bandit:
          enabled: false
          exploration_rate: 0.05

        personalization:
          enabled: false

        provider_health:
          enabled: false
```

Decision-level controls can remain consistent:

```yaml
adaptations:
  session_aware:
    mode: apply
  bandit:
    mode: observe
  personalization:
    mode: bypass
```

The common contract is:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether the learning adaptation exists globally. |
| `mode` | Whether a decision allows that learning adaptation to affect the final result. |
| `observe` | Safe rollout mode that records evidence without changing routing. |
| diagnostics | Every learning adaptation must explain base result, final result, action, and reason. |

Potential future learning adaptations:

| Learning Adaptation | Purpose |
| --- | --- |
| `bandit` | Online exploration and exploitation from feedback or judged outcomes. |
| `personalization` | User or workspace preference learning. |
| `provider_health` | Avoid unhealthy model endpoints or providers. |
| `cost_optimizer` | Learn cost/performance trade-offs from actual traffic. |
| `quality_feedback` | Adjust routing from explicit feedback or automatic judges. |

These should not require new top-level decision fields. They should plug into
`global.router.learning.adaptations.<name>` and `decision.adaptations.<name>`.

## Confirmed Decisions

- The public namespace is `global.router.learning.memory` and
  `global.router.learning.adaptations.<name>`.
- Session-aware identity lives under
  `global.router.learning.adaptations.session_aware.identity`.
- Session-aware identity uses a generic map:
  `identity.headers.session` and `identity.headers.conversation`, rather than
  one hard-coded config field per header.
- Decision-level control uses top-level `decision.adaptations.<name>` and
  mirrors only the adaptation names registered under
  `global.router.learning.adaptations`.
- The default decision mode is `apply`.
- Hard policy boundaries explicitly use `mode: bypass`.
- Unknown `decision.adaptations.<name>` entries should fail validation unless
  the global adaptation exists.
- `scope: conversation` protects the active conversation and uses session memory
  only for soft trade-offs between conversations.
- `scope: session` preserves the first selected model for the session until the
  session idles out, a bypass decision wins, or an explicit future reset clears
  the session state.
- In `mode: apply`, a protected carry-over model may cross the current
  decision's `modelRefs` boundary. This is required for stable multi-decision
  agent conversations.
- The migration is a one-time breaking change. Old API shapes should fail
  validation with actionable errors rather than silently rewrite.
- Router Memory event-log records should reuse the existing Router Replay
  service, storage backends, route-local plugin, and replay API.
- Response diagnostics should use a small generic `x-vsr-learning-*` header
  family plus the existing `x-vsr-replay-id` pointer. Adaptation-specific
  header names should not become the stable API.
- The first implementation targets a single router replica.
- Missing identity should no-op session-aware learning without failing the
  request.
- Router Learning should not depend on raw replay payload fields. Payload
  persistence is governed by existing Router Replay capture configuration.
- New replay diagnostics use `learning.adaptations.<name>` only; the new
  implementation should not compatibility-fill older session-aware-specific
  replay fields.
- Replay-derived priors are roadmap work. The first implementation focuses on
  moving session-aware routing into Router Learning.

## Closed Decisions For First Implementation

### 1. Hot Path

Request-time routing must not require synchronous external storage reads.
Session-aware learning reads and writes in-process online state. Replay writes
are off the critical path and should fail open.

### 2. Deployment Scope

The first implementation targets a single router replica. Multi-replica online
state, shared Redis state, sticky routing requirements, and cross-replica
consistency are roadmap items.

### 3. Missing Identity

Missing session or conversation identity should not fail the request. If the
identity required for a session-aware protection is absent, the adaptation
no-ops and the base routing decision proceeds. Diagnostics may record the
missing identity when replay/debug evidence is captured.

### 4. Replay Payloads

Router Learning should not add a second payload persistence policy. Payload
storage is governed by the existing Router Replay capture configuration. If an
operator enables payload capture, payloads may be stored; Router Learning and
its evals simply do not use those payload fields.

### 5. Replay Schema

Use the new generic diagnostics shape:
`learning.adaptations.<name>`. Do not compatibility-fill older
session-aware-specific replay fields for the new implementation.

### 6. Priors Roadmap

Replay-derived priors remain in the design and roadmap, but they are not part of
the first implementation. The first implementation focuses on moving
session-aware routing into Router Learning.

### 7. Future Adaptation Ordering

The first implementation has only `session_aware`, so it does not need public
ordering controls. The design should leave room for future weighting,
priorities, or phases when multiple adaptations can modify the same decision.

### 8. Eval Contract

The agentic routing eval should cover semantic routing, session-aware stability,
new-conversation release, bypass behavior, cache evidence, learning latency
overhead, and replay explainability.

## Migration Tooling Decision

Do not provide an automatic config rewrite command in the first implementation.
This is a clean new API, not a compatibility layer. Old config shapes should
fail fast with actionable validation errors and documentation should show the
new shape.

## Success Criteria

- A recipe enables session-aware behavior once globally.
- Normal decisions do not need per-decision configuration.
- Hard policy decisions can bypass learning with one small block.
- Conversation scope and session scope are both expressible.
- A new conversation in the same session does not inherit tool-loop hard locks
  from the previous conversation.
- Session scope can express "keep the session's established model."
- Event-log records and headers explain every keep, switch, bypass, and observe
  result.
- Request-time routing does not require synchronous external storage reads.
- Replay backend slowness or outage does not fail the routing request.
- Missing session or conversation identity does not fail the request.
- The first implementation is valid for a single router replica.
- Future learning adaptations can be added without adding new decision-level
  concepts.
