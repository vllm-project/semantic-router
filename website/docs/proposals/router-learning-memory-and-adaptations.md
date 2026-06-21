# Router Learning: Memory and Adaptations

## Status

Accepted. The current implementation covers session-aware Router Learning with
conversation/session scopes, decision-level `apply` / `observe` / `bypass`,
generic learning headers, replay diagnostics, adaptation composition, the AMD
agentic routing recipe, and day-0 Router Learning migrations for bandit, Elo,
and personalization. Broader Router Learning memory features such as public
experience materializers, distributed states, and richer multi-adaptation
arbitration remain future work.

## Summary

The router needs one product concept for cross-request routing intelligence:
**Router Learning**.

Router Learning has four product layers:

- **Replay** records what happened as a durable event log.
- **States** keep the live routing facts needed by active adaptations.
- **Experience** materializes historical routing evidence from replay, evals,
  and operator overrides.
- **Adaptations** consume states and experience to adjust a base routing
  decision.

These layers unify concepts that were previously described separately:

- replay records as the event log
- session and conversation memory as live states
- lookup tables as legacy materialized experience
- learning-style selectors as adaptations plus shared states and experience

The primary production learning adaptation is session-aware routing stability.
It decides when to keep the current model and when to switch based on
conversation continuity, session continuity, prefix-cache evidence, handoff
cost, and model-switch history. Day-0 bandit, Elo, and personalization
adaptations use the same Router Learning states and feedback path.

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
| Session/conversation states | Implemented | First implementation uses low-latency in-process router states and single-replica assumptions. |
| Bandit day-0 adaptation | Implemented | `bandit` supports `linucb` and `linear_thompson` API values, decision-scoped local states, explicit feedback rewards, cost goals, and conservative no-op behavior when states are missing. |
| Generic learning response headers | Implemented | Uses `x-vsr-learning-methods`, `x-vsr-learning-actions`, `x-vsr-learning-scopes`, `x-vsr-learning-reasons`, and `x-vsr-learning-modes`. |
| Replay learning diagnostics | Implemented, partial | Records method/action/reason/scope, base/final model evidence, cache/cost evidence, and hashed identity diagnostics. The exact replay schema can evolve. |
| Clean break from old public learning-style algorithm config | Implemented | Old `session_aware`, `elo`, `rl_driven`, and `gmtrouter` algorithm types, old learning blocks, `model_switch_gate`, and public `lookup_tables` config are rejected instead of rewritten. |
| AMD agentic routing recipe and guide | Implemented | Recipe uses session-aware Router Learning plus decision bypasses for hard privacy/security boundaries. |
| Router Learning states contract | Implemented, internal | States are local/in-process and exposed to API-server feedback through a narrow runtime interface. No public states backend, TTL, or storage controls are exposed. |
| Router Learning experience | Implemented, minimal | Replay diagnostics include method-level experience status such as missing/used. Materializers from replay, evals, and overrides remain future work. |
| Adaptation composition | Implemented, minimal | The composer records every adaptation, hard changes win over soft changes, observe does not change the final route, and headers/replay stay method-keyed. |
| Stateful algorithm migration | Implemented, day-0 | Old public algorithm config is rejected. `bandit`, `elo`, and `personalization` have Router Learning states, shared feedback updates, and conservative fail-open routing behavior. |
| Distributed states semantics | Future work | Multi-replica states semantics, storage hot-path policy, and sticky routing are out of scope. |
| Architecture eval fixtures | Implemented | Deterministic fixtures report route correctness, base/final model, adaptation method/mode/action/reason/scope, cache/cost delta, bypass behavior, replay explainability coverage, and p50/p95 overhead. Live AMD validation remains a deployment gate. |

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
- Define Router Learning memory as one product surface with replay, states, and
  experience layers.
- Keep the user-facing API small while preserving advanced tuning knobs.
- Provide a common extension point for future router-memory learning adaptations.

## Non-Goals

- This proposal does not redesign semantic signals or projections.
- This proposal does not introduce prompt-visible user memory.
- The current implementation does not require a distributed memory backend or
  multi-replica states consistency.
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
  -> router learning states and experience read
  -> router learning adaptations
  -> final model
  -> router learning replay and states write
```

The ownership boundary is:

| Layer | Responsibility |
| --- | --- |
| `decision.rules` | Decide which semantic scenario matched. |
| `decision.modelRefs` | Define the base candidate set for the matched scenario. |
| `decision.algorithm` | Produce the proposed model for the matched scenario. |
| `router.learning.replay` | Durable event log through Router Replay. This remains configured by `global.services.router_replay`. |
| `router.learning.states` | Live session, conversation, provider, feedback, or personalization state owned by enabled adaptations. |
| `router.learning.experience` | Materialized historical evidence from replay, evals, and overrides. |
| `router.learning.adaptations` | Adjust the proposed model using Router Learning states and experience. |
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
memory. The router creates the low-latency states required by enabled
adaptations. Those states are runtime memory, not a new public storage backend knob.

States are not a public toggle. If an enabled adaptation needs session,
conversation, provider, or feedback states, the router owns those states
internally.

Router Learning event-log data should reuse the existing Router Replay service:
`global.services.router_replay`. Its `store_backend`, TTL, and async-write
settings remain the deployment controls for durable replay records. The
route-local `router_replay` plugin remains the capture policy surface for a
decision.

Decision-level `adaptations.<name>.mode` defaults to `apply`.

### Current Implementation Scope

The current implementation keeps request-time learning local and fail-open:

- single router replica only
- no required external storage reads on the request path
- session-aware, bandit, Elo, and personalization states stored in process
  memory
- Router Replay used only as the event log and eval/debug source
- Router Learning experience has diagnostics and a materialized-snapshot
  contract, but replay/eval/override materializers remain future work
- adaptation composition is deterministic and method-keyed; richer soft-policy
  arbitration remains future work

If the required identity is missing, session-aware learning should no-op rather
than fail the request. The base routing decision remains valid, and diagnostics
can record `identity_missing` when replay/debug evidence is available.

### Internal Adaptation Contract

Router Learning should use a strongly typed internal contract. It should not
grow by passing loose `map[string]interface{}` payloads between adaptations,
composition, replay, and headers.

The current implementation shape is:

```text
routerLearningAdapter
  Method() -> session_aware | bandit | elo | personalization | ...
  Apply(input) -> routerLearningAdaptationResult
  Observe(input, composed) -> optional post-routing state update
  constructed through a method-specific adapter factory

routerLearningAdaptationResult
  method, mode, scope, action, reason
  hard / changesModel
  selected model refs and selection result
  routerLearningPolicy

routerLearningPolicy
  envelope: learning, adaptation, mode, scope, action, reason
  detail: method-specific typed payload
  experience: typed experience diagnostics
```

Each adaptation owns its own detail struct:

- `session_aware` owns the typed session-policy trace and hashed identity
  diagnostics.
- `bandit` owns algorithm, weighted goals, selected score, and candidate score
  diagnostics.
- `elo` owns rating parameters, selected rating, and leaderboard diagnostics.
- `personalization` owns user hash, selected preference, and preference
  diagnostics.

The only place where these details become `map[string]interface{}` is the
serialization boundary for response headers and Router Replay diagnostics.
Headers read only the compact envelope fields. Replay receives the full
method-keyed diagnostic map. This keeps the hot-path implementation
type-checked while preserving a flexible JSON event-log schema.

New adaptations should follow the same pattern:

1. Add a concrete `routerLearningAdapter` implementation and constructor.
2. Add a method-specific typed detail payload.
3. Keep state ownership inside Router Learning states, not inside decision
   algorithms.
4. Serialize to replay only at the boundary.

Do not add generic `policy.Set(...)`-style APIs or anonymous temporary structs
for adaptation diagnostics. If a field is important enough to appear in replay,
it should belong to a named detail type with tests.

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
| `automix` | mostly `algorithm` | Escalation and verification are request-time selection behavior. Durable verifier success experience should move to Router Learning. |
| `knn`, `kmeans`, `svm`, `mlp` | `algorithm` | Trained model artifacts are offline selector assets, not online router memory. |
| `session_aware` | `router.learning.adaptations.session_aware` | It is cross-request continuity and stay/switch behavior. |
| `model_switch_gate` | `router.learning.adaptations.session_aware` | It is also stay/switch behavior and overlaps with session-aware routing. |
| `lookup_tables` | `router.learning.experience` | They are legacy materialized evidence, not selector configuration. |
| `elo` | `router.learning.adaptations.elo` plus `states` and optional `experience` | Elo ratings and updates are cross-request learned state. Seed ratings or aggregates can become experience. |
| `rl_driven` / Thompson | `router.learning.adaptations.bandit` plus `states` and optional `experience` | Bandit posterior and reward updates are cross-request learning. |
| `gmtrouter` | `router.learning.adaptations.personalization` plus `states` and optional `experience` | User/model/task interaction history is cross-request personalization memory. |

This means `router_dc` can remain an algorithm, as long as it is doing
request-time semantic matching. If it later learns query/model affinity from
traffic, that learned affinity becomes Router Learning experience that
`router_dc` may read.

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
bandit experience, but those facts should be read from Router Learning. The hybrid
algorithm should not own the rating store, posterior, interaction graph, or
feedback update loop.

### Split Pattern for Learning Adaptations

Learning-capable selectors should be decomposed into:

```text
states/update loop -> router.learning.states
historical aggregates -> router.learning.experience
read-time learning policy -> router.learning.adaptations.<name>
base request-time selector -> decision.algorithm
```

For example, Thompson Sampling should not keep posterior state inside
`algorithm.rl_driven`. The posterior belongs to Router Learning states:

```yaml
global:
  router:
    learning:
      adaptations:
        bandit:
          enabled: true
          algorithm: linear_thompson
          goals:
            quality: 0.7
            cost: 0.2
            latency: 0.1
          scope: decision
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

The ratings are Router Learning states. Historical ratings or offline estimates
can seed experience, but config should not imply that a decision-local
algorithm owns the rating store.

## Router Learning Memory Product Model

Router Learning memory should be the product-level umbrella for replay, states,
and experience. Users should not need to reason about three independent systems,
but the implementation must keep their latency and durability contracts
separate.

```text
Router Learning memory
  replay
    replay records
    immutable or append-oriented history for audit, debug, eval, and replay

  states
    session state, conversation state, provider health, bandit posterior,
    Elo ratings, personalization graph

  experience
    materialized historical evidence derived from replay, eval, and optional
    operator overrides:
    handoff_penalty, remaining_turn_estimate, quality_gap, reward_stats
```

The product relationship is:

```text
requests and responses
  -> Router Learning replay
  -> aggregation / outcome scoring
  -> Router Learning experience
  -> Router Learning adaptations
```

Replay is the event-log layer. Session and conversation memory are
states-layer views.
The old lookup table implementation is migration material for experience, but
`lookup_tables` is not a public product concept.

### Latency Model

Router Learning must not turn every routed request into an external storage
round trip. The memory layers have different latency contracts:

| Layer | Hot Request Path | Storage Path | Examples |
| --- | --- | --- | --- |
| `states` | Read and update locally during routing. This is latency-sensitive. | Optional shared synchronization later, but not required for every request. | current model, tool-loop state, turn count, cache warmth, switch history, Elo ratings, bandit posterior |
| `experience` | Read from an in-process or locally cached snapshot. | Recomputed from replay, eval, or overrides by background jobs or offline commands. | handoff penalties, remaining-turn estimates, quality gaps, reward statistics |
| `replay` | Append after the routing decision, preferably async or fail-open. Reads are not required for the current decision. | Durable Router Replay backend. | replay records, diagnostics, eval traces |

The hot path should therefore be:

```text
request
  -> read states from local memory
  -> read experience from local snapshot
  -> compute adaptations
  -> update states locally
  -> append replay event asynchronously or fail-open
```

External storage calls are appropriate for:

- replay writes and replay APIs
- background experience materialization
- cold-start seeding of states or experience snapshots
- optional shared states synchronization when an operator chooses that
  deployment mode

External storage calls should not be required to decide every request. If a
shared states backend is introduced, it needs a small timeout, fail-open
behavior, and local fallback so storage latency or transient outages do not
become routing latency spikes.

### Memory API

Router Learning memory is the product model, but the first follow-up should not
introduce a broad public memory config. Public config should remain behavior
oriented:

- Router Replay storage remains configured through `global.services.router_replay`.
- Enabled adaptations create the states they need internally.
- Experience gets a public config only when the first materializer is
  implemented and users can make meaningful choices about source, refresh, and
  staleness.

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
append-oriented history used for audit, eval, and experience materialization; it
is not the low-latency states that decide the next request.

The replay layer is Router Replay:

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

`states` is intentionally not configurable as a generic top-level switch. It is
required runtime memory for enabled adaptations, so exposing
`states.enabled` would let users create invalid or confusing configs.

`experience` is optional and advanced. It exists for materialized historical
evidence and manual overrides. When implemented, experience snapshots are built
from the router's learning inputs such as Router Replay, evals, and offline
imports. If experience is unavailable, adaptations still work from live states
and built-in defaults.

When experience is configured, learning adaptations can consume it by default.
For example, `session_aware` reads conversation state, session state, cache
accounting, and materialized handoff or remaining-turn experience when those
views exist.

### Memory Views

Initial Router Learning memory should expose these categories:

| View | Layer | Source | Consumers |
| --- | --- | --- | --- |
| `replay_record` | `replay` | request/response records | audit, eval, replay, experience |
| `session_state` | `states` | request/response telemetry | `session_aware` |
| `conversation_state` | `states` | request/response telemetry | `session_aware` |
| `provider_health` | `states` | transport outcomes | `provider_health`, `multi_factor`, `hybrid` |
| `elo_rating` | `states` | feedback/outcome updates | `elo`, `hybrid` |
| `bandit_posterior` | `states` | feedback/outcome updates | `bandit`, `hybrid` |
| `interaction_graph` | `states` | feedback, user/session history | `personalization` |
| `handoff_penalty` | `experience` | replay/eval aggregation, overrides | `session_aware` |
| `remaining_turn_estimate` | `experience` | replay/eval aggregation, overrides | `session_aware` |
| `quality_gap` | `experience` | replay/eval aggregation, outcomes, overrides | `hybrid`, `bandit`, future quality adaptations |

The important part is that these are shared memory views. They should not be
private fields inside individual algorithms.

### Lookup Table Cleanup

The current `lookup_tables` concept is useful implementation material but sits
in the wrong product layer. It should not be renamed directly into the public
API.

Instead:

- Remove `model_selection.lookup_tables` from the public contract.
- Keep or refactor the existing lookup-table code only as internal migration
  material for experience snapshots.
- Do not expose `storage_path` or `auto_save_interval` as Router Learning
  memory knobs.
- Add a public `router.learning.experience` shape only when the first
  experience materializer exists.

Manual overrides remain useful, but they should become experience overrides,
not selector-local tables.

### States vs Experience

States and experience are different layers of Router Learning memory.

| Memory Layer | Scope | Purpose |
| --- | --- | --- |
| `states` | live session, conversation, provider, user, or decision | Protect current continuity, cache, tool loops, health, preference, and switch history. |
| `experience` | aggregate traffic history or offline eval output | Provide learned handoff costs, remaining-turn estimates, quality gaps, and reward statistics. |

Session-aware learning should read both:

```text
conversation_state says: active tool loop, keep current model
session_state says: cache is warm, switching has been frequent
experience says: switching from local to frontier has handoff penalty 0.05
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

`model_selection.lookup_tables` is no longer a public concept. Its useful
implementation ideas belong to future Router Learning experience, not to
selector-local config.

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
| `model_selection.lookup_tables` | no direct replacement; future experience materializers may reuse the implementation ideas |
| `algorithm.type: elo` | `router.learning.adaptations.elo`; add a base `algorithm` only if the decision needs one |
| `model_selection.elo` | `router.learning.adaptations.elo` |
| `algorithm.type: rl_driven` | `router.learning.adaptations.bandit`; add a base `algorithm` only if the decision needs one |
| `algorithm.rl_driven.use_thompson_sampling` | `router.learning.adaptations.bandit.algorithm: linear_thompson` |
| `algorithm.type: gmtrouter` | `router.learning.adaptations.personalization`; add a base `algorithm` only if the decision needs one |
| `algorithm.gmtrouter.storage_path` | no direct replacement; personalization states are owned by Router Learning and future materialized seeds belong to experience |

These old fields should not have public replacements in the clean API:

| Old Field | Handling |
| --- | --- |
| `stay_bias` | Fold into `switch_margin` and internal defaults. |
| `tool_loop_stay_bias` | Internal behavior; tool-loop protection should not require score tuning. |
| `tool_loop_hard_lock` | Default enabled in session-aware learning. |
| `context_portability_hard_lock` | Default enabled in session-aware learning. |
| `decision_drift_reset` | Replaced by explicit conversation identity. |
| `quality_gap_multiplier` | Internal selector-normalization detail. |
| `remaining_turn_prior_weight` | Internal or future experience advanced setting. |
| `remaining_turn_prior_horizon` | Internal or future experience advanced setting. |
| `min_remaining_turn_prior_samples` | Internal or future experience advanced setting. |

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
future Router Learning experience. Remove lookup_tables from the public config.
```

Runtime config loading should not silently rewrite old config, and the first
implementation should not provide an automatic config rewrite command. Migration
is a deliberate manual edit from the old API shape to the new one. Silent or
automatic rewrites would preserve ambiguity in the API.

## Current Implementation Closure Plan

The current implementation should close the first Router Learning follow-up
tranche without attempting to implement the full Router Learning roadmap. The
closure boundary is:

- session-aware routing is a global adaptation, not a decision algorithm
- bandit day0 is a Router Learning adaptation, not a decision algorithm
- decision-level `apply`, `observe`, and `bypass` are supported
- conversation and session scopes are supported
- generic learning headers and replay diagnostics explain the result
- old learning-style public algorithm config fails validation with actionable
  errors instead of being rewritten
- AMD agentic routing recipe and guide use the new API
- docs clearly describe remaining work without implying it already exists

### 1. Terminology and Public Surface

Finish the vocabulary cleanup:

- Use `replay`, `states`, `experience`, and `adaptations` as the four Router
  Learning memory layers.
- Remove `priors` from the public proposal language.
- Treat `lookup_tables` as legacy implementation material, not a public API.
- Keep `decision.algorithm` scoped to current-request base selection.
- Keep `global.router.learning.adaptations.<name>` as the place where
  cross-request learning adjusts the base route.

Acceptance:

- The proposal and issue roadmap use the same terminology.
- No section suggests that `lookup_tables` is the future public config name.
- No section suggests that `router.learning.states.enabled` or a broad public
  memory backend is required for the current implementation.

### 2. Session-Aware Adaptation Contract

Lock the first supported adaptation contract:

- global config:
  `global.router.learning.adaptations.session_aware`
- decision config:
  `routing.decisions[].adaptations.session_aware`
- modes: `apply`, `observe`, `bypass`
- scopes: `conversation`, `session`
- identity headers under:
  `identity.headers.session` and `identity.headers.conversation`
- states are in-process and single-replica for the current implementation

Acceptance:

- Missing identity no-ops learning and falls back to the base route.
- `bypass` decisions cannot be overwritten by learning.
- `observe` records the learning result without changing the final model.
- `session` scope protects the session model until idle release.
- `conversation` scope protects one agent run while allowing a later
  conversation to re-route.

### 3. Replay and Diagnostics

Keep diagnostics generic and bounded:

- learning response headers remain method-keyed:
  `x-vsr-learning-methods`, `x-vsr-learning-actions`,
  `x-vsr-learning-scopes`, `x-vsr-learning-reasons`, and
  `x-vsr-learning-modes`
- `x-vsr-replay-id` points to full diagnostics
- Router Replay stores learning details under
  `learning.adaptations.<name>`
- replay diagnostics include base model, final model, action, reason, scope,
  mode, cache evidence, and hashed identity diagnostics where available

Acceptance:

- The old session-aware replay shape is not compatibility-filled for new
  diagnostics.
- Compact headers do not become a dumping ground for full score traces.
- Full explanation lives in Router Replay.

### 4. AMD Agentic Recipe and Guide

Keep the AMD recipe semantic:

- simple tasks route to the simple/local model
- complex tasks route to the stronger model
- privacy-sensitive tasks route to the local/private model
- domain tasks route to the domain model
- session-aware learning protects continuity, tool loops, cache, and handoff
  cost after the base decision proposes a model

Acceptance:

- The recipe does not add a synthetic `agentic_session_route`.
- Privacy, security, and local-only decisions use `bypass`.
- Normal simple, complex, and domain decisions inherit `apply`.
- The guide explains both `conversation` and `session` protection.

### 5. Documentation Closure

The proposal should be the source of truth for follow-up agents. It must state:

- what the current implementation covers
- what the current implementation deliberately does not implement
- which follow-up tasks belong to the roadmap
- how old learning-style algorithms should move into the new architecture
- why eval is the final architecture validation step, while normal tests still
  belong with each implementation PR

Acceptance:

- A future agent can read this proposal and know which task belongs to which
  phase.
- Follow-up tasks do not depend on ambiguous terms such as `priors` or
  `lookup table` as product concepts.

## Follow-Up Roadmap

The follow-up roadmap is clean-break only. Old config paths should fail
validation with actionable errors; they should not be rewritten or silently
mapped into the new API.

### 1. Router Learning Runtime Contract

Current status: implemented for session-aware, bandit, Elo, and personalization.
The contract is internal, keeps request-time reads local, records method-keyed
diagnostics, and lets the API server forward explicit feedback into Router
Learning states without depending on extproc internals.

Define the internal contract that all future adaptations use:

- adaptation input:
  base route, candidate set, matched decision, request identity, states snapshot,
  and experience snapshot
- adaptation output:
  method, mode, scope, action, reason, base model, final model, score deltas,
  hard/soft classification, and diagnostics
- states update contract:
  request-time updates are local and fail-open
- replay contract:
  records facts after the decision for audit, eval, and experience generation

Acceptance:

- A second adaptation can be implemented without adding new decision-level
  concepts.
- Request-time routing does not require synchronous external storage reads.
- Missing states or experience falls back to the base route or built-in defaults.

### 2. Router Learning Experience

Current status: minimal diagnostics implemented. Adaptation policies can record
whether expected experience views are `used` or `missing`; full replay/eval
materializers, refresh policy, and operator overrides remain future work.

Implement experience as materialized historical routing evidence, not as a
direct rename of lookup tables. Experience is not a user-managed feature flag.
Enabled adaptations declare what experience they can use, and the router
consumes available experience automatically.

Do not add public config like:

```yaml
experience:
  enabled: true
  source: ...
```

Those fields do not give users a meaningful product choice today. Router Replay
is the canonical event log, eval and offline imports can feed the same internal
experience snapshot, and missing experience should only affect warm start or
calibration.

Experience should be defined by adaptation needs:

| Adaptation | Experience examples |
| --- | --- |
| `session_aware` | switch cost, cache-loss estimate, multi-turn length pattern |
| `elo` | initial ratings, domain/model win rates, pairwise outcome seeds |
| `bandit` | posterior warm start, reward calibration, cost/latency baseline |
| `personalization` | user cluster seeds, workspace preference summaries, interaction snapshots |
| `provider_health` | reliability baseline, region/model failure pattern |
| `cost_optimizer` | cost-quality curve, cache-retention value |

Acceptance:

- Experience is generated asynchronously from replay, eval, or operator
  overrides.
- The request path reads an in-memory or locally cached snapshot only.
- Experience entries carry source, sample count, version, freshness, and
  applicable model or recipe scope.
- Adaptations declare the experience views they can consume.
- Missing, stale, or invalid experience falls back to built-in defaults or
  cold-start state and is recorded in replay diagnostics as
  `used`, `missing`, `stale`, or `ignored`.

### 3. Adaptation Composition

Current status: minimal deterministic composer implemented. It records every
adaptation, ignores final-model changes from `observe`, and lets hard changes
win over soft changes. Rich soft-policy arbitration remains a future extension.

Define the composer before adding multiple final-model-changing adaptations.

Minimum rules:

- decision `bypass` wins for that adaptation
- hard guards win over soft score adjustments
- `observe` records a hypothetical result but does not change the final model
- soft adjustments flow into one final arbiter
- replay records each adaptation result and the final combined result

Acceptance:

- Two adaptations can disagree and the final route is deterministic.
- Headers remain compact and method-keyed.
- Replay explains both per-adaptation decisions and the final combined route.

### 4. Stateful Algorithm Migration

Current status: public config migration is implemented for the old learning
algorithm paths. `session_aware`, `elo`, `rl_driven`, and `gmtrouter` are
rejected as public `decision.algorithm.type` values. `bandit`, `elo`, and
`personalization` have conservative day-0 Router Learning states and feedback
update loops.

Migrate learning-style algorithms into the new architecture instead of keeping
private state inside `decision.algorithm`.

Targets:

- `elo` becomes `router.learning.adaptations.elo`
  - ratings are states
  - historical rating seeds can be experience
- `rl_driven` and Thompson sampling become
  `router.learning.adaptations.bandit`
  - posterior and reward updates are states
  - historical reward aggregates can be experience
  - day0 algorithms: `linucb` as default and `linear_thompson` as supported
  - `epsilon_greedy` can exist as an internal/test baseline, not the
    recommended public algorithm
- `gmtrouter` becomes a personalization adaptation
  - user interaction graph and preference history are states and experience
- `lookup_tables` are removed as a public concept and only reused as internal
  migration material if helpful

Non-targets:

- `router_dc` remains a base algorithm.
- `hybrid` remains a base algorithm, but any learned evidence it consumes must
  come from Router Learning states or experience.
- Router-R1 LLM-as-router should be evaluated separately as a base algorithm.
- Multi-round aggregation should be treated as an execution strategy, not
  forced into adaptation semantics.

Acceptance:

- No learning-style algorithm owns a separate long-lived rating store,
  posterior store, or interaction graph outside Router Learning.
- Feedback and reward updates use the shared learning states/replay path.
- Old public algorithm config for migrated capabilities is rejected with clear
  errors.
- Bandit uses a global algorithm choice and allows per-decision sparse
  overrides for `mode`, `goals`, and necessary `tuning`, but not per-decision
  algorithm overrides.

### 5. Distributed States Semantics

Define deployment behavior after the single-replica implementation:

- single replica: fully supported
- sticky multi-replica: recommended production path for session/conversation
  state
- shared experience plus local states: acceptable because experience is a
  snapshot
- shared hot states: future design only, with strict latency and failure-mode
  requirements

Acceptance:

- Kubernetes guidance states when sticky routing is required.
- Non-sticky deployments degrade or observe explicitly; they do not silently
  claim stable session protection.
- Shared state is not introduced without timeout, fail-open, and local fallback
  semantics.

### 6. Architecture Eval

Eval is the final validation step for the full Router Learning architecture.
Each implementation PR still needs focused unit and integration tests.

The architecture eval should cover:

- route correctness
- base model versus final model
- adaptation method, mode, action, reason, and scope
- cache/cost delta
- unnecessary switch rate
- conversation-scope and session-scope protection
- privacy/security bypass behavior
- Elo, bandit, and personalization migration behavior
- replay explainability coverage
- learning overhead at p50 and p95

Acceptance:

- The eval can identify reasonable and unreasonable routing choices.
- The eval quantifies cost/cache impact in auto mode.
- The eval can be run from deterministic replay fixtures and reported in a
  concise human-readable plus machine-readable form.

## Confirmed Follow-Up Decisions

- The historical evidence layer is called `experience`.
- Experience is internal to Router Learning and is not exposed through a broad
  `experience.enabled`, `experience.source`, or `experience.views` public API.
- Enabled adaptations declare which experience views they can consume.
- `states` is not a broad public config object in the first follow-up.
- Bandit day0 supports `linucb` as the default and `linear_thompson` as a
  supported algorithm.
- Bandit follow-ups should track non-stationary, budgeted, preference,
  adversarial, dueling, and neural bandit variants.
- Bandit goals use a weighted map. Day0 goals are `quality`, `cost`, and
  `latency`, with defaults favoring quality and cost.
- Decision-level bandit overrides can include `mode`, `scope`, `goals`, and
  necessary `tuning`. They should not override the global bandit algorithm.
- The current implementation closes session-aware Router Learning, method-keyed
  diagnostics, minimal composition, clean-break public config migration, bandit
  day-0 runtime, Elo rating states, and personalization preference states.
  Experience materialization, distributed states, and architecture eval remain
  follow-up tasks.
- Existing local lookup-table code may be refactored or removed during the
  experience work; the public `lookup_tables` concept should not survive.

## Future Learning Extensions

`router.learning.adaptations` is intentionally broader than session-aware
routing. Future adaptations should use the same replay, states, experience, and
decision override model.

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

      adaptations:
        session_aware:
          enabled: true
          scope: conversation

        bandit:
          enabled: false
          algorithm: linucb
          goals:
            quality: 1.0
            cost: 0.25
            latency: 0.10
          tuning:
            exploration_budget: 0.05

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
    goals:
      quality: 1.0
      cost: 0.10
    tuning:
      exploration_budget: 0.02
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
| states | Live cross-request facts owned by enabled adaptations. |
| experience | Optional materialized historical evidence consumed by adaptations when available. |
| goals | Weighted optimization goals for adaptations that optimize multiple objectives. |
| tuning | Sparse advanced controls for the adaptation when defaults are not enough. |

`goals` is a weighted map rather than a string objective. The router owns the
direction of each built-in goal: `quality` is maximized, while `cost` and
`latency` are minimized. Future goals such as `reliability` and
`cache_efficiency` can be added without changing the API shape. Privacy and
security remain hard decision boundaries, not soft optimization goals.

Bandit day0 should support:

| Algorithm | Role |
| --- | --- |
| `linucb` | Default contextual bandit for routing with query/request features. |
| `linear_thompson` | Supported migration path for Thompson-style exploration. |
| `epsilon_greedy` | Internal/test baseline only, not the recommended public algorithm. |

Bandit follow-ups should track:

| Algorithm Family | Use Case |
| --- | --- |
| `discounted_linucb`, `sliding_window_linucb` | Non-stationary traffic and model drift. |
| `budgeted_linucb` / bandits with knapsack | Hard budget constraints. |
| preference-conditioned contextual bandits | User or tenant cost/performance preference vectors. |
| dueling bandits | Pairwise preference feedback. |
| `exp3` | Adversarial or high-shift traffic. |
| `neural_linear`, `neural_ucb`, `neural_ts` | Higher-capacity contextual bandits after data and explainability justify them. |

Decision-level bandit overrides should be sparse and should not override the
global bandit algorithm. Allow `mode`, `goals`, and necessary `tuning`; keep the
algorithm family global so states and diagnostics remain coherent.

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

- The public namespace is `global.router.learning` and
  `global.router.learning.adaptations.<name>`.
- Router Learning memory is described as replay, states, and experience.
- Replay is configured through `global.services.router_replay`, not a second
  learning-specific replay backend.
- States are runtime memory owned by enabled adaptations; they are not a broad
  public `states.enabled` switch.
- Experience is the future materialized historical evidence layer. It is not a
  direct rename of lookup tables.
- Experience does not expose broad public `enabled`, `source`, or `views`
  toggles. Enabled adaptations declare their usable experience internally and
  fall back when experience is missing.
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
- Router Learning replay records should reuse the existing Router Replay
  service, storage backends, route-local plugin, and replay API.
- Response diagnostics should use a small generic `x-vsr-learning-*` header
  family plus the existing `x-vsr-replay-id` pointer. Adaptation-specific
  header names should not become the stable API.
- The current implementation targets a single router replica.
- Missing identity should no-op session-aware learning without failing the
  request.
- Router Learning should not depend on raw replay payload fields. Payload
  persistence is governed by existing Router Replay capture configuration.
- New replay diagnostics use `learning.adaptations.<name>` only; the new
  implementation should not compatibility-fill older session-aware-specific
  replay fields.
- Router Learning experience materialization is roadmap work. The current
  implementation records method-level experience diagnostics and keeps
  request-time routing on local states and built-in defaults.
- Bandit day0 uses a weighted `goals` map for multi-objective optimization.
  Built-in day0 goals are `quality`, `cost`, and `latency`.
- Bandit day0 supports `linucb` as the default and `linear_thompson` as a
  supported algorithm. Decision-level bandit overrides can tune `mode`,
  `goals`, and necessary `tuning`, but not the global algorithm family.

## Closed Decisions For Current Implementation

### 1. Hot Path

Request-time routing must not require synchronous external storage reads.
Learning adaptations read and write in-process state. Replay writes are off the
critical path and should fail open.

### 2. Deployment Scope

The current implementation targets a single router replica. Multi-replica state,
shared Redis state, sticky routing requirements, and cross-replica
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

### 6. Experience Roadmap

Router Learning experience materialization remains in the design and roadmap.
The current implementation records minimal experience diagnostics, but it does
not build replay-derived, eval-derived, or operator-override snapshots.

### 7. Future Adaptation Ordering

The current implementation has a minimal deterministic composer: hard changes
win over soft changes, `observe` does not change the final route, and every
method is recorded. Public weighting, priorities, and phases remain future
work for richer multi-adaptation arbitration.

### 8. Eval Contract

The agentic routing eval should cover semantic routing, session-aware stability,
new-conversation release, bypass behavior, cache evidence, learning latency
overhead, and replay explainability.

## Migration Tooling Decision

Do not provide an automatic config rewrite command for this clean-break API.
This is a new public contract, not a compatibility layer. Old config shapes
should fail fast with actionable validation errors and documentation should show
the new shape.

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
- The current implementation is valid for a single router replica.
- Future learning adaptations can be added without adding new decision-level
  concepts.
