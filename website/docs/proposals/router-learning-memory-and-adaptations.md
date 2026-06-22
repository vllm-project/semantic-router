# Router Learning: Adaptation, Protection, and Recipe Learning

## Status

This is the system design proposal and implementation contract for the next
Router Learning surface in vLLM Semantic Router. It is intended to be the source
of truth for follow-up implementation work, not a migration guide.

The design has three product responsibilities:

- **adaptation**: online model-choice learning from runtime experience
- **protection**: online stability control for sessions, conversations, cache,
  tool loops, and handoff cost
- **recipe learning**: offline agent-driven eval and experiment loops that
  improve recipes

This is a clean-break design. The old public learning shapes are removed rather
than kept as compatibility aliases. In particular, public config names such as
`session_aware`, `bandit`, `elo`, `personalization`, `rl_driven`,
`gmtrouter`, and `lookup_tables` should not remain on the new learning API.

The public API should stay small and operator-readable. Internal implementations
may use posterior scoring, guarded stochastic sampling, online experience,
identity-scoped state, EWMA telemetry, or replay-derived seed packs, but those
implementation details should not become the primary product vocabulary.

The first implementation target is intentionally narrow:

- one online adaptation strategy: `routing_sampling`
- one protection runtime with conversation and session scopes
- one replay-linked outcome API
- one offline recipe learning loop that produces findings, metrics, candidate
  recipes, recipe patches, and optional experience seed packs

Future algorithms can extend this design, but they should not make the first
operator-facing API larger than necessary.

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
- Do not expose storage backends, priors, or replay implementation details as
  first-version learning config.
- Do not preserve old learning config names as aliases.

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
| Recipe learning | Offline agent loop that uses replay, outcomes, and evals to propose recipe changes. |

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
  vectors, priors, storage layout, and implementation-specific score details
  belong in typed runtime code and replay diagnostics.

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
| Offline recipe learner | No | Run evals and experiments, then propose findings, recipe patches, and seed packs. |

Request-time routing must use in-process online state. Durable storage is used
for replay and offline work, not synchronous request-path lookups.

## Public API

The public API has two levels:

- global learning components under `global.router.learning`
- per-decision controls under `routing.decisions[].adaptations`

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
          protection_weight: 1.0
          cache_weight: 0.20
          handoff_penalty: 0.05
          handoff_penalty_weight: 1.0
          switch_history_weight: 0.04
          max_cache_cost_multiplier: 2.5

routing:
  decisions:
    - id: example
      adaptations:
        mode: apply
        adaptation:
          mode: apply
        protection:
          mode: apply
        coordination:
          protection_weight: 1.0
          switch_margin: 0.05
```

Most recipes should not need the per-decision block. It exists for bypassing
sensitive decisions, observing learning without changing routes, or tuning the
adaptation/protection trade-off for a specific route.

The first public API deliberately does not expose:

- `global.router.learning.adaptations`
- `router.learning.memory`
- `experience.enabled`
- `goals`
- old algorithm-family blocks such as `session_aware`, `bandit`, `elo`,
  `personalization`, `rl_driven`, `gmtrouter`, or `lookup_tables`

Adaptation and protection are the product concepts. Strategy names and state
shape are implementation details unless they affect operator behavior.

### Global Learning

`global.router.learning.enabled: true` enables the learning system. When enabled
without component overrides, both adaptation and protection are enabled with
conservative defaults:

- `adaptation.strategy: routing_sampling`
- `adaptation.candidate_set: decision`
- `protection.scope: conversation`
- protection identity headers default to `x-session-id` and `x-conversation-id`

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
          protection_weight: 1.0
          cache_weight: 0.20
          handoff_penalty: 0.05
          handoff_penalty_weight: 1.0
          switch_history_weight: 0.04
          max_cache_cost_multiplier: 2.5
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

### Adaptation Config

Adaptation has one public strategy on day 0:

```yaml
adaptation:
  enabled: true
  strategy: routing_sampling
  candidate_set: decision
```

`routing_sampling` is the default online learning strategy. It reads experience,
scores candidates, and may draw bounded random samples when protection allows
exploration. It is enabled by default when adaptation is enabled.

`strategy` is the field that selects the online adaptation algorithm.
`routing_sampling` is a strategy value, not a separate nested component and not
a second enable switch. The first API deliberately avoids exposing algorithm
internals such as priors, reward weights, feature vectors, or posterior
parameters.

Future online strategies can be added as new `strategy` values only when they
represent distinct operator-facing behavior. They must reuse the same outcome,
experience, protection, header, and replay contracts.

### Candidate Sets

`candidate_set` controls where adaptation may search for models.

| Value | Candidate Models |
| --- | --- |
| `decision` | Models from the matched decision's `modelRefs`. |
| `tier` | Union of `modelRefs` from all decisions with the same `decision.tier` as the matched decision. |
| `global` | Union of all `modelRefs` from routable decisions in the recipe. |

`tier` uses the existing decision `tier` field. It is a route tier, not a model
quality label. If `candidate_set: tier` is configured and the matched decision
has no positive tier, the router falls back to `decision`.

`global` is the broadest mode. It should use stricter cost and reliability
guards because it can propose a model outside the matched decision's local
candidate set.

Candidate set construction must be deterministic:

- de-duplicate models while preserving the recipe's stable decision/model order
- include only routable `modelRefs` present in the deployed recipe
- apply model health, capability, provider, and static cost eligibility before
  scoring
- never use another decision's `adaptations` block as policy for the matched
  request
- respect the matched decision's `adaptations.mode: bypass` before candidate
  construction

`candidate_set: global` and `candidate_set: tier` expand where adaptation may
search. They do not turn learning into an unrestricted model router and they do
not bypass the matched decision's policy boundary.

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
    protection_weight: 1.0
    cache_weight: 0.20
    handoff_penalty: 0.05
    handoff_penalty_weight: 1.0
    switch_history_weight: 0.04
    max_cache_cost_multiplier: 2.5
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
- `protection.mode` controls whether stability protection can apply
- `coordination` tunes the adaptation/protection trade-off

Do not add per-decision algorithm internals unless they solve a concrete recipe
authoring problem that cannot be expressed by these controls.

Decision-level coordination tunes the trade-off between adaptation and
protection:

```yaml
adaptations:
  coordination:
    protection_weight: 1.5
    switch_margin: 0.05
```

Higher `protection_weight` favors stability. Lower `protection_weight` makes it
easier for adaptation to switch models. This is the first-version coordination
surface; do not add per-guard booleans unless a concrete operator workflow
needs them.

`coordination` is intentionally small. It adjusts how strongly protection
pushes back on an adaptation proposal for this decision. It does not configure
the adaptation algorithm, outcome schema, replay format, or storage behavior.

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
5. Protection switch guard decides whether to hold, accept, or rescue the
   `proposal_model`.
6. The router sends the request to the final model.
7. Compact headers summarize active methods, actions, scopes, and reasons.
8. Replay records base, proposal, final, decision, tier, and diagnostics.
9. Outcomes update experience and feed offline recipe learning.

Protection is intentionally split into two guard points:

- **preflight**: suppress or allow random exploration before adaptation scores.
- **switch guard**: accept, reject, or rescue the `proposal_model` after
  scoring.

Adaptation answers "which model looks better from experience?" Protection
answers "is it safe and worthwhile to explore or switch at this point in the
agent flow?" If adaptation is disabled or observing only, protection still
answers that question for the base selector's proposed model.

## Adaptation

Adaptation improves model selection within the configured candidate set while
the recipe remains deployed.

The day-0 strategy is `routing_sampling`. It combines:

- offline or default quality priors
- accepted, rejected, overused, and failed outcomes
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

### Evidence Channels

Adaptation should learn from bounded evidence channels:

| Channel | Source | Online Use |
| --- | --- | --- |
| Model outcomes | `POST /v1/router/outcomes` with `target: model` | Update model fit, overuse, or failure evidence. |
| Router telemetry | response status, latency, cache, token, retry, and provider evidence | Update reliability, latency, cache, and effective cost summaries. |
| Offline seed packs | recipe learning output | Initialize quality priors for cold start. |

Route, policy, and stability outcomes are preserved for replay and offline
recipe learning. They should not directly mutate model-fit counters, because a
wrong decision is not the same failure as a weak model.

User interfaces do not have to expose the API verdict names directly. A chat
client, eval harness, or agent can map its own feedback language into the typed
outcome schema, as long as the router receives normalized verdicts and reasons.

### Reward Semantics

The first online reward model is deliberately small and typed:

| Evidence | Experience Update |
| --- | --- |
| `target: model`, `verdict: accepted` | Increase model fit for the matched decision/tier/model context. |
| `target: model`, `verdict: rejected` | Decrease model fit, usually because the model was underpowered or inappropriate. |
| `target: model`, `verdict: overused` | Increase overuse or cost pressure without reducing model quality. |
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
| `quality_prior` | Offline or imported quality estimate, defaulting to neutral. |
| `prior_weight` | Pseudo-count strength for the quality prior. |
| `accepted_count` | Outcomes where the model fit the task. |
| `rejected_count` | Outcomes where the model was underpowered or inappropriate. |
| `overused_count` | Outcomes where the model worked but was unnecessarily expensive or strong. |
| `failed_count` | Provider or execution failures. |
| `latency_ewma` | Observed latency score. |
| `cache_hit_ewma` | Observed cache reuse. |
| `cache_write_ewma` | Observed cache write pressure. |
| `input_cost_multiplier_ewma` | Effective input cost multiplier. |
| `last_updated` | Freshness and decay input. |

The first strategy consumes experience as follows:

| Evidence | Runtime Use |
| --- | --- |
| `quality_prior`, `prior_weight` | Cold-start posterior before enough local outcomes exist. |
| `accepted_count`, `rejected_count` | Posterior model-fit estimate for the current decision/tier/model context. |
| `overused_count` | Overuse and cost penalty when a cheaper or smaller model is likely sufficient. |
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

Online experience should update from bounded evidence only:

- router telemetry can update latency, cache, cost, and provider failure fields
- model-targeted outcomes can update fit, overuse, and failure fields
- route, policy, and stability outcomes are preserved for offline recipe
  learning and should not directly mutate model-fit counters
- offline recipe learning can import seed packs by setting quality priors and
  prior weights

### Routing Sampling

For each request, `routing_sampling` runs these steps:

1. Build the candidate set from `decision`, `tier`, or `global`.
2. Apply static eligibility filters.
3. Load local experience with fallback from decision/tier/model to tier/model
   to model.
4. Compute a posterior quality estimate for each candidate.
5. Apply cost, overuse, reliability, latency, and cache adjustments.
6. Use posterior means when protection suppresses sampling.
7. Draw bounded posterior samples when protection allows exploration.
8. Select a winner and emit either the base model or a `proposal_model`.
9. Send the proposal to protection for the final switch decision.

For each candidate model, adaptation computes the posterior quality estimate:

```text
alpha = prior_weight * quality_prior + accepted_count + smoothing
beta  = prior_weight * (1 - quality_prior) + rejected_count + smoothing
mean  = alpha / (alpha + beta)
```

When protection allows exploration:

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

`overused_count` should not reduce quality. It increases the cost or overuse
penalty. `failed_count` affects reliability, not task quality.

The score formula is an implementation contract, not a public tuning surface in
the first API. Operators should tune recipe policy, candidate set, component
modes, and protection coordination before asking for new score knobs.

The strategy should use a small typed action set in replay and compact headers:

| Action | Meaning |
| --- | --- |
| `score_only` | Deterministic posterior scoring was used. |
| `sampled` | Bounded posterior sampling was used for at least one eligible candidate. |
| `keep_base` | The base model remained the best candidate. |
| `propose_switch` | Adaptation proposed a different `proposal_model` for the switch guard. |
| `observe` | Adaptation ran diagnostics but could not change the route. |
| `bypass` | The matched decision disabled adaptation. |

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
smaller than a full contextual bandit. It learns per decision, tier, and model
from outcomes and telemetry. It does not yet fit or update a high-dimensional
request feature model on the hot path.

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
switch if proposal_gain >= switch_margin + protection_weight * switch_cost
```

Where:

- `proposal_gain` is the proposed model's advantage over the protected, current,
  or base model.
- `switch_cost` is derived from cache, handoff, tool-loop, session, and switch
  history evidence.
- `switch_margin` is the minimum gain required before any switch.
- `protection_weight` controls how much stability matters.

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
| `allow_sampling` | Stochastic adaptation is allowed for this request. |
| `suppress_sampling` | Adaptation must score deterministically for this request. |
| `hold_current` | The proposal model was rejected by the switch guard. |
| `allow_switch` | The proposal model cleared the switch guard. |
| `rescue_switch` | The switch was allowed because rescue evidence outweighed stability cost. |
| `bypass` | The matched decision disabled the protection component. |

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
  "verdict": "rejected",
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
| `target` | `model`, `route`, `policy`, or `stability`. |
| `target_ref` | Optional model, decision, route, or policy id. For `target: model`, omitted means the final routed model from replay. |
| `verdict` | `accepted`, `rejected`, `overused`, or `failed`. |
| `reason` | Optional structured reason string. |
| `score` | Optional normalized evidence strength. |
| `metadata` | Optional bounded metadata for run ids, eval ids, or provider ids. |

Outcome semantics:

| Target | Verdict | Runtime Update |
| --- | --- | --- |
| `model` | `accepted` | Increase model fit for the decision/tier/model key. |
| `model` | `rejected` | Decrease model fit, usually because capability was insufficient. |
| `model` | `overused` | Increase overuse or cost penalty without reducing quality. |
| `model` | `failed` | Update reliability or provider failure state. |
| `route` | any | Feed offline recipe learning; do not directly update model quality. |
| `policy` | any | Feed offline recipe learning; do not let adaptation override the boundary. |
| `stability` | any | Feed protection tuning and offline analysis. |

Online adaptation should update experience only from `target: model`
outcomes. Route, policy, and stability outcomes are preserved for offline
recipe learning and analysis. This keeps "wrong decision" feedback separate
from "bad model" feedback.

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

## Headers and Replay Diagnostics

Response headers should stay compact. Detailed explanation belongs in Router
Replay.

Recommended compact headers:

```text
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=sampled,protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=posterior_win,protection=switch_allowed
x-vsr-replay-id: replay_...
```

Do not encode a mini language in one learning header. Headers identify active
methods and final high-level actions only. Detailed guard inputs, scores,
samples, costs, state hashes, and candidate traces belong in replay.

Header values should be comma-separated lowercase tokens or `method=value`
pairs. They should not use semicolon-delimited parameter strings. Old values
such as `session_aware;mode=apply;action=switch` are not part of the clean API.

Inactive components should be omitted from component headers. For example, if
only protection is active:

```text
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=hold_current
x-vsr-learning-scopes: protection=session
x-vsr-learning-reasons: protection=cache_cost_high
```

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
      "protection_weight": 1.0,
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
| `experience_seed_pack` | Offline priors for adaptation cold start. |

Offline recipe learning is where recipe edits happen. It uses replay, outcomes,
eval cases, and experiment metrics to produce findings and candidate patches.
It does not silently apply those patches on the request path.

The offline loop is the place where classifier-like recipe improvement happens.
It can inspect replay, outcomes, failed eval cases, and successful counterfactual
experiments, then propose edits to signals, examples, thresholds, decisions,
tiers, priorities, and modelRefs. Those edits are recipe patches, not hidden
runtime mutations.

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

### Offline Loop Contract

The offline loop should consume:

| Input | Purpose |
| --- | --- |
| Replay routes | Reconstruct base, proposal, final model, decision, tier, headers, and learning diagnostics. |
| Outcomes | Explain model fit, overuse, route mistakes, policy misses, and stability problems. |
| Eval cases | Provide expected route/model behavior and cost/latency expectations. |
| Candidate recipe | The current recipe plus generated recipe variants. |
| Optional seed packs | Prior quality estimates for cold-start experiments. |

The loop should produce:

| Output | Required Contents |
| --- | --- |
| Findings | Per-decision explanation of observed routing problems and supporting replay ids. |
| Metrics | Per-decision and per-tier correctness, cost, latency, cache, switch, and outcome metrics. |
| Candidate recipes | Complete recipe variants that can be replayed or served in a test environment. |
| Experiment results | Side-by-side metrics for baseline and candidate recipes. |
| Recipe patch | Minimal proposed config diff. |
| Experience seed pack | Optional quality priors and prior weights for adaptation cold start. |

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
| `DecisionAdaptationsConfig` | `mode`, `adaptation.mode`, `protection.mode`, `coordination`. |
| `LearningPipelineInput` | base selection, matched decision, tier, request evidence, identity, replay ids. |
| `AdaptationDecision` | action, candidate set, strategy, base model, proposal model, score diagnostics. |
| `ProtectionDecision` | action, scope, protected model, proposal model, final model, switch cost, reason diagnostics. |
| `RouterOutcome` | replay id, source, target, target ref, verdict, reason, score, metadata. |
| `ExperienceRecord` | typed online state keyed by decision, tier, and model. |

The learning pipeline owns ordering and coordination. Adaptation and protection
components should not call each other directly.

Global component `enabled` fields should preserve the tri-state semantics from
the public API: omitted means inherit the default-on behavior under
`learning.enabled: true`; explicit `false` disables that component.

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

### Phase 0: Clean API Cutover

- Remove old public learning config blocks instead of keeping aliases.
- Replace old runtime method names with `adaptation` and `protection`.
- Replace old single-value learning header values with the compact header set.
- Replace generic replay learning maps with typed `protection_preflight`,
  `adaptation`, and `protection` diagnostics.
- Keep old docs and tutorials from presenting removed API shapes.

### Phase 1: Runtime Interfaces and State Contracts

- Add a learning pipeline that receives the base selection result and returns a
  final selection result.
- Add typed component interfaces for protection and adaptation.
- Add typed online protection state keyed by session or conversation identity.
- Add typed experience records keyed by decision, tier, and model.
- Add a typed outcome model shared by API handlers, replay storage, and online
  experience updates.

### Phase 2: Protection Runtime

- Implement conversation and session scope.
- Implement identity resolution from configured headers.
- Implement preflight sampling suppression for agent, tool, protocol, routine,
  and decision-bypass cases.
- Compute switch cost from cache, tool-loop, handoff, turn, and switch history
  evidence.
- Apply `proposal_gain >= switch_margin + protection_weight * switch_cost`.
- Implement deterministic bounded rescue switching.
- Record stay/switch diagnostics in replay and compact headers.

### Phase 3: Adaptation Runtime

- Build candidate sets for `decision`, `tier`, and `global`.
- Implement experience fallback lookup:
  `decision + tier + model -> tier + model -> model`.
- Implement `routing_sampling` with posterior mean and guarded sampling.
- Apply cost, overuse, reliability, latency, and cache adjustments.
- Respect protection preflight, decision bypass, component `observe`, and
  component `bypass`.
- Record candidate scores, sampling metadata, and selected proposals in replay.

### Phase 3b: Future Adaptation Strategies

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

Future strategies should:

- consume typed replay, outcomes, experience, and protection state
- preserve decision-level bypass and component modes
- keep request-time reads local and bounded
- emit the same compact headers and typed replay diagnostics
- prove value with route correctness, cost, latency, cache, and switch metrics

Base routing algorithms that are not online self-improvement mechanisms remain
under the recipe's routing algorithm surface, not under learning.

### Phase 4: Outcome Ingestion

- Add `POST /v1/router/outcomes`.
- Validate source, target, target ref, verdict, score, and bounded metadata.
- Store outcomes linked by `replay_id`.
- Update online experience for `target: model` outcomes.
- Preserve route, policy, and stability outcomes for offline recipe learning.

### Phase 5: Offline Recipe Learning Loop

- Build replay/outcome export for agent experiments.
- Run route correctness, model fit, cost, latency, cache, and switch-rate evals.
- Generate findings for wrong decisions, overuse, underpowered selections,
  excessive switching, and excessive holds.
- Generate candidate recipe patches and optional experience seed packs.
- Run experiments against candidate recipes and report metrics.

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

Do not add contextual feature learning, external hot-path storage, automatic
recipe promotion, or multi-replica shared state before this boundary is closed.

### Phase 6: Storage and Multi-Replica Follow-Up

- Keep request-time state local in the first implementation.
- Define sticky-session requirements for multi-replica deployments.
- Define shared-state timeout and fail-open behavior before adding external
  hot-path storage.
- Define replay-to-state rebuild semantics after the single-replica path is
  stable.

### Deferred Follow-Ups

These are explicit follow-ups, not blockers for the first implementation:

- contextual posterior sampling with a fixed feature vector
- confidence-bound scoring for low-data candidate sets
- preference or pairwise outcome support
- reliability-aware candidate pruning
- replay-derived seed-pack generation and import ergonomics
- automatic or semi-automatic recipe promotion gates
- multi-replica state sharing, sticky-session guidance, and replay-to-state
  rebuild
- richer eval dashboards and long-running regression suites

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
- Eval/inferoa integration proves the offline loop can compute correctness,
  switch, cache, latency, and cost metrics from replay and outcomes.

## Confirmed Decisions

- This is a breaking public API change; old learning config names are removed.
- Public runtime concepts are `adaptation` and `protection`.
- Decision-level controls live under `routing.decisions[].adaptations`.
- `learning.enabled: true` defaults adaptation and protection on unless each
  component is explicitly disabled.
- `routing_sampling` is the day-0 online adaptation strategy and is the default
  when adaptation is enabled.
- `candidate_set` values are `decision`, `tier`, and `global`.
- `tier` candidate set is the union of `modelRefs` from decisions with the same
  `decision.tier`.
- `candidate_set: tier` falls back to `decision` when the matched decision has
  no positive tier.
- Adaptation may use stochastic sampling online when protection allows it.
- Protection has a preflight guard before adaptation sampling and a switch
  guard after adaptation scoring.
- Protection can suppress stochastic exploration without hard-locking the
  current model.
- Protection can guard base-route switches even when adaptation is disabled.
- Protection can allow bounded rescue switches when an agent appears stuck or
  underpowered.
- Compact learning headers use comma-separated tokens and `method=value` pairs,
  not semicolon parameter strings.
- Decision `adaptations.mode: bypass` is how users express sensitive policy
  boundaries.
- `goals` is not part of the first public API.
- The first implementation keeps request-time state in-process and uses replay
  as the durable event log.
- Outcome ingestion is the shared feedback path for users, agents, evals,
  providers, routers, and operators.
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
