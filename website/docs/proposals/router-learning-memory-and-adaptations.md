# Router Learning: Adaptation, Protection, and Recipe Learning

## Status

Design proposal for the vLLM Semantic Router learning system.

The proposal defines one self-improving routing loop with three responsibilities:

- **adaptation**: online model-choice learning from runtime experience
- **protection**: online stability control for sessions, conversations, cache,
  and tool loops
- **recipe learning**: offline agent-driven eval and experiment loops that
  improve recipes

The public API should expose those responsibilities directly. Internal
implementations may use model-experience stores, posterior scoring, guarded
sampling, or identity-scoped state, but those implementation names should not
become the primary product vocabulary.

## Goals

- Keep recipes as the explicit operator control plane.
- Add online learning that can improve model choice while a recipe is deployed.
- Add stability protection so online learning does not cause unnecessary model
  churn inside agent sessions.
- Provide a feedback path that works for users, agents, evals, providers, and
  operators.
- Support offline agent loops that can run evals, run experiments, and propose
  recipe changes.
- Keep all policy boundaries user-controlled through decision configuration.

## Non-Goals

- This proposal does not make the router silently rewrite a deployed recipe on
  the request path.
- This proposal does not require synchronous external storage reads during
  request routing.
- This proposal does not make privacy, security, or local-only behavior a
  hard-coded router concept. Users express those boundaries through decisions.
- This proposal does not expose algorithm-family names or model-training
  internals as first-class public APIs.

## System Model

```text
request
  -> recipe policy
  -> base model selection
  -> protection preflight
  -> adaptation proposal
  -> protection switch decision
  -> final model
  -> replay record
  -> outcome ingestion
  -> online experience update
  -> offline recipe learning loop
```

| Layer | Responsibility |
| --- | --- |
| Recipe policy | Signals, decisions, priority, tier, modelRefs, base algorithms, and per-decision learning controls. |
| Adaptation | Uses online experience and seeded priors to propose a better model. |
| Protection | Controls whether adaptation may explore now, then decides whether the router should switch to the adapted model. |
| Outcome ingestion | Records feedback from users, agents, evals, providers, and operators. |
| Online experience | Low-latency state updated from outcomes and telemetry. |
| Offline recipe learning | Runs evals and experiments, then proposes findings, recipe patches, and experience seed packs. |

The most important separation is:

```text
adaptation changes model choice within the active recipe
recipe learning changes the recipe offline
```

## Recipe Policy

The recipe remains the control plane. It owns:

- signal definitions
- decision rules
- decision `priority`
- decision `tier`
- decision `modelRefs`
- base selection algorithm
- per-decision learning controls

Decision `tier` is an existing decision field. It is a route tier, not a model
capability tier. The current decision engine uses lower numeric tiers first
when tiered selection is active, and only then compares confidence and priority.

Router learning should use `decision.tier` as a sharing key for experience and
candidate-set construction. It should not reinterpret `tier` as a model quality
label.

## Public API

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

      adaptation:
        enabled: true
        candidate_set: decision
        strategy: routing_sampling

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

`global.services.router_replay` remains the event-log storage configuration.
Learning attaches diagnostics and outcomes to replay records; it does not define
a second replay backend.

`adaptation.strategy` defaults to `routing_sampling` when adaptation is enabled.
`routing_sampling` is the first online model-choice learning algorithm. It uses
online experience to score candidate models and may sample under protection
guards. Future online learning algorithms can be added as new `strategy` values
without changing the adaptation/protection split.

### Candidate Sets

`adaptation.candidate_set` controls where adaptation can search for models.

| Value | Candidate Models |
| --- | --- |
| `decision` | Models from the matched decision's `modelRefs`. |
| `tier` | Union of `modelRefs` from all decisions with the same `decision.tier` as the matched decision. |
| `global` | All configured routable models. |

If `candidate_set: tier` is used and the matched decision has no positive
`tier`, the router should fall back to `decision`.

`candidate_set: global` is the broadest mode. It should be guarded more
strictly because it can propose models outside the matched decision's local
candidate set.

### Decision Controls

Decision controls live under `adaptations` because they describe how a matched
decision interacts with the global learning system.

Most decisions can omit this block and inherit global defaults.

To disable all learning adjustments for a sensitive decision:

```yaml
adaptations:
  mode: bypass
```

This is how users express policy boundaries such as local-only, security,
privacy, compliance, or operational containment. The router does not hard-code
which decisions are sensitive.

To control adaptation and protection separately:

```yaml
adaptations:
  adaptation:
    mode: apply
  protection:
    mode: apply
```

Allowed modes:

| Mode | Meaning |
| --- | --- |
| `apply` | The component may affect the final route. |
| `observe` | The component runs and records diagnostics but cannot change the final route. |
| `bypass` | The component does not adjust this decision. |

Decision-level coordination can tune the trade-off between adaptation and
protection:

```yaml
adaptations:
  coordination:
    protection_weight: 1.5
    switch_margin: 0.05
```

Higher `protection_weight` favors stability. Lower `protection_weight` favors
the adapted model when online experience finds a better candidate.

## Runtime Flow

```text
1. decision engine chooses a decision
2. base selector chooses the base model
3. protection preflight reads request state and sets a sampling policy
4. adaptation builds a candidate set and proposes an adapted model
5. protection computes switch cost and decides stay, switch, or rescue
6. router sends the request to the final model
7. replay records base, adapted, final, decision, tier, and diagnostics
8. outcomes update online experience and feed offline recipe learning
```

The ordering is fixed:

```text
base selector -> protection preflight -> adaptation -> protection switch guard -> final model
```

Protection is intentionally split into two guard points:

- **preflight**: decide whether stochastic exploration is allowed for this
  request.
- **switch guard**: decide whether the adapted model is worth using now.

Adaptation answers "which model looks better from experience?" Protection
answers "is it safe and worthwhile to explore or switch at this point in the
agent flow?"

## Adaptation

Adaptation is online model-choice learning. It improves model selection within
the configured candidate set while the recipe remains deployed.

The default strategy is `routing_sampling`.

### Experience Key

Online experience is keyed by decision, decision tier, and model:

```text
decision_id + decision_tier + model
```

Fallback order:

```text
decision_id + decision_tier + model
  -> decision_tier + model
  -> model
```

This lets a new decision use tier-level or model-level evidence during cold
start without treating every decision as identical.

### Experience State

Each experience record should be typed. Initial fields:

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

HTTP success updates reliability and latency evidence. It is not a quality
reward by itself.

### Routing Sampling

For each candidate model, adaptation computes a posterior quality estimate:

```text
alpha = prior_weight * quality_prior + accepted_count + smoothing
beta  = prior_weight * (1 - quality_prior) + rejected_count + smoothing
mean  = alpha / (alpha + beta)
```

When guards allow exploration:

```text
predicted_quality = sample_beta(alpha, beta)
```

When guards block exploration:

```text
predicted_quality = mean
```

The guard decision comes from protection preflight. This keeps stochastic
exploration separate from model lock-in. For example, an agent step carrying
tool results may suppress random sampling for that request, while still
allowing the switch guard to move to a different model if the deterministic
evidence is strong enough.

The score uses fixed router semantics rather than a user-facing weighted-goals
map:

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
penalty. `failed_count` should affect reliability, not task quality.

### Sampling Guards

Sampling is enabled by `strategy: routing_sampling`, but it must be bounded.

Initial guards:

- If the decision uses `adaptations.mode: bypass`, adaptation cannot change the
  route.
- If adaptation mode is `observe`, sampling can be simulated and recorded but
  cannot change the final model.
- If protection preflight returns `sampling: suppressed`, adaptation scores
  candidates by posterior mean instead of drawing a sample.
- Agent/tool/protocol steps suppress sampling by default because they often
  carry implicit state that should not be disturbed by random exploration.
- Routine, low-risk AUTO traffic suppresses sampling when the current model is
  healthy and there is no capability pressure.
- `candidate_set: global` uses stricter cost and reliability guards than
  `tier`, and `tier` uses stricter guards than `decision`.
- Expensive candidates need a larger sampled gain before they can win.
- Candidates with low reliability cannot win through sampling after enough
  failure evidence exists.
- Protection may still reject a sampled model switch when the conversation,
  session, tool loop, or cache state makes switching too costly.

Replay diagnostics should record whether a candidate used posterior mean or a
sampled quality value. For reproducible eval and debugging, record enough
sampling metadata to explain the route, such as sampled value, posterior mean,
and a bounded seed or sample id.

## Protection

Protection is online stability control. It keeps agent sessions coherent and
protects prefix cache, tool-loop continuity, and handoff cost.

Protection is not a policy classifier and does not decide that privacy,
security, or local-only traffic is special. Those boundaries are expressed by
the matched decision through `adaptations.mode: bypass` or component-level
`bypass`.

Protection supports two scopes:

| Scope | Meaning |
| --- | --- |
| `conversation` | Strong protection within one conversation id. A new conversation can re-route while still using session-level history as soft evidence. |
| `session` | Strong protection across the full session. The first final model after idle release becomes the protected baseline until the session idles again or the matched decision bypasses learning. |

Protection reads:

- session id
- conversation id
- current protected model
- turn count
- tool-loop state
- protocol state, such as tool requests and tool results
- cache evidence
- model switch history
- handoff cost
- provider failure and retry evidence
- agent pressure, such as repeated tool failures or failed verification

Protection does not choose the best model. It decides whether adaptation may
explore now and whether the adapted model is worth switching to now.

### Preflight Sampling Guard

Before adaptation samples a candidate, protection produces a sampling policy:

| Policy | Meaning |
| --- | --- |
| `allowed` | Adaptation may draw stochastic samples for eligible candidates. |
| `suppressed` | Adaptation must use deterministic posterior means for this request. |

Sampling should be suppressed by default for:

- agent steps that carry tool-call or tool-result state
- protocol continuation steps where a model change would be hard to interpret
- routine low-risk traffic with a healthy current model
- decisions whose adaptation mode is `observe` or `bypass`

Suppressing sampling is not the same as hard-locking the model. The later switch
guard can still allow a deterministic switch if the adapted candidate has a
large enough advantage.

### Switch Guard

The switch rule is:

```text
switch if adaptation_gain >= switch_margin + protection_weight * switch_cost
```

Where:

- `adaptation_gain` is the adapted candidate's advantage over the current or
  base model.
- `switch_cost` is derived from cache, handoff, tool-loop, session, and switch
  history evidence.
- `switch_margin` is the minimum gain required before any switch.
- `protection_weight` controls how much stability matters.

Expensive candidates should pay a larger effective switch cost unless there is
strong quality, reliability, or rescue evidence. This prevents routine traffic
from drifting toward expensive models because of a small sampled gain.

### Rescue Guard

Protection should also avoid trapping an agent on a weak model. When request
evidence shows that the current model is likely underpowered, protection may
allow a bounded rescue switch even when cache or handoff cost is high.

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
- it does not enable random exploration during tool/protocol steps unless the
  preflight guard allows it

### Guard Actions

Protection should use a small typed action set in replay and compact headers:

| Action | Meaning |
| --- | --- |
| `allow_sampling` | Stochastic adaptation is allowed for this request. |
| `suppress_sampling` | Adaptation must score deterministically for this request. |
| `hold_current` | The adapted model was rejected by the switch guard. |
| `allow_switch` | The adapted model cleared the switch guard. |
| `rescue_switch` | The switch was allowed because rescue evidence outweighed stability cost. |
| `bypass` | The matched decision disabled the protection component. |

The first public API should not expose separate booleans for every guard class.
Enabling protection enables preflight, switch, cost, routine, and rescue guards
with conservative defaults. Public tuning starts with identity, scope, idle
release, switch margin, protection weight, cache weight, and handoff penalty.

## Outcome Ingestion

Feedback enters the system as outcomes. Outcomes are used by both online
adaptation and offline recipe learning.

First-party endpoint:

```http
POST /v1/router/outcomes
```

Example:

```json
{
  "replay_id": "replay_123",
  "source": "agent",
  "target": "model",
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
| `verdict` | `accepted`, `rejected`, `overused`, or `failed`. |
| `reason` | Optional structured reason. |
| `score` | Optional normalized confidence or reward score. |
| `metadata` | Optional bounded metadata. |

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

Examples:

```json
{
  "replay_id": "replay_123",
  "source": "agent",
  "target": "model",
  "verdict": "rejected",
  "reason": "insufficient_capability"
}
```

```json
{
  "replay_id": "replay_124",
  "source": "eval",
  "target": "model",
  "verdict": "overused",
  "reason": "cheaper_model_sufficient"
}
```

```json
{
  "replay_id": "replay_125",
  "source": "eval",
  "target": "route",
  "verdict": "rejected",
  "reason": "wrong_decision"
}
```

## Replay Diagnostics

Response headers should stay compact. Detailed explanation belongs in Router
Replay.

Recommended compact headers:

```text
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=sampled;protection=allow_switch
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=posterior_win;protection=switch_allowed
x-vsr-replay-id: replay_...
```

Headers should identify the active methods and final high-level actions only.
Detailed guard inputs, scores, samples, and state hashes belong in replay.

Replay diagnostics should include:

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
      "adapted_model": "frontier-model",
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
      "adapted_model": "frontier-model",
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

The first implementation should focus on making this loop run end to end. It
does not need to define a full promotion gate before it can be useful.

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

Metrics should be reported both per decision and per decision tier.

## Implementation Plan

### Phase 1: Public API and State Contracts

- Add `global.router.learning.adaptation`.
- Add `global.router.learning.protection`.
- Add decision-level `adaptations.mode`.
- Add decision-level component modes for `adaptation` and `protection`.
- Add typed replay diagnostics for adaptation, protection preflight, and
  protection switch decisions.
- Add typed online experience records keyed by decision, tier, and model.

### Phase 2: Outcome Ingestion

- Add `POST /v1/router/outcomes`.
- Store outcomes linked by `replay_id`.
- Update online experience for model-targeted outcomes.
- Feed route, policy, and stability outcomes into offline recipe learning data.

### Phase 3: Adaptation Runtime

- Build candidate sets for `decision`, `tier`, and `global`.
- Implement typed experience records and fallback lookup.
- Implement `routing_sampling`.
- Record sampling and scoring diagnostics in replay.

### Phase 4: Protection Runtime

- Implement conversation and session scope.
- Implement preflight sampling suppression for agent, tool, protocol, routine,
  and decision-bypass cases.
- Compute switch cost from cache, tool-loop, handoff, turn, and switch history
  evidence.
- Apply the coordination rule:
  `adaptation_gain >= switch_margin + protection_weight * switch_cost`.
- Implement bounded rescue switching for stuck or underpowered agent flows.
- Record stay/switch diagnostics in replay.

### Phase 5: Offline Recipe Learning Loop

- Build replay/outcome export for agent experiments.
- Run candidate recipe experiments.
- Output findings, metrics, recipe patches, and experience seed packs.

### Phase 6: Multi-Replica and Storage Semantics

- Keep request-time state local in the first implementation.
- Define sticky-session requirements for multi-replica deployments.
- Define shared-state timeout and fail-open behavior before adding external
  hot-path storage.

## Confirmed Decisions

- Public runtime concepts are `adaptation` and `protection`.
- Decision-level controls stay under `adaptations`.
- `candidate_set` values are `decision`, `tier`, and `global`.
- `tier` candidate set is the union of `modelRefs` from decisions with the same
  `decision.tier`.
- `candidate_set: tier` falls back to `decision` when the matched decision has
  no positive tier.
- Adaptation uses online experience and `routing_sampling`.
- Protection has a preflight guard before adaptation sampling and a switch guard
  after adaptation scoring.
- Protection can suppress stochastic exploration without hard-locking the
  current model.
- Protection can allow bounded rescue switches when an agent appears stuck or
  underpowered.
- Decision `adaptations.mode: bypass` is how users express sensitive policy
  boundaries.
- `goals` is not part of the first public API.
- Offline recipe learning can propose changes to rules, tier, priority,
  modelRefs, and learning config.
- The first offline loop should run end to end before promotion gates are
  designed.

## Remaining Questions

These details should be confirmed during implementation:

1. What default sampling guard constants should be used for `decision`, `tier`,
   and `global` candidate sets?
2. What default rescue threshold and rescue window should be used for stuck
   agent flows?
3. What retention and decay policy should online experience use?
4. Should outcome ingestion be synchronous with immediate state update, or
   accepted asynchronously with a best-effort update path?
5. Which fields from adaptation and protection diagnostics should appear in
   compact response headers versus replay only?

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
- Outcomes update online model experience and feed offline recipe learning.
- Offline agent loops can produce findings, metrics, candidate recipes, recipe
  patches, and experience seed packs.
- Request-time routing does not require synchronous external storage reads.
- Replay explains base, adapted, and final model choices.
