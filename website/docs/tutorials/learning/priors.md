# Priors

## Overview

Priors are materialized Router Memory views derived from replay records or
operator overrides. They are roadmap state for future adaptations, not required
for the first session-aware implementation.

## Key Advantages

- Keeps expensive aggregation out of the request path.
- Turns replay history into compact read-time evidence.
- Gives future adaptations a shared place for learned routing facts.
- Preserves Router Replay as the event log source of truth.

## What Problem Does It Solve?

Some routing evidence needs many past requests: quality gaps, handoff penalties,
remaining-turn estimates, model health, or tenant preferences. Reading all
events during a request would be too slow. Priors are precomputed snapshots that
future adaptations can load from local memory.

## When to Use

- A learning adaptation needs aggregate evidence from replay.
- An operator wants to override known model quality or handoff costs.
- Offline evals produce priors that should influence online routing.
- A future bandit or personalization adaptation needs durable state.

## Configuration

The first implementation does not require public priors config. The intended
future shape keeps priors under Router Learning memory:

```yaml
global:
  router:
    learning:
      memory:
        priors:
          enabled: true
          source: router_replay
```

Until priors are implemented, session-aware learning uses in-process online
state plus model pricing, cache accounting, and switch history.
