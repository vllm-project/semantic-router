# Experience

## Overview

Experience is the materialized Router Learning evidence derived from replay
records, evals, or operator overrides. It is the roadmap layer for historical
facts that should influence future routing without adding storage reads to
every request.

## Key Advantages

- Keeps expensive aggregation out of the request path.
- Turns replay history into compact read-time evidence.
- Gives adaptations a shared place for learned routing facts.
- Preserves Router Replay as the event log source of truth.

## What Problem Does It Solve?

Some routing evidence needs many past requests: quality gaps, handoff
penalties, remaining-turn estimates, reward statistics, model health, or tenant
preferences. Reading all events during a request would be too slow. Experience
is precomputed into snapshots that adaptations can load into local memory.

## When to Use

- A learning adaptation needs aggregate evidence from replay or evals.
- An operator wants to override known model quality or handoff costs.
- Offline evals produce experience that should influence online routing.
- A learning adaptation needs materialized evidence beyond its live in-process
  states.

## Configuration

The current implementation does not require public experience config.
Experience gets a public API only when materializers and refresh semantics are
available.

Until public experience materializers are implemented, session-aware learning
uses in-process states plus model pricing, cache accounting, switch history, and
internal lookup-table migration material when present. Bandit, Elo, and
personalization record missing experience views until replay/eval materializers
exist.
