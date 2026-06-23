# Experience

## Overview

Experience is the online Router Learning evidence used by adaptation. It is
kept in process on the request path and summarized from bounded outcomes and
telemetry. Offline recipe learning can export seed-pack artifacts for cold-start
analysis and future warmup flows, but the first public API does not expose a
runtime seed-pack import switch.

## Key Advantages

- Keeps expensive aggregation out of the request path.
- Turns outcomes and telemetry into compact read-time evidence.
- Gives adaptation a typed place for learned model-choice facts.
- Preserves Router Replay as the event log source of truth.

## What Problem Does It Solve?

Some routing evidence needs past requests: model fit, overuse, provider
failures, latency, cache reuse, and effective cost. Reading all events during a
request would be too slow. The router updates compact model experience in
memory and writes durable evidence to Router Replay for offline analysis.

## When to Use

- Adaptation needs aggregate evidence from outcomes and runtime telemetry.
- An operator wants offline evals to explain cold-start model quality evidence.
- Offline recipe learning needs seed-pack artifacts for experiments or future
  warmup flows.

## Configuration

The current public API does not expose `experience.enabled`,
`experience.source`, or a runtime seed-pack import field. If adaptation is
enabled, model experience is part of the `routing_sampling` implementation.

Online experience is keyed by matched decision, decision tier, and model:

```text
decision_id + decision_tier + model
  -> decision_tier + model
  -> model
```

Outcome ingestion updates model-targeted experience. Route, policy, stability,
provider, and router outcomes feed replay diagnostics and offline recipe
learning instead of directly mutating model quality.
