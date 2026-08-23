# Memory And Replay

## Overview

Router Learning uses in-process online state on the hot path. Router Replay can
record events when enabled; persistence across restarts requires a durable
backend. Request routing does not depend on synchronous replay-store reads.

## Key Advantages

- Keeps hot-path learning reads local and bounded.
- Can preserve replay evidence for audit and evaluation when a durable backend
  is configured.
- Separates mutable protection state from long-lived replay evidence.
- Gives offline recipe learning replay data without adding replay-store reads
  to request routing.

## What Problem Does It Solve?

Learning needs history, but request routing cannot scan storage or replay logs
on every call. The router keeps compact in-process state for protection and
adaptation. When replay is enabled, it also writes records for audit,
debugging, outcomes, and offline recipe experiments; their durability depends
on the selected backend.

## When to Use

- You need detailed learning diagnostics beyond compact response headers.
- You want evals or agents to inspect routing evidence after the request.
- You want outcomes to update online experience while remaining linked to a
  replay record.
- You plan to enable replay and run offline recipe learning from production or
  test data.

## Layers

| Layer | Hot path | Responsibility |
| --- | --- | --- |
| Protection state | Yes | Current protected model, identity scope, turn count, cache/tool-loop evidence, and switch history. |
| Model experience | Yes | Quality, overuse, reliability, latency, cache, and cost evidence for adaptation. |
| Router Replay | No | Optional route, response, outcome, and learning diagnostics; durability depends on the backend. |
| Offline recipe learning | No | Evals, findings, candidate recipes, recipe patches, and experience seed packs. |

## Configuration

Enable Router Replay with the existing service config:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
```

This example uses Postgres for persistence. The default `memory` backend is
process-local and loses records on restart.

Learning diagnostics are written into replay records when replay is enabled:

```json
{
  "learning": {
    "protection_preflight": {
      "action": "allow_sampling",
      "scope": "conversation",
      "reason": "no_tool_or_protocol_state"
    },
    "adaptation": {
      "strategy": "routing_sampling",
      "candidate_set": "decision",
      "base_model": "small-model",
      "proposal_model": "frontier-model",
      "reason": "posterior_win"
    },
    "protection": {
      "action": "allow_switch",
      "base_model": "small-model",
      "proposal_model": "frontier-model",
      "final_model": "frontier-model",
      "switch_cost": 0.03,
      "reason": "switch_allowed"
    }
  }
}
```

Raw session, conversation, user, tenant, and workspace identifiers should not be
stored in learning diagnostics. Store bounded hashes and source/status fields.

## Outcomes

Submit typed feedback through the replay-linked outcome endpoint:

```http
POST /v1/router/outcomes
```

```json
{
  "replay_id": "replay_123",
  "target": "model",
  "target_ref": "model/frontier-model",
  "target_revision": 3,
  "verdict": "good_fit",
  "reason": "solved_complex_task",
  "score": 1.0
}
```

Authenticate this public-listener request with the same logical API key (or a
delegated inference session) that owns the replay and include an
`Idempotency-Key` header. `target: model` outcomes enter a durable,
revisioned learning projection. Other target types remain immutable evidence
for replay and offline recipe learning unless a typed consumer exists.

## Recipe Learning Command

Run the offline loop from replay:

```bash
vllm-sr eval recipe-learning \
  --replay-file replay.json \
  --recipe-file config.yaml \
  --output-dir ./router-learning-report
```

The command writes:

- `metrics.json`
- `findings.json`
- `experiment_results.json`
- `recipe_patch.json`
- `experience_seed_pack.json`
- candidate recipe YAML files when `--recipe-file` is provided
