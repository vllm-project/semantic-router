# Memory And Replay

## Overview

Router Learning uses in-process online state on the hot path. Router Replay can
record events when enabled; persistence across configuration reloads and restarts
requires a durable backend. Request routing does not depend on synchronous
replay-store reads.

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

This example uses Postgres for persistence. The default `memory` backend loses
records on configuration reload or restart, even when a reload keeps the router
process running. Use durable storage to keep session traces available in the API
and Dashboard while tuning recipes.

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
POST /api/v1/observability/outcomes
```

```json
{
  "replay_id": "replay_123",
  "source": "agent",
  "target": "model",
  "target_ref": "frontier-model",
  "verdict": "good_fit",
  "reason": "solved_complex_task",
  "score": 1.0
}
```

`target: model` outcomes update online model experience. `target: route`,
`target: policy`, `target: stability`, `target: provider`, and
`target: router` outcomes are kept for replay and offline recipe learning unless
a typed online consumer exists.

## Recipe Learning Command

Run the offline loop from replay:

```bash
vllm-sr optimize recipe-learning \
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

### Choose a default model from agent-run outcomes

An agent run makes many model calls, but its outcome, such as whether the
task's tests passed, belongs to the whole run. Report it once, as a
`target: model` outcome on one of the run's replay records, and send the same
`x-session-id` and `x-conversation-id` on every call of the run. Under the
default `conversation` protection scope the router records the conversation
only when both headers are present. Recipe learning then groups records into
runs, adds up each run's recorded cost, and reports for each decision and
model how many runs succeeded, with a 95% interval, and what a run cost:

```bash
vllm-sr optimize recipe-learning \
  --replay-file replay.json \
  --recipe-file config.yaml \
  --value-per-success 1 --value-per-success 4 \
  --output-dir ./router-learning-report
```

`--value-per-success` states what one successful run is worth in the replay's
cost currency. For each value, `metrics.json` names the model with the highest
expected value minus cost under `runs.per_decision`, keyed `recipe/decision`
by the recipe that served the run. When that model wins even at the low end of
its interval against every other model's high end, and it is not already the
decision's first `modelRefs` entry in that recipe, the command adds a
`default_model` finding and a patch that moves it to the front. Models
need at least 20 runs with outcomes before they are compared. Without `--limit`,
`--value-per-success` reads every replay record from the endpoint so runs are
not cut short.

A run whose calls went to more than one model is counted under
`switched_runs` and left out of the rates, because its outcome and cost belong
to several models. That covers a run the router moved, a harness falling back
to another model, and a subagent that shares the run's session but runs on a
different model.
