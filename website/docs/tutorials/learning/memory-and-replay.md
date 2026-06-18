# Memory And Replay

## Overview

Router Learning uses Router Memory in layers. The hot path uses in-process
online state. Router Replay remains the durable event log for audit, debugging,
and eval.

## Key Advantages

- Keeps request routing free of required external storage reads.
- Reuses the existing Router Replay service and storage backends.
- Stores full learning diagnostics under `learning.adaptations`.
- Separates payload capture policy from learning behavior.

## What Problem Does It Solve?

Learning adaptations need state from earlier requests, but synchronous storage
calls would make routing fragile. The router keeps low-latency online state in
memory and writes durable event records through Router Replay. Evals and future
prior materializers can read replay records without changing the request path.

## When to Use

- You need to debug why an adaptation stayed or switched models.
- You want evals to replay model choices and cache evidence.
- You need durable audit records for Router Learning decisions.
- You plan to build replay-derived priors later.

## Configuration

Enable Router Replay with the existing service config:

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
```

Learning diagnostics are written into replay records when replay is enabled:

```json
{
  "learning": {
    "adaptations": {
      "session_aware": {
        "action": "stay",
        "reason": "stay_has_best_adjusted_score",
        "identity": {
          "session": {
            "source": "header:x-session-id",
            "status": "present",
            "hash": "4f2a8c0e9b7d3411"
          },
          "conversation": {
            "source": "header:x-conversation-id",
            "status": "present",
            "hash": "0bb97f4a3c812efe"
          }
        },
        "memory_cached_tokens": 2048,
        "last_cache_accounting_source": "backend_reported"
      }
    }
  }
}
```
