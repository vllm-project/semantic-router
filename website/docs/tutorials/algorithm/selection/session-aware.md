# Session Aware

## Overview

`session_aware` selects one model from a decision's `modelRefs` while respecting agentic multi-turn context. It wraps a base selector such as `hybrid`, then applies a router-owned stay-vs-switch policy for sessions, tool loops, idle timeout, handoff cost, switch history, and prefix-cache cost.

Use it when clients send a stable `x-session-id` header or use Response API conversation IDs and you want long-running agent sessions to avoid unnecessary model churn. Provider conversation history is treated as cache; the router keeps its own session memory so model selection can reason about the conversation even when the next turn might move to another backend.

It aligns to `config/algorithm/selection/session-aware.yaml`.

## Key Advantages

- Preserves KV/prefix-cache locality across long-horizon agent sessions.
- Hard-locks active tool loops to avoid mid-loop model changes.
- Lets idle sessions reselect after the cache is likely cold.
- Uses replay-derived remaining-turn priors to be stricter for task families that usually continue for many turns.
- Scales switch cost up for expensive/frontier model checkouts.
- Records `session_policy` in router replay for audit, experiments, and paper/blog analysis.

## What Problem Does It Solve?

Single-turn routers often pick the best model for the latest message only. In long-running agent loops that can churn models between tool calls, waste prefix-cache locality, and make frontier model checkouts unnecessarily expensive. `session_aware` makes the router aware of session continuity before it decides whether a switch is worth the cost.

## When to Use

- Clients set a stable `x-session-id` header or use Response API conversation IDs.
- The route serves agents that call tools over multiple turns.
- Candidate models have materially different costs or prefix-cache behavior.
- You want replayable policy traces for experiments and release validation.

## Configuration

```yaml
routing:
  decisions:
    - name: agentic_routing
      rules:
        operator: AND
        conditions:
          - type: conversation
            name: active_tool_use
      modelRefs:
        - model: qwen3-8b
        - model: qwen3-32b
      algorithm:
        type: session_aware
        session_aware:
          fallback_method: hybrid
          idle_timeout_seconds: 300
          tool_loop_hard_lock: true
          prefix_cache_weight: 0.20
          switch_history_weight: 0.04
```

## Policy

- Tool loops stay on the previous model while tool calls/results are still active.
- Non-idle sessions pay a prefix-cache and handoff penalty before switching.
- Idle sessions can reselect after `idle_timeout_seconds`.
- Expensive/frontier models increase the prefix-cache penalty, so checkout churn is stricter for higher-cost candidates.
- Recent switch history increases the cost of another switch, preventing long-horizon agents from bouncing between models.
- If lookup tables contain `remaining_turn_prior` for the matched category or decision, that prior lifts continuation mass for early turns and decays as the session advances.
- Router replay stores `session_policy`, including base scores, adjusted scores, hard-lock reasons, cache warmth, handoff penalties, and net switch advantage.
- Provider-reported cached prompt tokens are recorded as telemetry and costed with `cached_input_per_1m`; client-facing usage is not rewritten.
