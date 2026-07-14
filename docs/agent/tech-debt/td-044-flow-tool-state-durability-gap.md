# TD044: Flow Tool State Durability Follow-Up

## Status

Open; file durability is mitigated, while runtime ownership and backend
lifecycle remain unresolved.

Implementation tracker:
[#2471](https://github.com/vllm-project/semantic-router/issues/2471).

## Owner Plan

[PL0035 Router Flow Workflows](../plans/pl-0035-router-flow-workflows.md)

## Release Relevance

Router Flow can resume workflow worker tool calls through a router-owned state
store. The default file backend survives process restarts on a preserved local
filesystem, and the Redis backend provides the intended shared-state path for
multi-replica deployments. Production readiness still needs Redis integration
coverage and deployment guidance.

## Scope

- `src/semantic-router/pkg/looper/workflows_tool_state.go`
- `src/semantic-router/pkg/looper/workflows_state_store.go`
- `src/semantic-router/pkg/looper/workflows.go`
- `global.integrations.looper.flow.state`

## Summary

Workflow worker tool calls are stored in a pluggable TTL state backend and are
resumed by embedding a workflow state prefix into returned `tool_call_id`
values. The store supports `memory`, `file`, and `redis` backends. Resume
validates that returned tool results match the pending workflow state and the
exact requested `tool_call_id` set before routing the result back to the worker.
The extproc currently constructs a new Workflows Looper and state store per
request. That makes the memory backend unable to resume across HTTP turns and
creates a new unclosed Redis client/pool for every workflow request.

## Evidence

- `workflowToolStateStore` is now an interface implemented by memory, file, and
  Redis backends.
- File-state tests cover cross-looper resume and preservation of prior step
  outputs needed by `access_list` and final synthesis.
- Access-list tests cover both role-level and agent-level visibility, so a
  later worker can see one selected parallel worker output without seeing the
  other worker's tool trajectory.
- Redis state uses a consume-once Lua `GET`/`DEL` path but still needs an
  integration test against a real Redis instance.
- [Looper request factory](../../../src/semantic-router/pkg/extproc/req_filter_looper.go)
  constructs algorithm instances per request.
- [Workflow state-store factory](../../../src/semantic-router/pkg/looper/workflows_state_store.go)
  creates backend clients without a shared runtime owner or close contract.

## Why It Matters

Agentic coding and terminal benchmarks can require several tool turns. Losing
state mid-loop turns a recoverable tool result into a failed workflow and makes
benchmark scores sensitive to process placement rather than algorithm quality.

## Desired End State

Pending workflow state should be production-validated against Redis with TTL,
replica-safe lookup, and documented cleanup/operational behavior. The state
format should continue to preserve per-agent tool trajectories while keeping
access-list visibility isolated between agents.
The state service should be owned once per runtime generation, shared across
requests, and drained/closed when that generation retires.

## Exit Criteria

- Add a Redis integration test for put/take, expiry, and duplicate-consume
  behavior.
- Prove pause/resume across two independent extproc requests for memory, file,
  and Redis backends.
- Keep Redis connection counts stable across sustained workflow traffic and
  close the shared client exactly once on generation retirement.
- Add an end-to-end tool loop smoke using `global.integrations.looper.flow.state`
  with Redis.
- Document production state-store recommendations for local file versus Redis.
- Decide whether the dashboard config UI should expose advanced Flow state-store
  knobs or keep them YAML-only.
