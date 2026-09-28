# Router Flow Signal and Evaluation Design

This guide defines how benchmark recipes turn request signals into routing
decisions and how those recipes are evaluated. It is for config authors and
benchmark operators; it does not track campaign progress or completed runs.

## Design signals around the task

A signal should describe an observable property of the request, not the model
the author wants to select. Useful properties include:

- benchmark or task family;
- required answer format;
- code, terminal, tool, or long-context requirements;
- ambiguity, risk, and expected reasoning depth;
- latency or response-budget constraints.

A decision combines those signals with an algorithm, `modelRefs`, output
contract, timeout, concurrency, fallback behavior, and trace labels. Keep each
decision narrow enough that its behavior can be explained and evaluated.

## Choose an algorithm for a reason

| Algorithm shape | Use it when | Main configuration to verify |
| --- | --- | --- |
| Direct selection | One suitable model is enough and latency matters | model reference, reasoning mode, output contract, fallback |
| Fusion | Independent answers and a judge or synthesizer can reduce error | worker diversity, synthesis model, minimum successful responses, grounding policy |
| ReMoM | Iterative critique or refinement is likely to help | breadth schedule, synthesis model, compaction, response budget, timeout |
| Static Flow | Solver, verifier, tool, or finalizer roles are known in advance | node roles, edges, access lists, per-step contracts |
| Dynamic Flow | The task needs bounded planning at runtime | planner schema, allowed models and tools, maximum steps and parallelism, fallback |

An `auto` entrypoint is a benchmark-specific policy that may select any of
these shapes. It is not one universal algorithm. A recipe intended to
demonstrate router-side scaling should contain an actual loop or multi-model
decision, rather than labeling a direct call as a routed system.

## Preserve contracts and failure behavior

Every benchmark recipe should have a lowest-priority catch-all decision. The
fallback must preserve the requested output contract and select an explicitly
configured safe route; unmatched traffic must not depend on an implicit model.

Dynamic plans must be validated against the configured `modelRefs`,
`max_steps`, `max_parallel`, access lists, and tool policy. A planner parse
error or timeout should take the configured fallback path, not bypass policy.

Collect enough trace data to diagnose the selection without exposing prompts
or secrets. At minimum, retain the matched signals, selected decision, selected
algorithm, and fallback reason.

## Evaluate in stages

1. Run a dry run or configuration validation before sending traffic.
2. Use a small smoke subset to catch endpoint, schema, output-format, and
   sandbox failures.
3. Classify failures as routing, output contract, planner schema, worker,
   sandbox, judge, scorer, or model errors.
4. Change the smallest responsible layer and rerun only the failed subset.
5. Run a representative slice once the smoke subset is stable.
6. Freeze the recipe, prompts, generation settings, judge, dataset revision,
   and scoring code before collecting a formal result.

Do not tune on the formal run. A post-freeze change creates a new evaluation
identity and requires a fresh result.

## Keep comparisons aligned

External reference values are comparable only when dataset version, subset,
date window, retry policy, tool or sandbox setup, and scorer match. Keep public
values in `public_reference_scores.json` with their source metadata and label
them as references unless they were reproduced locally.

Each locally reported row must identify the exact benchmark-specific recipe.
Store a config snapshot with the result, or record an immutable source revision
plus every command-line override. A score without that identity is a
development probe, not publishable evidence.

## Estimate runtime before a formal run

```text
wall_time ~= ceil(item_count / effective_parallelism)
             * p95_request_seconds
             + setup_time
             + collection_time
```

Use the minimum of evaluation batch size, safe router concurrency, provider
rate limit, and sandbox parallelism as `effective_parallelism`. Loop algorithms
fan out one request to several workers, so provider and sandbox limits often
matter more than local CPU capacity.

## Publication checklist

- The selected adapter is the official or explicitly documented benchmark
  path, not the proxy prompt harness.
- The result contains no missing selected model/benchmark cells.
- Smoke or filtered results name their exact limit and subset.
- Reference columns remain distinguishable from locally measured columns.
- Source revision, immutable image identity, recipe, dataset revision, seed,
  generation settings, concurrency, judge, and scorer are recorded.
- Raw artifacts have been checked for credentials, private endpoints, prompts,
  and user data.
