# Algorithms

## Overview

An algorithm runs after a decision matches. A **selector** chooses one model
from the decision's `modelRefs` before calling it. An **executor** calls
several candidates and returns one response: it escalates through them, runs a
panel, or executes a workflow.
It does not decide whether the route is eligible; signals and decisions do
that first.

## Key Advantages

- Keeps route eligibility separate from model choice.
- Makes selection and orchestration policy reviewable per decision.
- Supports both one-call selection and bounded multi-model execution.

## What Problem Does It Solve?

A matched route may have several valid model candidates. Algorithms make the
choice explicit: fixed ordering, runtime factors such as quality, latency and
cost, the decision model's answer, or multi-model execution.

## When to Use

Add an algorithm when a decision has more than one candidate or deliberately
runs a multi-model workflow. With one candidate, omit the algorithm unless the
chosen executor supports and needs a single-model execution plan.

## Configuration

Algorithms are decision-local. Each type's settings are the block named after
the type:

```yaml
recipes:
  - name: default
    decisions:
      - name: responsive-route
        description: Prefer the model with the best observed latency.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small-model
          - model: large-model
        algorithm:
          type: multi_factor
          minimum_candidates: 2
          multi_factor:
            weights:
              latency: 1
            latency_percentile: 90
```

Choose an algorithm from the inventory below, then follow its guide for the
required fields and dependencies.

`minimum_candidates` is a common algorithm field:

- A recipe without models can declare it before any models are bound.
- Once an entrypoint publishes the recipe, validation requires that many
  distinct `modelRefs`.
- The Router checks it again after request-time context filtering, so a panel
  or selector never runs silently with a smaller pool than its policy declares.

## Algorithm Inventory

### Selectors

Selectors return one candidate model. Before scoring, the Router drops the
candidates that can't take the request: too little context, an incompatible
backend wire format, or a missing declared capability. The selector then
compares only the models that remain.

- **What the checks read.** They include the conversation content the Router
  keeps, and preview the decision's request-parameter and no-tools policies
  without running those plugins twice.
- **Unannotated models** keep wire-only compatibility checks. Hard quality and
  SLO constraints still apply, and Router Learning checks the same
  capabilities.
- **Fail closed.** An empty pool, or one below `minimum_candidates`, fails the
  request.
- **Final check.** Dispatch validates the final request again after plugins
  change it. A late capability mismatch returns an error instead of rescoring
  or silently switching models; a pinned model is never replaced.

A selector that fails falls back to the first candidate.

| Type | Status | Goal | Main dependency | Guide |
|---|---|---|---|---|
| `static` | supported | Use declared order or fixed domain scores | None | [Static](./selection/static) |
| `random` | supported | Pick uniformly among eligible declared candidates | None | [Random](./selection/random) |
| `multi_factor` | supported | Balance quality, latency, cost, and load with optional SLO filters | Model metadata and live local metrics | [Multi Factor](./selection/multi-factor), [by latency](./selection/latency-aware) |
| `systemone` | supported | Ask the decision model which candidate should answer | System One deployment with Choice support | [System One](./selection/systemone) |

> [!NOTE]
> **Migrating from v0.3 `session_aware`**: In v0.4, the legacy `session_aware` selection algorithm is retired (`algorithm.type=session_aware` is rejected by config validation). Session continuity and agent tool-loop stability are now decoupled from per-decision selection algorithms and managed globally via [Protection](../learning/protection). Decisions configure their base model selection method (e.g., `latency_aware`, `multi_factor`, or `hybrid`) and use `adaptations` for per-decision protection controls (`apply`, `observe`, or `bypass`). See [Migrating from v0.3 Session-Aware Algorithm](../learning/protection#migrating-from-v03-session-aware-algorithm).

### Executors

Executors make additional model calls, in process. Each call runs the matched
decision's plugins and goes to the called model's
`providers.models[].backend_refs`, so every model an executor calls needs a
backend.

:::warning Executors cost more and share content
The extra calls add latency and tokens, and intermediate content goes to
every worker in the run. Apply privacy and provider-boundary decisions first.
:::

Every executor reads two shared settings beside its type:

| Setting | Meaning |
| --- | --- |
| `budget` | `deadline` is the run's wall-clock limit, and `max_calls` the model calls it may make, below the Router's per-request limit. A run that exhausts its budget fails rather than serving a partial answer. |
| `on_error` | What a failed call does. `fusion`, `remom`, `ratings` and `workflows` take `skip` (continue with the calls that succeeded) or `fail`. `escalate` takes `last_attempt` (serve the last judged answer when none is accepted) or `fail`. When a model's answer ends the run, the client gets that model's status if it is a rate limit (429) or a server error (5xx), and 500 otherwise. |

To give an executor its own public model name, publish an entrypoint whose
`model_names` point to a recipe with a decision that runs it.

| Type | Status | Goal | Guide |
|---|---|---|---|
| `escalate` | supported | Call the candidates in order until one answer is accepted, on Chat or System One | [Escalate](./looper/escalate) |
| `ratings` | supported | Return one choice from each candidate with bounded concurrency | [Ratings](./looper/ratings) |
| `remom` | supported | Explore several reasoning paths over multiple rounds, then synthesize | [ReMoM](./looper/remom) |
| `fusion` | experimental | Run an analysis panel and judge/synthesis pass | [Fusion](./looper/fusion) |
| `workflows` | experimental | Execute a bounded static or planner-generated worker flow | [Router Flow](./looper/workflows) |

Treat experimental algorithms as evaluation features: validate them on your
traffic before using them for production routing.

### System One Requests

A recipe published with `api: systemone` uses native execution: its matched
decision runs `escalate` with `escalate.stages`, and returns complete typed
responses.

- **One budget from arrival.** The request's deadline and call count start
  when it arrives. Before a decision is chosen, the ceiling is the largest
  `algorithm.budget` among the recipe's native decisions.
- **The decision tightens it.** Once chosen, the decision's own budget
  applies to its stages, judges and transport retries, and calls already
  spent on signals count toward its `max_calls`.
- **No chat fallback.** Native execution uses an explicit candidate roster
  and quality contract. It never runs chat plugins or falls back to a chat
  model.

<details>
<summary>Removed algorithm types and what replaces them</summary>

A configuration that still names one of these fails at load, and
`vllm-sr config migrate` rewrites it. The migration notes what to review, such
as dropped weights or trained artifacts it no longer uses.

| Removed type | Becomes |
|---|---|
| `confidence` (Looper) | `escalate` with `escalate.accept.confidence` |
| `cascade` (System One) | `escalate` with `escalate.stages` |
| `prompt`, `router_dc` | `systemone` |
| `knn`, `kmeans`, `svm`, `mlp` | `systemone`, which needs no trained model |
| `automix`, `hybrid`, `latency_aware` | `multi_factor` |

</details>

## Operational Boundaries

- **A candidate is a full reference.** Where an algorithm accepts repeated
  model references, a candidate includes its LoRA and reasoning controls, not
  just its model name. Scoring and dispatch keep the exact winning reference,
  and a model-only result that matches several candidates is rejected.
- **LoRA candidates.** The frontend routes by the base model, while the
  provider request names the adapter.
- **Session protection keeps the exact choice.** Router Learning remembers the
  selected candidate's controls, and protection can hold that choice across a
  tool loop. If policy excludes it mid-loop or during a continuation that
  can't move, the request is rejected rather than switched. See
  [Protection](../learning/protection).
- Candidate model names must resolve through `providers.models` in a complete
  config.
- Latency and load observations are local to a Router process; they are not a
  cluster-wide scheduler.
- Executor-generated planner, worker, verifier, judge, and synthesis prompts are
  checked against each target Model's known context window before dispatch.
  Missing context metadata remains eligible for compatibility.
- Validate a complete config with `vllm-sr config validate --config config.yaml`.
