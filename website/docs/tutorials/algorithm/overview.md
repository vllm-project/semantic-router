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

`minimum_candidates` is a common algorithm field. A model-free Recipe may
declare it before any Models are bound; once an Entrypoint materializes the
Recipe, validation requires that many distinct `modelRefs`. The same boundary
is checked again after request-time context eligibility filtering, so a panel
or selector does not silently run with a smaller pool than its policy
declares.

## Algorithm Inventory

### Selectors

Selectors return one candidate model. For a full inference request, the Router
filters exact candidate references by context, backend wire support, and
declared model task capabilities **before** scoring. The configured selector
compares the surviving pool; it does not choose an incapable winner and then
replace it with the first compatible sibling.

Capability checks include Router-retained conversation content and preview the
decision's request-parameter and no-tools policies without executing those
plugins twice. Unannotated models retain wire-only compatibility checks. Hard
quality/SLO constraints still apply, and Router Learning checks the same
capabilities when considering additional candidates.

An empty pool or a violated `minimum_candidates` requirement fails closed.
Dispatch validates the final request again after mutations: a late capability
mismatch returns an error rather than restarting decision evaluation, rescoring,
or silently changing the selected candidate. Explicitly pinned models use the
same final validation but are not replaced by another model. A selector that
fails falls back to the first candidate.

| Type | Status | Goal | Main dependency | Guide |
|---|---|---|---|---|
| `static` | supported | Use declared order or fixed domain scores | None | [Static](./selection/static) |
| `random` | supported | Pick uniformly among eligible declared candidates | None | [Random](./selection/random) |
| `multi_factor` | supported | Balance quality, latency, cost, and load with optional SLO filters | Model metadata and live local metrics | [Multi Factor](./selection/multi-factor), [by latency](./selection/latency-aware) |
| `systemone` | supported | Ask the decision model which candidate should answer | System One deployment with Choice support | [System One](./selection/systemone) |

### Executors

Executors make additional model calls. The Router makes each call in process:
the call runs the matched decision's plugins and goes to the called model's
`providers.models[].backend_refs`, so every model an executor calls needs a
backend. The calls increase latency and token usage, and intermediate content
is sent to every configured worker involved in the run.

Every executor reads two shared settings beside its type:

- `budget` bounds the run: `deadline` is its wall-clock limit and `max_calls`
  the number of model calls it may make, below the Router's per-request limit.
  A run that exhausts its budget fails rather than serving a partial answer.
- `on_error` says what a failed call does. `fusion`, `remom`, `ratings` and
  `workflows` take `skip` (continue with the calls that succeeded) or `fail`.
  `escalate` takes `last_attempt` (serve the last judged answer when none is
  accepted) or `fail`.

The Router makes an executor's model calls in process, so
`global.integrations.looper.endpoint` is refused at load, and
`vllm-sr config migrate` drops it. A Looper algorithm's public name is an
entrypoint whose `model_names` name a recipe with a decision that runs it.
`global.integrations.looper.<algorithm>.model_names` is refused at load, and
migrate writes those names as such an entrypoint, for a copy of the default
recipe with only its decisions of that algorithm.

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

A recipe published with `api: systemone` uses native execution. Its matched
decision runs `escalate` with `escalate.stages`, preserving complete typed
responses. The request deadline and physical-call ledger start when the request
arrives. Before a decision is selected, the ceiling uses the largest
`algorithm.budget` deadline and `max_calls` among the recipe's native
decisions; signal-specific timeouts remain upper bounds. After selection, the
decision's `algorithm.budget` tightens the same deadline and ledger for its
stages, judges and transport retries. Calls already spent on signals count
toward that decision's `max_calls`. Native execution uses an explicit candidate
roster and quality contract; it does not run Chat plugins or silently fall back
to a Chat model.

### Removed Algorithms

These types were replaced by the selectors and executors above. A
configuration that still names one fails at load with a pointer to
`vllm-sr config migrate`, which rewrites it:

| Removed type | Becomes |
|---|---|
| `confidence` (Looper) | `escalate` with `escalate.accept.confidence` |
| `cascade` (System One) | `escalate` with `escalate.stages` |
| `prompt`, `router_dc` | `systemone` |
| `knn`, `kmeans`, `svm`, `mlp` | `systemone`, which needs no trained model |
| `automix`, `hybrid`, `latency_aware` | `multi_factor` |

The migration notes what a person must review, such as weights it dropped or
trained artifacts it no longer uses.

## Operational Boundaries

- Where an algorithm accepts repeated model references, the candidate includes
  its LoRA and reasoning controls, not just its model name. Scoring, composition,
  and dispatch retain the exact winning reference. A legacy model-only result
  that matches multiple different candidates is rejected rather than resolved
  to the first reference. For a LoRA candidate, the serving frontend routes by the selected
  base model while the provider request names the adapter.
- Router Learning session memory retains the selected candidate's controls.
  Protection can hold that exact choice across tool-loop continuations even
  when a later base selection prefers another effort of the same model. If that
  exact owner is excluded during an active tool loop that is still on the
  decision that admitted it, or during a nonportable continuation, routing
  rejects the request rather than restoring the owner or falling back to a
  different effort. A drifted decision releases the tool-loop lock and selects
  from the new decision's candidates; opaque provider state stays bound to the
  model that created it. Portable turns may select a new eligible candidate;
  observe and bypass modes do not enforce the protection decision.
- Candidate model names must resolve through `providers.models` in a complete
  config.
- Latency and load observations are local to a Router process; they are not a
  cluster-wide scheduler.
- Executors share request content with their configured workers. Apply
  privacy and provider-boundary decisions before choosing them.
- Executor-generated planner, worker, verifier, judge, and synthesis prompts are
  checked against each target Model's known context window before dispatch.
  Missing context metadata remains eligible for compatibility.
- Validate a complete config with `vllm-sr config validate --config config.yaml`.
