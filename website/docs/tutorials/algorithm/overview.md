# Algorithms

## Overview

An algorithm runs after a decision matches. It either selects one model from
the decision's `modelRefs` or coordinates several of them through the Looper.
It does not decide whether the route is eligible; signals and decisions do
that first.

## Key Advantages

- Keeps route eligibility separate from model choice.
- Makes selection and orchestration policy reviewable per decision.
- Supports both stateless policies and bounded multi-model execution.

## What Problem Does It Solve?

A matched route may have several valid model candidates. Algorithms make the
choice explicit: fixed ordering, semantic fit, observed latency, multiple
runtime factors, a learned selector, or multi-model orchestration.

## When to Use

Add an algorithm when a decision has more than one candidate or deliberately
runs a multi-model workflow. With one candidate, omit the algorithm unless the
chosen Looper supports and needs a single-model execution plan.

## Configuration

Algorithms are decision-local:

```yaml
routing:
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
        type: latency_aware
        latency_aware:
          tpot_percentile: 90
          ttft_percentile: 95
```

Choose an algorithm from the inventory below, then follow its guide for the
required fields and dependencies.

## Algorithm Inventory

### Selection Algorithms

Selection algorithms return one candidate model.

| Type | Status | Goal | Main dependency | Guide |
|---|---|---|---|---|
| `static` | supported | Use declared order or fixed domain scores | None | [Static](./selection/static) |
| `router_dc` | supported | Match request semantics to model descriptions | Embedding runtime and useful model cards | [Router DC](./selection/router-dc) |
| `latency_aware` | supported | Prefer the candidate with the best observed TTFT/TPOT | Per-process latency observations | [Latency Aware](./selection/latency-aware) |
| `multi_factor` | supported | Balance quality, latency, cost, and load with optional SLO filters | Model metadata and live local metrics | [Multi Factor](./selection/multi-factor) |
| `hybrid` | supported | Blend several selector scores | Component selector inputs | [Hybrid](./selection/hybrid) |
| `automix` | experimental | Optimize an estimated cost-quality value | Candidate pricing and quality metadata | [AutoMix](./selection/automix) |
| `prompt` | experimental | Let a bounded helper model choose from declared candidates | OpenAI-compatible helper model and Looper endpoint | [Prompt](./selection/prompt) |
| `knn` | experimental | Follow similar labeled examples | Trained selector artifact and embeddings | [KNN](./selection/knn) |
| `kmeans` | experimental | Route through learned traffic clusters | Trained selector artifact and embeddings | [KMeans](./selection/kmeans) |
| `svm` | experimental | Apply a learned decision boundary | Trained selector artifact and embeddings | [SVM](./selection/svm) |
| `mlp` | experimental | Apply a learned nonlinear classifier | Trained selector artifact | [MLP](./selection/mlp) |

### Looper Algorithms

Looper algorithms make additional model calls through
`global.integrations.looper.endpoint`. They increase latency and token usage,
and intermediate content is sent to every configured worker involved in the
run.

| Type | Status | Goal | Guide |
|---|---|---|---|
| `confidence` | supported | Escalate sequentially until confidence clears a threshold | [Confidence](./looper/confidence) |
| `ratings` | supported | Return one choice from each candidate with bounded concurrency | [Ratings](./looper/ratings) |
| `remom` | supported | Explore several reasoning paths over multiple rounds, then synthesize | [ReMoM](./looper/remom) |
| `fusion` | experimental | Run an analysis panel and judge/synthesis pass | [Fusion](./looper/fusion) |
| `workflows` | experimental | Execute a bounded static or planner-generated worker flow | [Router Flow](./looper/workflows) |

Treat experimental algorithms as evaluation features: validate them on your
traffic before using them for production routing.

#### Optional per-decision budget

Any Looper algorithm above (Confidence, Ratings, ReMoM, Fusion, Router Flow)
can declare `algorithm.budget` alongside its algorithm-specific block:

```yaml
algorithm:
  type: confidence
  budget:
    max_prompt_tokens: 8000
    max_completion_tokens: 2000
    max_total_tokens: 9000
    max_estimated_cost: 0.50
    max_wall_time_ms: 15000
  confidence:
    confidence_method: hybrid
    threshold: 0.72
```

Every field is optional and defaults to unlimited. Exhausting any one
dimension stops further escalation deterministically (rounds, panel
members, or workflow steps) and returns the best response already
obtained, the same way the algorithm would degrade if it simply ran out of
candidate models. Cost is estimated from actual token usage against the
decision's model pricing, so it is a running total rather than a
pre-call prediction. `algorithm.budget` is a resource policy, not a
call-count limit: it never raises or overrides the router's separate,
non-configurable hard cap on total upstream calls per request, and config
validation rejects `algorithm.budget` on algorithm types that don't
execute through Looper (e.g. `router_dc`, `rl_driven`).

## Operational Boundaries

- Candidate model names must resolve through `routing.modelCards` and
  `providers.models` in a complete config.
- Learned selectors need artifacts produced for the same embedding dimension
  and candidate labels used at runtime.
- Latency and load observations are local to a Router process; they are not a
  cluster-wide scheduler.
- Looper algorithms share request content with their configured workers. Apply
  privacy and provider-boundary decisions before choosing them.
- Validate a complete config with `vllm-sr validate --config config.yaml`.
