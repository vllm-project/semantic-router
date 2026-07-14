# Routing Scope: Why the Router Stays Per-Query, and Where Capacity Belongs

**Version:** 1.0
**Authors:** vLLM Semantic Router Team
**Status:** Decision record

## Abstract

A recurring question is whether the semantic router should adopt **batch-level,
capacity-aware routing** — jointly assigning a whole batch of concurrent queries to
models under a shared cost budget and per-model concurrency limits, as proposed in
recent LLM-routing research (e.g. *Robust Batch-Level Query Routing for LLMs under
Cost and Capacity Constraints*). This document records why we **do not** fold
batch-level or capacity-aware optimization into the routing decision, and instead keep
the router **per-query** and treat capacity as a **load-balancing concern** layered
below routing. The short version: the router already captures the value that matters
(iso-quality cost savings), the batch/capacity headroom is small and unstable in our
own measurements, and capacity control belongs in the serving/LB layer for clean
separation of concerns.

## 1. What the router does today (and why it is the right thing)

The semantic router is a **per-query classifying gateway**. For each request it
predicts, from the prompt, which model in the pool is the best target and forwards the
request — a single, sub-millisecond decision on the Envoy `ext_proc` hot path.

The value proposition this delivers is **cost savings at near-iso quality**: route
easy queries to cheap models and hard queries to capable ones, so the pool matches a
strong single model's quality at a fraction of the cost. On a public multi-task
routing benchmark (RouterBench, 11 models), a leakage-free trained per-query router
recovers substantial savings for a small quality trade:

| Quality retained (vs best single model) | Cost vs best single | Savings |
|---|---|---|
| 99% (−0.8 pts) | ~55% | **~45%** |
| 98% (−1.6 pts) | ~32% | **~68%** |
| 95% | ~19% | ~81% |

*(Trained gradient-boosted per-model quality estimator; random-row split; router picks
the target with no access to test labels. Absolute numbers depend on the pool and the
estimator — see §4.)*

This is the legitimate router pitch — **trade a fraction of a quality point for large
cost savings** — and it is entirely a **per-query** effect. It requires no batching and
no capacity model.

## 2. What batch-level, capacity-aware routing would add

The research proposal reframes routing as a per-batch constrained optimization. Instead
of picking a model per query independently, it takes a **batch of N concurrent queries**
and solves an assignment that maximizes total predicted quality subject to:

- a **shared cost budget** across the batch, and
- **per-model capacity limits** (max concurrent queries a model instance can serve).

The claimed benefit over per-query routing is better behavior under **non-uniform or
adversarial batches** (many hard/expensive queries arriving together), plus a robust
variant that uses the lower bound of a predicted-quality interval to hedge estimation
error.

If we ever pursued this, the honest design would **not** put a solver on the hot path.
It would look like this — routing stays per-query and fast; capacity is a separate
load-balancing layer that the router is merely *aware* of:

```
                    ┌─────────────────────────────────────────┐
   request ───────► │  Semantic Router (ext_proc, per-query)   │
                    │  prompt → predict best model → target     │
                    └───────────────┬───────────────────────────┘
                                    │ target model (+ fallbacks)
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │  Capacity / Load-Balancing layer          │
                    │  • per-model concurrency & queue depth    │
                    │  • least-loaded / spillover / admission   │
                    │  • fleet sizing (see fleet-sim)           │
                    └───────────────┬───────────────────────────┘
                                    ▼
                         backend model instances (GPUs / APIs)
```

In this shape, **capacity is a serving concern, not a routing concern**: the router
emits a preferred model (and ordered fallbacks); the LB layer honors real-time
concurrency limits, spills over when a pool saturates, and is sized offline by the
[fleet simulator](../fleet-sim/overview). Batch-level *joint* optimization would only
ever be justified for **bulk / asynchronous** workloads, never for interactive traffic.

## 3. Why we do not fold this into the routing decision

### 3.1 Hot-path conflict

The router is per-request and sub-millisecond. Batch-level assignment requires
collecting N concurrent queries and solving an integer/constrained program per batch.
Grafting that onto `ext_proc` would convert a fast, stateless classifier into a
batching, solver-bound component — destroying the property that makes the router
deployable as an inline gateway.

### 3.2 The measured headroom is small — and an easy artifact

We reproduced the batch-vs-per-query comparison on RouterBench. At first glance
batch-level routing looked far ahead of per-query — **+12 to +16 points** under
adversarial batching. **That gap is an accounting artifact.** It appears only because
the batch arm was given a *larger total budget*: the per-batch budget was set to the
most expensive batch's cost and applied to every batch, letting the batch arm spend
**1.3×–6.2× more** in total. When both arms are held to the **same total spend**, the
gap collapses:

| Batching | Budget ratio (batch/per-query) | Naive gap | **Matched-total-cost gap** |
|---|---|---|---|
| Random | 1.1–1.4× | +1.2 to +2.4 pts | **−0.2 to −0.1 pts** |
| Adversarial | 2.1–6.2× | +12 to +16 pts | **+0.1 to +0.5 pts** |

At equal cost, batch-level joint routing gives **≈0** over per-query. The
"adversarial wins big" story was just "adversarial batches concentrate cost → higher
per-batch budget → the batch arm gets to spend more."

### 3.3 Capacity coupling helps only when it binds hard — and is fragile

The one place batch-level assignment *can* beat greedy per-query routing is when
**per-model capacity binds**: good models have scarce slots, so a joint assignment can
allocate them to the queries that need them most, while greedy first-come routing
wastes them. We measured exactly this — the effect is real but small, and it reverses
under cost pressure:

| Scenario | Batch − greedy (accuracy) |
|---|---|
| Capacity non-binding (control) | −0.1 pts (≈0, as expected) |
| Capacity binding, moderate | +0.3 pts |
| Capacity binding, tight | **+1.2 pts** |
| Capacity binding **+ cost pressure** | **−1.2 to −3.4 pts** |

So the coupling buys ~1 point in the best case (tight capacity, no cost pressure) and
**turns negative** once you also push on cost. That is not a stable enough win to
justify a solver on the routing path.

### 3.4 Separation of concerns

Real-time capacity — concurrency limits, queue depth, saturation, spillover — is a
**serving-layer** property that changes on a per-second basis and is specific to the
deployment's hardware. It is best handled by load balancing and admission control, and
sized offline by the fleet simulator. Encoding it inside the semantic routing decision
couples a slow-moving, semantics-driven choice (which model is *good* for this prompt)
to a fast-moving, infrastructure-driven one (which instance has a *free slot*). Keeping
them separate keeps the router simple, fast, and portable across deployments.

## 4. Honest limitations of the measurements

- **Benchmark, not production traffic.** RouterBench is multiple-choice-style tasks
  with 0/1 correctness. Real traffic is open-ended, where "which cheap model is also
  correct" is harder to define and label — the true bottleneck for the per-query value
  prop in §1 is obtaining reliable per-model quality labels on live traffic, not the
  routing algorithm.
- **Estimator-dependent.** The cost-savings frontier in §1 depends on the quality
  estimator. A stronger estimator on the same features roughly doubled near-iso savings
  (17% → 45% at 99% quality) versus a weaker one — the headroom worth chasing is in the
  **estimator**, not in batch/capacity optimization.
- **Cannot match best-single quality exactly.** When one model dominates on hard
  queries, no router reaches 100% of its quality at any cost; the value is the near-iso
  trade, not strictly beating the best model.
- **Operating point.** The cost knob (λ) should be tuned on a validation split in
  production, not on the evaluation set.

## 5. Decision

- **Keep routing per-query.** It captures the real, sizable value (iso-quality cost
  savings) on the hot path.
- **Do not fold batch-level or capacity-aware optimization into the routing decision.**
  Measured headroom over matched-cost per-query routing is ≈0 (batch) to ~1 point
  (tight capacity), and unstable under cost pressure.
- **Treat capacity as a load-balancing layer** below routing (least-loaded / spillover
  / admission), sized by the [fleet simulator](../fleet-sim/overview). The router may
  emit ordered fallbacks the LB layer can use, but does not solve a capacity program
  itself.
- **Invest in the quality estimator instead** — that is where the remaining cost-savings
  headroom lives.

## 6. When to revisit

Reopen this decision if any of the following hold: capacity genuinely binds hard in
production (scarce good-model slots that dominate the cost picture), a dedicated
**bulk/asynchronous** routing path is added where per-batch solve latency is acceptable,
or the per-query quality estimator is strong enough that the remaining headroom
demonstrably shifts to allocation rather than estimation.
