# TD048: Online Evaluation Assignment Evidence Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

The Evaluation Plane can ship with deterministic replay, live controlled
execution, and offline preference comparison. Production randomized A/B,
interleaving, unbiased off-policy evaluation, and automated adaptation remain
non-blocking extensions and must report insufficient evidence until this gap is
closed.

## Scope

- Router Replay action and outcome evidence
- online evaluation assignment and exposure records
- protected, fallback, and executed action lineage
- propensity-aware preference and off-policy metrics

## Summary

The runtime records route decisions, replay diagnostics, executed model
evidence, and typed replay-linked outcomes, but it does not yet provide one
durable experiment record that joins all information needed for an unbiased
online comparison:

- experiment and treatment assignment;
- alternatives exposed to the assignment mechanism;
- proposed action and behavior probability or propensity;
- protection or fallback action;
- action that actually executed;
- delayed preference or task outcome.

The Evaluation Plane therefore evaluates imported paired preferences and
controlled live runs, but marks production online gates unavailable instead of
inferring missing propensities or writing synthetic feedback to
`/v1/router/outcomes`.

## Evidence

- Router Replay exposes decision, selected-model, execution, latency, token,
  cost, guardrail, and tool-trace evidence.
- `/v1/router/outcomes` accepts authenticated replay-linked typed outcomes with
  idempotency, but is not an experiment-assignment or pairwise-preference API.
- The runtime has no canonical assignment/exposure record that freezes the
  candidate set and behavior propensity for every executed decision.

## Why It Matters

An online win rate without randomized assignment or recorded behavior
probability can be dominated by traffic selection, fallbacks, protection
rules, or missing outcomes. Treating that number as causal would promote a
recipe or model pool on invalid evidence and make off-policy estimators
untrustworthy.

## Desired End State

Online trials have a versioned assignment contract that records exposure,
propensity, protection, fallback, and executed action before the outcome is
observed. Replay, preference, and outcome evidence join through stable IDs;
experiment policy is separately authorized from production routing policy;
promotion remains gated and reversible.

## Exit Criteria

- Define and validate a versioned assignment/exposure contract with stable
  experiment, treatment, unit, decision-point, and replay identifiers.
- Record candidate set, behavior distribution, propensity, proposed action,
  protected/fallback action, and executed action before accepting an outcome.
- Add authenticated, idempotent ingestion for paired or scalar preference
  evidence without overloading `/v1/router/outcomes`.
- Validate randomized A/B or interleaving balance, missingness, contamination,
  and effective sample size in a controlled deployment.
- Add IPS/SNIPS or doubly robust estimators with overlap and clipping
  diagnostics, plus deterministic tests against known fixtures.
- Add canary, rollback, and drift gates before any automatic adaptation uses
  online evaluation results.
