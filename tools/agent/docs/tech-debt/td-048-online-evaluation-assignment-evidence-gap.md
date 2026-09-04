# TD048: Online Evaluation Assignment Evidence Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

The Evaluation Plane can consume a sealed production experiment ledger and
fail closed when that capability is absent. It must not claim unbiased online
preference lift, off-policy quality, or automatic adaptation until the runtime
also owns assignment and exposure evidence.

## Scope

- production experiment assignment and exposure records;
- protected, fallback, and executed action lineage;
- propensity-aware preference and off-policy evaluation.

## Summary

The worker fetches a server-configured production ledger through the Go-owned
broker and validates its runtime, policy, topology, Mixture, assignment, and
outcome lineage. The repository does not yet produce that ledger or expose an
authenticated, idempotent assignment/preference ingestion contract. Endpoint
presence therefore proves connectivity to an external capability, not that
every routed decision recorded its candidate set, propensity, fallback, and
executed action before an outcome arrived.

This is an explicit capability boundary, not permission to synthesize missing
propensities or overload `/v1/router/outcomes` with experiment semantics.

## Evidence

- `production_experiment_ledger.py` fetches and validates a brokered external
  ledger but does not create assignments or exposures.
- `live_runtime_collection.py` requires the configured ledger for production
  preference collection and fails closed when it is absent.
- The runtime exposes no canonical assignment/preference ingestion API that
  records behavior probability before the delayed outcome.

## Why It Matters

Traffic selection, protection rules, fallbacks, and missing delayed outcomes
can dominate an apparent online win rate. Promotion or adaptation based on
that evidence would be scientifically invalid even when the report is
operationally reproducible.

## Desired End State

The runtime owns a versioned assignment ledger and preference-ingestion seam.
Replay, assignment, exposure, fallback, execution, and delayed outcomes join
through stable IDs, while promotion and adaptation remain separately
authorized, reversible, and statistically qualified.

## Exit Criteria

- Define one versioned assignment/exposure contract with stable experiment,
  treatment, decision, replay, and outcome identities.
- Persist candidate set, behavior distribution and propensity, proposed
  action, protected/fallback action, and executed action before outcome
  ingestion.
- Add authenticated and idempotent paired/scalar preference ingestion without
  overloading the replay outcome endpoint.
- Validate balance, overlap, clipping, missingness, contamination, and
  effective sample size in a controlled deployment.
- Add deterministic IPS/SNIPS or doubly robust estimator fixtures plus canary,
  rollback, and drift gates before online evidence drives adaptation.
