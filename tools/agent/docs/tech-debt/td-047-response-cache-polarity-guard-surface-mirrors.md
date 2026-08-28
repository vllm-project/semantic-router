# TD047: Response-Cache Polarity Guard Is Not Mirrored On CRD And Dashboard Surfaces

## Status

Open

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Not release-blocking. The guard is opt-in and defaults to the existing
behaviour; only operators who configure the router through the Kubernetes CRD
or the dashboard config editor cannot yet reach it.

## Scope

`global.stores.response_cache.polarity_guard` across the router config contract,
the operator CRD, and the dashboard config types.

## Summary

The router config contract owns `polarity_guard` (`mode`, `nli.contradiction_threshold`)
on `ResponseCacheStoreConfig`, and the reference config, validator, docs, and
E2E coverage were updated with it. The two derived config surfaces that mirror
store fields by hand were intentionally left unchanged so the runtime change
stayed on one seam:

- the operator CRD `SemanticCacheConfig` lists cache fields explicitly and has no
  `polarity_guard` block, so CRD-driven deployments cannot enable the NLI tier;
- the dashboard `response_cache` config type lists cache fields explicitly and
  does not expose `polarity_guard`, so the config editor neither shows nor
  round-trips it through schema-driven forms.

The Python CLI needs no change: it does not type `global.stores` and passes the
block through.

## Evidence

- `src/semantic-router/pkg/config/runtime_config.go` — `PolarityGuard` on
  `ResponseCacheStoreConfig`.
- `deploy/operator/api/v1alpha1/semanticrouter_types.go` — `SemanticCacheConfig`
  enumerates `backend_type`, `similarity_threshold`, `max_entries`, `ttl_seconds`,
  `eviction_policy`, backend blocks, and `embedding_model` only.
- `dashboard/frontend/src/types/config.ts` — `response_cache` type enumerates
  `enabled`, `backend_type`, `similarity_threshold`, `max_entries`,
  `ttl_seconds`, `eviction_policy` only.

## Why It Matters

Hand-mirrored config surfaces drift silently: a field that exists in the router
contract but not in the CRD or dashboard type is either unreachable or dropped
on save, and neither failure is reported. The `config-platform-change` skill
requires such gaps to be recorded rather than left implicit.

## Desired End State

`polarity_guard` is reachable from every config surface that exposes
`response_cache`, or the mirrors are generated from the router contract so the
next store field cannot drift.

## Exit Criteria

- The operator CRD accepts `polarity_guard` and the controller translates it
  into the canonical config.
- The dashboard config type and the safety/cache section expose `mode` and
  `nli.contradiction_threshold` and round-trip them on save.
- A test on each surface fails when a `ResponseCacheStoreConfig` field is
  missing from the mirror.
