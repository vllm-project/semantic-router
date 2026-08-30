# TD049: Attested Live Evaluation Target Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

The generic runtime target can ship as an E0 diagnostic for routing reachability,
one multimodal probe, and bounded concurrency. Model-pool, joint-system, and
promotion-grade capacity claims remain unavailable until this gap is closed.

## Scope

- server-owned direct-arm execution
- routed-request correlation and executed-model identity
- live model/runtime revision attestation
- repeatable load profiles and declared SLOs

## Summary

The Router owns logical model selection and Envoy owns upstream transport, but
the current runtime API has no evaluation-only seam that executes a declared
logical arm and returns attested physical execution evidence. Sending a model
name through the normal routed endpoint would not prove that the requested arm
executed. The bounded load probe also lacks the warm-up, duration, repetitions,
arrival distribution, and SLO contract needed for a capacity claim.

The catalog therefore advertises neither model-pool nor joint evaluation for a
generic runtime target. Current routing, multimodal, and capacity probes remain
E0 diagnostics even when HTTP requests succeed.

## Evidence

- Runtime target arms contain public logical identities and digests but no
  callable endpoint or server-minted execution handle.
- Routed chat evidence does not yet bind route ID, selected logical arm,
  executed backend revision, response, and retry/fallback lineage in one
  evaluation record.
- The diagnostic load executor uses a small bounded concurrency set rather
  than a qualified load campaign.

## Why It Matters

Without an attested direct-arm matrix, pool oracle, marginal contribution, and
router regret can be computed against the wrong physical model. Without a
frozen load contract, throughput and latency numbers are neither comparable nor
safe inputs to placement or promotion decisions.

## Desired End State

The serving control plane exposes a short-lived, evaluation-only target
capability. It binds opaque arm handles to immutable model and runtime
revisions, executes direct-arm and routed calls, and emits signed correlation
evidence without revealing infrastructure addresses. Capacity campaigns bind a
workload, arrival process, warm-up, duration, repetitions, SLO, and resource
snapshot before execution.

## Exit Criteria

- Define a versioned direct-arm target contract with short-lived authorization
  and server-owned opaque arm handles.
- Return executed logical arm, immutable model/runtime digests, route and
  request correlation, retry/fallback lineage, and token/latency accounting.
- Prove direct and routed calls reach the intended physical revisions in an AMD
  integration test without exposing host or endpoint details in public output.
- Define load-profile, warm-up, repetition, arrival, SLO, and resource-snapshot
  contracts and validate saturation/headroom reproducibly.
- Promote model-pool, joint, or capacity evidence above E0 only when every row
  carries the required attestation.
