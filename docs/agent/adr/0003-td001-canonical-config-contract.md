# ADR 0003: Use a Canonical Authoring Contract for TD001 Config Surfaces

## Status

Accepted

## Context

TD001 tracks one config concept being represented through multiple overlapping contracts:

- the router persists and validates a flat runtime-oriented `RouterConfig`
- the Python CLI authors a nested `UserConfig` and then translates it back into flat router fields
- the dashboard keeps local TypeScript config DTOs and supports both python-cli and legacy-flat payloads
- the DSL compiler and emitter compile through `RouterConfig` and then denormalize maps back into user-facing YAML
- the operator splits router behavior across `spec.config` and `spec.vllmEndpoints`, then synthesizes additional config sections during reconciliation

This creates repeated schema edits, drift risk, and hidden compatibility seams such as CLI passthrough fields, dashboard hybrid-format handling, and CRD extra-field preservation.

The repo needs one durable source of truth for router authoring shape before the TD001 migration can proceed safely.

## Decision

Adopt a versioned canonical config authoring contract that is separate from the router runtime struct and separate from platform deployment adapters.

The repository will follow these rules:

- `RouterConfig` remains the runtime-oriented compiled form during migration, not the long-term authoring source of truth.
- New router behavior fields must be added to the canonical authoring contract first, then compiled or adapted into runtime and platform-specific forms.
- CLI, dashboard, DSL, and operator surfaces must converge on shared or generated canonical bindings where practical; if generation is deferred, each surface must still depend on a thin adapter around the canonical contract rather than owning a new handwritten schema.
- Kubernetes deployment fields such as service discovery, resources, autoscaling, and similar platform concerns remain adapter-owned and must not redefine the router authoring contract.
- Compatibility work should use dual-read, single-write migration: legacy and hybrid formats may be read temporarily, but canonical format becomes the default write path.

The repository explicitly rejects these alternatives as the long-term TD001 target:

- using the current flat `RouterConfig` as the canonical authoring contract
- using the current Python `UserConfig` unchanged as the canonical contract
- continuing with separate handwritten schemas for router, CLI, dashboard, DSL, and operator paths

## Consequences

- TD001 work should now proceed by introducing a canonical config module plus a compile boundary into runtime config, not by expanding existing translation glue.
- A migration layer, shared bindings, or generation flow will be needed before the dashboard, CLI, DSL, and operator paths can fully converge.
- Some short-term duplication will remain while the repo is in dual-read migration mode, but new drift should be reduced by making the canonical contract the first edit point.
- Cross-surface contract tests become mandatory because correctness now depends on adapters preserving one chosen contract instead of each surface validating itself in isolation.
