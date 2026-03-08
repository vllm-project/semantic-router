# ADR 0001: Layer the Agent Harness Around a Short Entry and Executable Contract

## Status

Accepted

## Context

The repository accumulated agent guidance across `AGENTS.md`, temporary task lists, local hotspot notes, skills, manifests, and CI workflows. This made it easy for temporary workflows to be mistaken for durable contract and hard for agents to discover the real source of truth.

## Decision

Use a layered harness model:

- `AGENTS.md` is the short entrypoint.
- `docs/agent/*` is the human-readable system of record.
- `tools/agent/*`, `tools/make/agent.mk`, and CI workflows are the executable contract.
- local `AGENTS.md` files are indexed supplements for hotspots, not alternate top-level contracts.
- temporary execution notes do not belong in the canonical harness.

## Consequences

- Agents get progressive disclosure instead of a monolithic rule file.
- Important rules can be validated mechanically through `make agent-validate`.
- Canonical docs and executable rules must evolve together.
- More governance artifacts exist, but they are better layered and easier to audit.
