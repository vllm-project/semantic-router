# Glossary

## Agent Harness

The repository-specific rule system that combines docs, manifests, scripts, Make targets, and CI checks so coding agents can navigate and validate work reliably.

## Canonical Harness

The stable shared contract represented by `AGENTS.md`, `docs/agent/*`, `tools/agent/*`, `tools/make/agent.mk`, and the related CI workflows.

## Primary Skill

The project-level change archetype selected first for a task. Primary skills decide the default surfaces and validation posture.

## Fragment Skill

A narrower skill that supports a primary skill with subsystem-specific implementation guidance.

## Support Skill

A skill that does not drive task selection but provides close-out or cross-cutting guidance.

## Legacy Reference Skill

A compatibility bridge retained to help contributors migrate from older task-entry habits to the project-level skill model.

## Change Surface

A stable project-level contract area such as `router_runtime`, `python_cli_schema`, or `harness_exec`. Surfaces connect changed paths, skills, and validation expectations.

## Local Rules

Directory-local `AGENTS.md` files that supplement the shared harness near hotspot code without replacing the canonical contract.

## Signal Layer

The layer that extracts structured information from request or response content. Heuristic, semantic, or learned text-understanding features belong here first.

## Decision Layer

The layer that combines signals and other route conditions with boolean control logic. Matching, gating, thresholds, and priority-driven branching belong here first.

## Algorithm Layer

The layer that runs after a decision matches and chooses among multiple candidate models for that decision. Latency-aware, cost-aware, or similar per-decision model-selection policies belong here first.

## Plugin Layer

The layer that performs processing because a routing or decision result requires it. Cache behavior, prompt rewriting, and similar post-decision handling belong here.

## Global Level

The layer for intentionally cross-cutting behavior that should apply across the whole router or config model rather than living inside one signal, decision, algorithm, or plugin.
