# Harness Roadmap Execution Plan

This document is the first canonical execution plan for the repository's agent harness. It tracks the maturity snapshot, active refactor loop, and task-by-task progress for the harness workstream itself.

## Goal

- Bring the repository's harness engineering setup to a stable, agent-friendly baseline.
- Keep the harness refactor loop recoverable across multiple long-horizon iterations.

## Scope

- Docs and governance for the agent harness
- Executable rule layer, validation, and scorecard support
- Task tracking for the harness refactor itself

## Exit Criteria

- The harness has one short entrypoint, one indexed docs layer, and one executable contract layer.
- Canonical docs, local rules, ADRs, technical debt, and execution plans are all indexed and mechanically validated.
- The execution-plan mechanism is itself part of the canonical harness.

## Current Maturity Snapshot

- Entry and navigation: `4/5`
  - `AGENTS.md` is now a short entrypoint and `docs/agent/README.md` is the index.
  - Remaining gap: the docs index still needs stronger coverage and testing/module-boundary docs.
- Human-readable system of record: `3/5`
  - Core docs exist, but testing strategy and explicit module-boundary guidance are still under-specified.
- Executable rule layer: `3/5`
  - `make agent-validate` exists, but it does not yet enforce index coverage or required governance sections.
- Contributor interface: `3/5`
  - `CONTRIBUTING.md` points to the harness, but README and PR-facing guidance still need tighter alignment.
- Drift resistance: `2/5`
  - Basic manifest validation exists, but orphan-doc and cross-link decay checks are still missing.

## Target State for This Loop

- The harness has one short entrypoint, one docs index, and one explicit governance doc.
- Testing strategy and module boundaries are documented as durable shared assets.
- Canonical docs are indexed and mechanically validated instead of relying on memory.
- Contributor-facing surfaces point to the same harness entrypoints and validation commands.

## Task List

- [x] `H001` Create a durable harness roadmap with a maturity snapshot and an execution loop.
  - Done when this file exists under `docs/agent/plans/` and becomes the working plan for the current refactor loop.
- [x] `H002` Add a dedicated testing-strategy doc and wire it into the harness index.
  - Done when testing expectations live in a stable doc instead of being implied only by scattered commands.
- [x] `H003` Add a dedicated module-boundaries doc and wire it into the harness index.
  - Done when subsystem boundaries and cross-layer expectations are discoverable without reading several files.
- [x] `H004` Strengthen validation so canonical docs cannot become orphaned or unindexed.
  - Done when `make agent-validate` checks docs inventory and docs-index coverage.
- [x] `H005` Strengthen validation so the docs index and governance docs keep their required sections.
  - Done when the critical harness docs cannot silently decay into arbitrary prose.
- [x] `H006` Align contributor-facing entrypoints with the new harness structure.
  - Done when README, CONTRIBUTING, and the PR template consistently point contributors to the same harness entry and validation path.

## Post-Loop Snapshot

- Entry and navigation: `5/5`
  - `AGENTS.md` is a short entrypoint and `docs/agent/README.md` is the enforced index.
- Human-readable system of record: `5/5`
  - Testing strategy, module boundaries, skill catalog, glossary, ADRs, local-rule inventory, and a durable technical-debt register are now canonical docs under `docs/agent/`.
- Executable rule layer: `5/5`
  - `make agent-validate`, `make agent-report`, `make agent-ci-gate`, and `make agent-scorecard` now expose the harness through stable executable entrypoints.
- Contributor interface: `5/5`
  - README, CONTRIBUTING, the PR template, and local hotspot rules now point to the same harness entry and validation path.
- Drift resistance: `5/5`
  - Temporary working notes are out of the canonical layer, local AGENTS rules are template-validated, CI path filters are checked against the manifests, and code/spec gaps can be tracked durably instead of living only in PR text.

## Remaining Gaps After This Loop

- Freshness is modeled and validated, but the repo still does not enforce calendar-based staleness policies.
- The harness now has ADR support, but additional architecture decisions should be recorded as the contract evolves.
- The scorecard is summary-oriented; if governance needs trend analysis later, that should be added as a separate reporting layer.
- The debt register is now durable and indexed, but future work could add ownership and age metrics per debt item instead of only open-item counting.

## Current Loop

- [x] `H007` Add a harness-specific primary skill and surfaces.
  - Done when harness-only changes resolve to a harness skill instead of startup/runtime skills.
- [x] `H008` Inventory and validate local `AGENTS.md` rules as first-class local harness supplements.
  - Done when local hotspot rule files are indexed, linked, and mechanically validated.
- [x] `H009` Add a human-readable skill catalog and validate `SKILL.md` templates.
  - Done when skill discovery is indexed in docs and skill files cannot silently drift in structure.
- [x] `H010` Add glossary and ADR support for the harness itself.
  - Done when harness terminology and key design decisions are recorded under `docs/agent/`.
- [x] `H011` Add ownership and freshness metadata for canonical harness docs.
  - Done when canonical docs have machine-readable stewardship and freshness expectations checked by `agent-validate`.
- [x] `H012` Validate CI change filters against the executable harness manifests.
  - Done when `ci-changes.yml` path filters are checked against `task-matrix.yaml` and `e2e-profile-map.yaml`.

## Decision Log

- Start from the highest-priority open task in this file.
- After each completed task, update this file before moving to the next one.
- If a durable architectural gap remains after a loop step, also update the matching indexed debt entry under `docs/agent/tech-debt/`.
- If the loop makes a durable governance decision, also update `docs/agent/adr/`.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)

## Current Loop Tasks

- [x] `H013` Split oversized harness validation code into smaller modules.
  - Done when `tools/agent/scripts/agent_validation.py` becomes a thin orchestrator and the validation helpers are separated into smaller doc/skill/CI-focused modules that satisfy the structure gate.
- [x] `H014` Reconcile executable CI classification with manifest-driven harness metadata.
  - Done when `make agent-validate` checks `ci-changes.yml` against `task-matrix.yaml` and `e2e-profile-map.yaml`, and the workflow paths no longer drift from the manifests.
- [x] `H015` Add stronger governance checks for local `AGENTS.md` and skill docs.
  - Done when local hotspot rules and `SKILL.md` files are both indexed and template-validated instead of only being listed.
- [x] `H016` Close the loop with a clean harness validation pass.
  - Done when `make agent-validate` and `make agent-ci-gate` pass for the harness refactor itself and this roadmap records the completed state.

## Next Loop Tasks

- [x] `H017` Split oversized agent gate logic into smaller resolution/report modules.
  - Done when `tools/agent/scripts/agent_gate.py` drops below the warning threshold and change matching, skill resolution, and report assembly live in smaller helpers.
- [x] `H018` Add a harness scorecard/reporting surface for ongoing governance.
  - Done when the harness can emit a compact quality report for docs inventory, local-rule coverage, skill coverage, and validation status through a stable command.
- [x] `H019` Add a durable technical-debt register for architecture and harness drift that cannot be retired in the same change.
  - Done when known code/spec gaps live in a canonical doc under `docs/agent/` instead of in PR text or chat only.
- [x] `H020` Expose technical-debt tracking through harness validation and scorecard reporting.
  - Done when the debt register is indexed, section-validated, and counted by `make agent-scorecard`.
- [x] `H021` Wire technical-debt tracking into contributor and harness close-out guidance.
  - Done when AGENTS, governance, checklist, and harness-governance materials all tell contributors how to record durable architectural divergence.
