# TD027: Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy

## Status

Open

## Scope

`src/fleet-sim/fleet_sim/optimizer/**`, `src/fleet-sim/fleet_sim/__init__.py`, and adjacent optimizer-facing tests or API callers

## Summary

The fleet-sim optimizer surface still concentrates too many responsibilities into too few seams. `fleet_sim/optimizer/base.py` now delegates threshold Pareto analysis to `optimizer/threshold.py`, but it still owns analytical sizing, DES calibration and verification, simulation reporting, compatibility exports, and several public dataclasses/constants in the same module. `fleet_sim/optimizer/__init__.py` is now the primary threshold Pareto export owner, but root-package curation and other optimizer feature boundaries still need tightening. TD016 and TD017 track fleet-sim style and structure migration, but they do not define the narrower architecture target for the optimizer and public API boundary itself.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
- [src/fleet-sim/fleet_sim/optimizer/threshold.py](../../../src/fleet-sim/fleet_sim/optimizer/threshold.py)
- [src/fleet-sim/fleet_sim/optimizer/__init__.py](../../../src/fleet-sim/fleet_sim/optimizer/__init__.py)
- [src/fleet-sim/fleet_sim/__init__.py](../../../src/fleet-sim/fleet_sim/__init__.py)
- [src/fleet-sim/AGENTS.md](../../../src/fleet-sim/AGENTS.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [docs/agent/tech-debt/td-016-fleet-sim-shared-ruff-contract-gap.md](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [docs/agent/tech-debt/td-017-fleet-sim-structure-gate-migration-gap.md](td-017-fleet-sim-structure-gate-migration-gap.md)

## Why It Matters

- Optimizer changes currently have a broad blast radius: one feature can touch analytical math, DES verification, reporting helpers, and both package export layers at once.
- The public optimizer surface is too easy to widen accidentally because `optimizer/__init__.py` and the root `fleet_sim/__init__.py` both mirror a large symbol inventory.
- `base.py` is already large enough that structure-rule relief exists for the fleet-sim optimizer subtree, which means contributors need a clearer architecture target than "make the file smaller later."

## Desired End State

- Analytical sizing, DES verification/calibration, power or flexibility analysis, and reporting helpers live behind narrower sibling modules instead of one growing `base.py`.
- `fleet_sim/optimizer/__init__.py` becomes the primary optimizer export seam, and the root `fleet_sim/__init__.py` curates a deliberate subset instead of mirroring every optimizer detail.
- Optimizer-facing tests validate behavior through those narrower seams rather than depending on one all-purpose module.

## Progress

- 2026-05-24: `ThresholdResult`, `threshold_pareto`, Pareto marking, and `print_threshold_pareto` moved from `optimizer/base.py` to `optimizer/threshold.py`.
- 2026-05-24: `optimizer/__init__.py` exports the threshold Pareto API from the new module, while `optimizer/base.py` keeps compatibility exports for existing direct imports.
- 2026-05-24 validation: `make vllm-sr-sim-test` passed with 304 tests.

## Exit Criteria

- New optimizer features no longer require editing `base.py` unless they truly belong to the same analytical kernel.
- Public export changes can be made in one clear owner layer instead of synchronizing parallel symbol lists in both optimizer and root package exports.
- Local AGENT rules explicitly name the optimizer hotspot and the extraction-first path for future changes. This is satisfied by `src/fleet-sim/fleet_sim/optimizer/AGENTS.md`; the remaining work is implementation extraction.
