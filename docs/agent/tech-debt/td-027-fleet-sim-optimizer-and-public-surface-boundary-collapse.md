# TD027: Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy

## Status

Open

## Scope

`src/fleet-sim/fleet_sim/optimizer/**`, `src/fleet-sim/fleet_sim/__init__.py`, and adjacent optimizer-facing tests or API callers

## Summary

The fleet-sim optimizer surface still concentrates too many responsibilities into one seam. `fleet_sim/optimizer/base.py` owns analytical sizing, DES calibration, Pareto search helpers, grid-flex analysis, tokens-per-watt analysis, reporting helpers, and several public dataclasses/constants in the same module. At the same time, `fleet_sim/optimizer/__init__.py` and `fleet_sim/__init__.py` both re-export a broad optimizer surface, so adding or changing one optimization feature tends to require edits in internal modeling code and in multiple public export layers. TD016 and TD017 track fleet-sim style and structure migration, but they do not define the narrower architecture target for the optimizer and public API boundary itself.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
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

## Exit Criteria

- New optimizer features no longer require editing `base.py` unless they truly belong to the same analytical kernel.
- Public export changes can be made in one clear owner layer instead of synchronizing parallel symbol lists in both optimizer and root package exports.
- Local AGENT rules explicitly name the optimizer hotspot and the extraction-first path for future changes.
