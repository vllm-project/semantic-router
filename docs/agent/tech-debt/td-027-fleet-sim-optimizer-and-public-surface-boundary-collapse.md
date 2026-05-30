# TD027: Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

`src/fleet-sim/fleet_sim/optimizer/**`, `src/fleet-sim/fleet_sim/__init__.py`,
and adjacent optimizer-facing tests or API callers.

## Summary

The fleet-sim optimizer surface still concentrates too many responsibilities
into too few seams. `fleet_sim/optimizer/base.py` has started to shed threshold
Pareto analysis into `optimizer/threshold.py`, but it still owns analytical
sizing, DES calibration and verification, simulation reporting, public
dataclasses/constants, and export policy. `optimizer/__init__.py` is now the
primary optimizer export owner, but root-package curation and the remaining
optimizer feature boundaries still need tightening.

TD016 and TD017 cover fleet-sim style and structure gate migration. TD027 owns
the narrower architecture target for optimizer responsibility and public API
ownership.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
- [src/fleet-sim/fleet_sim/optimizer/threshold.py](../../../src/fleet-sim/fleet_sim/optimizer/threshold.py)
- [src/fleet-sim/fleet_sim/optimizer/__init__.py](../../../src/fleet-sim/fleet_sim/optimizer/__init__.py)
- [src/fleet-sim/fleet_sim/__init__.py](../../../src/fleet-sim/fleet_sim/__init__.py)
- [src/fleet-sim/AGENTS.md](../../../src/fleet-sim/AGENTS.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [docs/agent/tech-debt/td-016-fleet-sim-shared-ruff-contract-gap.md](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [docs/agent/tech-debt/td-017-fleet-sim-structure-gate-migration-gap.md](td-017-fleet-sim-structure-gate-migration-gap.md)

## Why It Matters

- Optimizer changes currently have a broad blast radius across analytical math,
  DES verification, reporting helpers, and package exports.
- The public optimizer surface is too easy to widen because the optimizer
  package and root `fleet_sim` package both expose optimizer symbols.
- Contributors need a concrete extraction target, not just a general warning
  that `base.py` is large.

## Desired End State

- Analytical sizing, DES verification/calibration, power or flexibility
  analysis, and reporting helpers live in narrower sibling modules.
- `fleet_sim/optimizer/__init__.py` owns optimizer exports, while root
  `fleet_sim/__init__.py` exposes only the deliberate top-level subset.
- Optimizer-facing tests validate behavior through narrow seams instead of one
  all-purpose module.

## Exit Criteria

- New optimizer features no longer require editing `base.py` unless they belong
  to the same analytical kernel.
- Public export changes are made in one clear owner layer instead of mirrored
  symbol lists.
- Fleet-sim local rules and implementation shape agree on the extraction-first
  path for optimizer work.
