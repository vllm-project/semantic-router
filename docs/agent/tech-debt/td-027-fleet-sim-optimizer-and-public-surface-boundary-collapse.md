# TD027: Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy

## Status

Closed

## Scope

`src/fleet-sim/fleet_sim/optimizer/**`, `src/fleet-sim/fleet_sim/__init__.py`, and adjacent optimizer-facing tests or API callers

## Summary

The optimizer boundary is now split across clear sibling seams instead of one all-purpose hotspot. `fleet_sim/optimizer/base.py` stays focused on analytical sizing, DES calibration, Pareto search, and `OptimizationReport`; grid-flex analysis and reporting now live in `fleet_sim/optimizer/grid_flex.py`; tokens-per-watt analysis and `_split_cdf` now live in `fleet_sim/optimizer/tpw.py`. `fleet_sim/optimizer/__init__.py` is now the owner of the root-facing optimizer export subset, and `fleet_sim/__init__.py` forwards only that curated list instead of manually mirroring a parallel symbol inventory. Optimizer-facing tests and fleet-sim docs now use those narrower seams directly, so this debt no longer needs to stay open.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
- [src/fleet-sim/fleet_sim/optimizer/grid_flex.py](../../../src/fleet-sim/fleet_sim/optimizer/grid_flex.py)
  - grid-flex analysis, throttled-profile simulation support, and reporting now live behind their own module seam
- [src/fleet-sim/fleet_sim/optimizer/tpw.py](../../../src/fleet-sim/fleet_sim/optimizer/tpw.py)
  - TPW dataclasses, `_split_cdf`, fleet-level aggregation, and reporting now live behind their own module seam
- [src/fleet-sim/fleet_sim/optimizer/__init__.py](../../../src/fleet-sim/fleet_sim/optimizer/__init__.py)
  - optimizer package exports remain the primary seam and own the curated root-facing subset
- [src/fleet-sim/fleet_sim/__init__.py](../../../src/fleet-sim/fleet_sim/__init__.py)
  - root package now forwards the optimizer-owned curated export list instead of hand-maintaining a second symbol inventory
- [src/fleet-sim/tests/test_imports_and_profiles.py](../../../src/fleet-sim/tests/test_imports_and_profiles.py)
  - import stability and the curated root-vs-optimizer seam now have focused regression coverage
- [website/docs/fleet-sim/use-cases.md](../../../website/docs/fleet-sim/use-cases.md)
- [website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/fleet-sim/use-cases.md](../../../website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/fleet-sim/use-cases.md)
  - advanced optimizer analyses now document `fleet_sim.optimizer` as the primary import seam
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

- Satisfied on 2026-04-06: new grid-flex and TPW work no longer requires widening `base.py`; those families now live in dedicated sibling modules while `base.py` stays on the analytical sizing and calibration path.
- Satisfied on 2026-04-06: root-facing optimizer exports are now owned by `fleet_sim/optimizer/__init__.py`, and `fleet_sim/__init__.py` forwards that curated subset instead of hand-maintaining a second parallel export list.
- Satisfied on 2026-04-06: local AGENT rules still name the optimizer hotspot and extraction-first path, and the new modules plus import regression tests make that narrower ownership executable.

## Retirement Notes

- `fleet_sim/optimizer/base.py` no longer owns grid-flex or TPW dataclasses, reporting helpers, or `_split_cdf`; it is back to the analytical sizing and DES calibration seam.
- `fleet_sim/optimizer/grid_flex.py` now owns the demand-response power/latency trade-off path, including DES verification helpers.
- `fleet_sim/optimizer/tpw.py` now owns the energy-efficiency path, including per-pool and fleet-level TPW aggregation plus the CDF-splitting helper used by routed-fleet analysis.
- `fleet_sim.optimizer` is now the documented primary seam for advanced optimizer analyses, while the root `fleet_sim` package only forwards the curated subset that the optimizer package owns.

## Validation

- `make vllm-sr-sim-test`
