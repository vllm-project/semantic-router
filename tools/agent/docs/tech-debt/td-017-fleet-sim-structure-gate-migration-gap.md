# TD017: Fleet Sim Still Depends on Relaxed Structure Gates

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

Fleet-sim optimizer and CLI functions that still exceed the shared AST
structure targets.

## Summary

The fleet-sim optimizer and CLI still need structure-gate relaxation because
some functions exceed the repo-wide function-size or nesting targets. The
active debt is to separate responsibilities until fleet-sim can use the same
AST structure gate as the rest of `src/` without special treatment. File length
is advisory and does not require a split by itself.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../../src/fleet-sim/fleet_sim/optimizer/base.py)
- [src/fleet-sim/run_sim.py](../../../../src/fleet-sim/run_sim.py)
- [tools/agent/structure-rules.yaml](../../../../tools/agent/structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../../../tools/agent/scripts/structure_check.py)

## Why It Matters

- Simulator packaging, standalone API, or service fixes should not have
  to carry unrelated extraction churn to pass changed-file validation.
- Maintainers need the exception list to point at concrete extraction targets.
- New fleet-sim modules should inherit the standard structure policy.

## Desired End State

- Fleet-sim orchestration, trace upload, optimizer, and test support are split
  into smaller modules with one main responsibility each.
- Structure validation for fleet-sim changes fails only on new architectural
  regressions.
- The shared structure gate no longer needs fleet-sim-specific function
  relaxations.

## Exit Criteria

- `make agent-lint` on fleet-sim changed files passes without fleet-sim
  structure exceptions.
- The scoped fleet-sim functions meet shared function and nesting thresholds.
- Any remaining fleet-sim-specific structure policy is documented as an active
  design choice, not a migration carve-out.
