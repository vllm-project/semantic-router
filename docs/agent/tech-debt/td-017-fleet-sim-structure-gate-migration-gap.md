# TD017: Fleet Sim Still Depends on Relaxed Structure Gates

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

Fleet-sim optimizer, CLI, API trace upload, and API regression-test files that
still exceed the shared structure targets.

## Summary

Several fleet-sim files still need structure-gate relaxation because they exceed
the repo-wide file-size, function-size, or nesting targets. The active debt is
to split those files until fleet-sim can use the same structure gate as the rest
of `src/` without special treatment.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
- [src/fleet-sim/run_sim.py](../../../src/fleet-sim/run_sim.py)
- [src/fleet-sim/fleet_sim/api/routes/traces.py](../../../src/fleet-sim/fleet_sim/api/routes/traces.py)
- [src/fleet-sim/tests/test_api.py](../../../src/fleet-sim/tests/test_api.py)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../../tools/agent/scripts/structure_check.py)

## Why It Matters

- Simulator packaging, dashboard integration, or service fixes should not have
  to carry unrelated extraction churn to pass changed-file validation.
- Maintainers need the exception list to point at concrete extraction targets.
- New fleet-sim modules should inherit the standard structure policy.

## Desired End State

- Fleet-sim orchestration, trace upload, optimizer, and test support are split
  into smaller modules with one main responsibility each.
- Structure validation for fleet-sim changes fails only on new architectural
  regressions.
- The shared structure gate no longer needs fleet-sim-specific file relaxations.

## Exit Criteria

- `make agent-lint` on fleet-sim changed files passes without fleet-sim
  structure exceptions.
- The scoped fleet-sim files meet shared file/function/nesting thresholds.
- Any remaining fleet-sim-specific structure policy is documented as an active
  design choice, not a migration carve-out.
