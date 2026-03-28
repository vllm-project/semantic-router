# TD017: Fleet Sim Migration Still Depends on Relaxed Structure Gates

## Status

Open

## Scope

- `src/fleet-sim/fleet_sim/optimizer/base.py`
- `src/fleet-sim/run_sim.py`
- `src/fleet-sim/fleet_sim/api/routes/traces.py`
- `src/fleet-sim/tests/test_api.py`
- `tools/agent/structure-rules.yaml`
- `tools/agent/scripts/structure_check.py`

## Summary

Moving the fleet simulator from `bench/fleet-simulator` into the maintained `src/fleet-sim` subtree brought several legacy monolith files under the shared structure gate for the first time. Those files already exceeded the repository-wide file-size, function-size, and nesting targets before the migration, so diff-scoped `make agent-lint` and `make agent-ci-gate` started failing on inherited simulator architecture debt instead of on branch-local regressions.

This change records that mismatch explicitly and adds narrow legacy-hotspot relaxations for the specific migrated files that still exceed the shared structure contract. The relaxations are limited to the known monoliths so new fleet-sim modules continue to inherit the default structure rules.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
  - migrated optimizer orchestrator still exceeds shared file/function thresholds materially
- [src/fleet-sim/run_sim.py](../../../src/fleet-sim/run_sim.py)
  - migrated CLI entrypoint still combines command parsing, workflow orchestration, and report rendering in one file
- [src/fleet-sim/fleet_sim/api/routes/traces.py](../../../src/fleet-sim/fleet_sim/api/routes/traces.py)
  - trace upload flow still exceeds the shared nesting limit
- [src/fleet-sim/tests/test_api.py](../../../src/fleet-sim/tests/test_api.py)
  - migrated API regression suite still exceeds the shared file-size limit
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
  - shared structure gate now needs explicit fleet-sim hotspot entries to avoid misreporting inherited monolith debt as new regressions
- `make agent-lint`
  - reported structure-check failures on the four migrated files immediately after the subtree moved under `src/`

## Why It Matters

- branch validation loses signal when simulator packaging, dashboard integration, or service fixes are blocked by pre-existing structural debt in migrated files
- reviewers cannot tell whether a fleet-sim PR worsened module shape or merely touched a file that was already above the global thresholds
- treating the shared structure contract as immediately authoritative for these migrated monoliths would force unrelated follow-up work to carry large extraction churn before behavior fixes can land

## Desired End State

- the migrated fleet-sim monoliths are decomposed so they satisfy the shared structure targets without file-scoped relaxations
- structure validation for `src/fleet-sim` changes fails only on new architectural regressions, not on the inherited migration backlog
- fleet-sim orchestration, API upload handling, and large regression suites are split into smaller modules with clearer responsibilities

## Exit Criteria

- `make agent-lint CHANGED_FILES="...,src/fleet-sim/...,..."` passes without the fleet-sim legacy-hotspot relaxations in `tools/agent/structure-rules.yaml`
- `src/fleet-sim/fleet_sim/optimizer/base.py`, `src/fleet-sim/run_sim.py`, `src/fleet-sim/fleet_sim/api/routes/traces.py`, and `src/fleet-sim/tests/test_api.py` meet the shared file/function/nesting thresholds directly
- the remaining fleet-sim structure contract is documented and enforced consistently by the shared harness without migration-specific carve-outs
