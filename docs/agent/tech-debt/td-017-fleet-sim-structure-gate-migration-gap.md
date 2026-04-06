# TD017: Fleet Sim Migration Still Depends on Relaxed Structure Gates

## Status

Closed

## Scope

- `src/fleet-sim/fleet_sim/optimizer/base.py`
- `src/fleet-sim/fleet_sim/optimizer/{pareto.py,reporting.py}`
- `src/fleet-sim/run_sim.py`
- `src/fleet-sim/fleet_sim/cli_{common,core,advanced,parser}.py`
- `src/fleet-sim/fleet_sim/api/routes/traces.py`
- `src/fleet-sim/tests/{api_test_support.py,conftest.py,test_api.py,test_api_routes.py,test_api_runner_contract.py,test_api_job_lifecycle.py}`
- `tools/agent/structure-rules.yaml`

## Summary

Moving the fleet simulator from `bench/fleet-simulator` into the maintained `src/fleet-sim` subtree had left four migrated hotspots behind relaxed structure gates. That gap is now resolved. `run_sim.py` is a thin entrypoint over dedicated CLI support modules, the aggregated optimizer keeps Pareto and reporting helpers on sibling seams, trace-route response shaping is handled through small helpers, and the old monolithic API regression suite is split across focused test modules with shared support fixtures. The fleet-sim-specific legacy hotspot relaxations have been removed from `tools/agent/structure-rules.yaml`, so the shared structure contract now applies directly.

## Evidence

- [src/fleet-sim/fleet_sim/optimizer/base.py](../../../src/fleet-sim/fleet_sim/optimizer/base.py)
  - aggregated optimizer hotspot now stays under the shared thresholds and keeps only analytical sizing plus DES orchestration
- [src/fleet-sim/fleet_sim/optimizer/pareto.py](../../../src/fleet-sim/fleet_sim/optimizer/pareto.py)
  - threshold-frontier dataclass and rendering moved behind a sibling seam
- [src/fleet-sim/fleet_sim/optimizer/reporting.py](../../../src/fleet-sim/fleet_sim/optimizer/reporting.py)
  - optimization-report printing moved out of the analytical hotspot
- [src/fleet-sim/run_sim.py](../../../src/fleet-sim/run_sim.py)
  - CLI entrypoint is now a thin wrapper around dedicated parser and command modules
- [src/fleet-sim/fleet_sim/cli_common.py](../../../src/fleet-sim/fleet_sim/cli_common.py)
- [src/fleet-sim/fleet_sim/cli_core.py](../../../src/fleet-sim/fleet_sim/cli_core.py)
- [src/fleet-sim/fleet_sim/cli_advanced.py](../../../src/fleet-sim/fleet_sim/cli_advanced.py)
- [src/fleet-sim/fleet_sim/cli_parser.py](../../../src/fleet-sim/fleet_sim/cli_parser.py)
- [src/fleet-sim/fleet_sim/api/routes/traces.py](../../../src/fleet-sim/fleet_sim/api/routes/traces.py)
  - trace upload and lookup helpers now keep route handlers under the shared nesting target
- [src/fleet-sim/tests/test_api.py](../../../src/fleet-sim/tests/test_api.py)
  - the old API suite is now just the compatibility anchor for smaller sibling test modules
- [src/fleet-sim/tests/test_api_routes.py](../../../src/fleet-sim/tests/test_api_routes.py)
- [src/fleet-sim/tests/test_api_runner_contract.py](../../../src/fleet-sim/tests/test_api_runner_contract.py)
- [src/fleet-sim/tests/test_api_job_lifecycle.py](../../../src/fleet-sim/tests/test_api_job_lifecycle.py)
- [src/fleet-sim/tests/api_test_support.py](../../../src/fleet-sim/tests/api_test_support.py)
- [src/fleet-sim/tests/conftest.py](../../../src/fleet-sim/tests/conftest.py)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
  - the fleet-sim-specific legacy hotspot entries for `base.py`, `run_sim.py`, `traces.py`, and `test_api.py` have been removed

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

## Validation

- `python3 /Users/bitliu/vs/tools/agent/scripts/structure_check.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/optimizer/base.py /Users/bitliu/vs/src/fleet-sim/run_sim.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/api/routes/traces.py /Users/bitliu/vs/src/fleet-sim/tests/test_api.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/cli_common.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/cli_core.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/cli_advanced.py /Users/bitliu/vs/src/fleet-sim/fleet_sim/cli_parser.py /Users/bitliu/vs/src/fleet-sim/tests/test_api_routes.py /Users/bitliu/vs/src/fleet-sim/tests/test_api_runner_contract.py /Users/bitliu/vs/src/fleet-sim/tests/test_api_job_lifecycle.py`
- `cd /Users/bitliu/vs/src/fleet-sim && pytest tests/test_api.py tests/test_api_routes.py tests/test_api_runner_contract.py tests/test_api_job_lifecycle.py tests/test_optimizer.py tests/test_imports_and_profiles.py`
