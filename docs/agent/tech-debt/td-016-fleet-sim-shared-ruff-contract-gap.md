# TD016: Fleet Sim Subtree Still Diverges from the Shared Ruff Contract

## Status

Closed

## Scope

- `src/fleet-sim/**`
- `tools/linter/python/.ruff.toml`
- `tools/agent/scripts/agent_support.py`

## Summary

The fleet simulator was moved from `bench/fleet-simulator` into the maintained `src/fleet-sim` subtree, but the repository initially handled the resulting Ruff mismatch by parking a broad fleet-sim exception block inside the shared Python config. That kept `make agent-ci-gate` from re-reporting the subtree's historical backlog, but it also left the actual fleet-sim lint contract implicit and split across temporary shared-config relief, a non-blocking local `make lint`, and a separate `pyproject.toml` Ruff stanza.

That gap is now resolved. Fleet-sim owns an explicit subtree-local Ruff contract in `src/fleet-sim/.ruff.toml`, the shared harness discovers and uses that local config for changed fleet-sim Python files, the global shared Ruff config no longer carries a fleet-sim-wide exception block, and the simulator-local `make lint` target now points at the same blocking Ruff contract that the harness enforces.

## Evidence

- [src/fleet-sim/.ruff.toml](../../../src/fleet-sim/.ruff.toml)
  - the subtree now owns its explicit Ruff contract by extending the shared repo policy with fleet-sim-specific ignores
- [src/fleet-sim/Makefile](../../../src/fleet-sim/Makefile)
  - simulator-local lint now uses the same explicit Ruff config and no longer treats Ruff failures as report-only
- [src/fleet-sim/pyproject.toml](../../../src/fleet-sim/pyproject.toml)
  - the redundant package-local Ruff metadata has been removed so the subtree has one lint source of truth
- [src/fleet-sim/AGENTS.md](../../../src/fleet-sim/AGENTS.md)
  - local rules now tell contributors to keep the fleet-sim lint contract in `src/fleet-sim/.ruff.toml` instead of reopening shared-config carve-outs
- [tools/linter/python/.ruff.toml](../../../tools/linter/python/.ruff.toml)
  - the shared Ruff policy no longer contains a fleet-sim-wide temporary exception block
- [tools/agent/scripts/agent_support.py](../../../tools/agent/scripts/agent_support.py)
  - diff-scoped harness lint now resolves the nearest subtree `.ruff.toml` before invoking Ruff
- `python /Users/bitliu/vs/tools/agent/scripts/agent_gate.py run-python-lint --changed-files-path /tmp/vsr_td016_changed.txt`
  - shared changed-file lint now runs fleet-sim files against `src/fleet-sim/.ruff.toml` instead of the global shared config
- `cd /Users/bitliu/vs/src/fleet-sim && python -m ruff check --config .ruff.toml run_sim.py fleet_sim tests`
  - the explicit fleet-sim-owned Ruff contract now passes on the subtree without relying on a global fleet-sim carve-out

## Why It Matters

- branch-level validation loses signal when simulator integration fixes are blocked by a large inherited style backlog that predates the branch
- reviewers cannot tell whether a `fleet-sim` PR introduced a regression or merely intersected the subtree's unretired Ruff debt
- leaving fleet-sim relief inside the shared repo config makes the repo-wide policy look broader than it really is and hides who owns the simulator-specific lint contract

## Desired End State

- `src/fleet-sim` converges on the repository-wide Ruff contract, or the repo defines a narrower explicit shared policy for that subtree instead of broad temporary exceptions
- harness validation for `fleet-sim` changes reports only PR-local regressions, not the entire inherited simulator style backlog
- the fleet-sim subtree exposes a clear maintainer-owned lint contract that matches what branch gates enforce

## Exit Criteria

- the shared Ruff config no longer contains a fleet-sim-wide temporary exception block
- changed-file Python lint resolves fleet-sim files through the explicit `src/fleet-sim/.ruff.toml` contract instead of forcing them through the repo-global Ruff config
- the repo-native fleet-sim lint contract is documented and enforced consistently by both subtree-local commands and the shared harness

## Retirement Notes

- `src/fleet-sim/.ruff.toml` now extends the shared repo Ruff policy and owns the subtree-specific ignore set explicitly.
- `tools/agent/scripts/agent_support.py` now groups changed Python files by the nearest subtree `.ruff.toml`, so diff-scoped harness lint uses the fleet-sim contract for simulator files and the shared contract elsewhere.
- `tools/linter/python/.ruff.toml` no longer needs a broad fleet-sim-specific carve-out, and `src/fleet-sim/pyproject.toml` no longer carries a second Ruff source of truth.
- `src/fleet-sim/Makefile` and `src/fleet-sim/AGENTS.md` now point contributors at the same blocking Ruff contract that the shared harness enforces.

## Validation

- `/Users/bitliu/vs/.venv-agent/bin/python -m ruff check --config /Users/bitliu/vs/tools/linter/python/.ruff.toml /Users/bitliu/vs/tools/agent/scripts/agent_support.py`
- `cd /Users/bitliu/vs/src/fleet-sim && /Users/bitliu/vs/.venv-agent/bin/python -m ruff check --config .ruff.toml run_sim.py fleet_sim tests`
- `cd /Users/bitliu/vs/src/fleet-sim && pytest tests/test_imports_and_profiles.py`
- `cd /Users/bitliu/vs/src/fleet-sim && /Users/bitliu/vs/.venv-agent/bin/python -m black --check --diff run_sim.py fleet_sim/__init__.py tests/test_imports_and_profiles.py tests/test_disagg.py tests/test_hardware.py tests/test_hf_import.py tests/test_models.py tests/test_profiles.py`
- `/Users/bitliu/vs/.venv-agent/bin/python /Users/bitliu/vs/tools/agent/scripts/agent_gate.py run-python-lint --changed-files-path /tmp/vsr_td016_changed.txt`
