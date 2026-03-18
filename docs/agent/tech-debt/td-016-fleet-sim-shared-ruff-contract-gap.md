# TD016: Fleet Sim Subtree Still Diverges from the Shared Ruff Contract

## Status

Open

## Scope

- `src/fleet-sim/**`
- `tools/linter/python/.ruff.toml`
- `tools/agent/scripts/agent_support.py`

## Summary

The fleet simulator was moved from `bench/fleet-simulator` into the maintained `src/fleet-sim` subtree, but the codebase still carries the legacy style and naming conventions from its original standalone package. The shared harness now lint-detects that subtree because it lives under `src/`, yet the repository-wide Ruff contract is materially stricter than the simulator's historical local workflow. Diff-scoped `make agent-ci-gate` therefore re-enters hundreds of pre-existing style violations whenever any `src/fleet-sim/**/*.py` file changes, even when the branch only fixes packaging, service, or dashboard integration behavior.

This change records that mismatch explicitly and scopes the shared Ruff gate away from the known fleet-sim legacy rule backlog so branch validation can distinguish new regressions from inherited simulator style debt.

## Evidence

- [src/fleet-sim/Makefile](../../../src/fleet-sim/Makefile)
  - simulator-local lint is lightweight and non-blocking today
- [src/fleet-sim/pyproject.toml](../../../src/fleet-sim/pyproject.toml)
  - the subtree still carries its own package-local formatting metadata
- [tools/linter/python/.ruff.toml](../../../tools/linter/python/.ruff.toml)
  - shared Ruff policy is stricter than the fleet-sim subtree currently satisfies
- [tools/agent/scripts/agent_support.py](../../../tools/agent/scripts/agent_support.py)
  - diff-scoped harness lint runs shared Ruff directly on changed Python files
- `python3 -m ruff check --config tools/linter/python/.ruff.toml --statistics src/fleet-sim`
  - reported 551 remaining violations across the migrated subtree after mechanical autofixes, dominated by `PLC0415`, `PLR2004`, `RUF00x`, and simulator-specific naming rules

## Why It Matters

- branch-level validation loses signal when simulator integration fixes are blocked by a large inherited style backlog that predates the branch
- reviewers cannot tell whether a `fleet-sim` PR introduced a regression or merely intersected the subtree's unretired Ruff debt
- treating the shared Ruff contract as immediately authoritative for `src/fleet-sim` would create a second conflicting source of truth until the simulator code is intentionally converged

## Desired End State

- `src/fleet-sim` converges on the repository-wide Ruff contract, or the repo defines a narrower explicit shared policy for that subtree instead of broad temporary exceptions
- harness validation for `fleet-sim` changes reports only PR-local regressions, not the entire inherited simulator style backlog
- the fleet-sim subtree exposes a clear maintainer-owned lint contract that matches what branch gates enforce

## Exit Criteria

- `python3 -m ruff check --config tools/linter/python/.ruff.toml src/fleet-sim` passes without the temporary subtree exceptions
- `make agent-ci-gate CHANGED_FILES="...,src/fleet-sim/...,..."` no longer needs broad shared-Ruff relief to validate simulator changes honestly
- the repo-native fleet-sim lint contract is documented and enforced consistently by both subtree-local commands and the shared harness
