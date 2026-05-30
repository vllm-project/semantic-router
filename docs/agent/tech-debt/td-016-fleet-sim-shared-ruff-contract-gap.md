# TD016: Fleet Sim Subtree Still Diverges from the Shared Ruff Contract

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

`src/fleet-sim/**`, the shared Python Ruff config, and agent diff-scoped Python
lint routing.

## Summary

`src/fleet-sim` is now a maintained subtree under `src/`, so the shared harness
detects it during changed-file validation. The subtree still does not satisfy
the repository-wide Ruff policy. Until that is fixed, fleet-sim changes can
report a large pre-existing style backlog instead of the regressions introduced
by the current branch.

## Evidence

- [src/fleet-sim/Makefile](../../../src/fleet-sim/Makefile)
- [src/fleet-sim/pyproject.toml](../../../src/fleet-sim/pyproject.toml)
- [tools/linter/python/.ruff.toml](../../../tools/linter/python/.ruff.toml)
- [tools/agent/scripts/agent_support.py](../../../tools/agent/scripts/agent_support.py)

## Why It Matters

- Branch-level validation loses signal when integration fixes are blocked by a
  style backlog outside the change.
- Reviewers need to tell new regressions apart from existing fleet-sim cleanup
  work.
- The subtree should have one lint contract that matches both local commands and
  shared harness gates.

## Desired End State

- `src/fleet-sim` satisfies the shared Ruff policy, or the repo defines a narrow
  explicit shared policy for that subtree.
- Fleet-sim validation reports PR-local regressions rather than the whole
  subtree backlog.
- Maintainer docs and local commands agree on the enforced lint contract.

## Exit Criteria

- `python3 -m ruff check --config tools/linter/python/.ruff.toml src/fleet-sim`
  passes without broad subtree relief.
- `make agent-ci-gate` on fleet-sim changed files does not need a special style
  bypass.
- Fleet-sim local and shared lint commands enforce the same policy.
