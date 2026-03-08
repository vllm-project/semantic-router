# TD007: End-to-End and Integration Test Surfaces Are Split Across Parallel Frameworks

## Status

Open

## Scope

test architecture and harness coverage

## Summary

The repository currently maintains multiple durable integration and E2E entrypoints that sit outside one coherent harness-wide selection story.

## Evidence

- [docs/agent/testing-strategy.md](../testing-strategy.md)
- [docs/agent/playbooks/e2e-selection.md](../playbooks/e2e-selection.md)
- [tools/agent/task-matrix.yaml](../../../tools/agent/task-matrix.yaml)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [e2e/cmd/e2e/main.go](../../../e2e/cmd/e2e/main.go)
- [e2e/testing/run_all_tests.py](../../../e2e/testing/run_all_tests.py)
- [.github/workflows/integration-test-memory.yml](../../../.github/workflows/integration-test-memory.yml)
- [.github/workflows/integration-test-vllm-sr-cli.yml](../../../.github/workflows/integration-test-vllm-sr-cli.yml)

## Why It Matters

- The repository currently maintains multiple durable integration and E2E entrypoints: the Go profile-based harness under `e2e/`, standalone Python suites under `e2e/testing/`, and workflow-driven integration coverage outside the harness selection path.
- The testing docs tell contributors to use `task-matrix.yaml` and `e2e-profile-map.yaml` as the selection contract, but those files do not model the full set of workflow-driven integration suites.
- This makes it unclear which suite is canonical for a given behavior change, weakens affected-test selection, and allows some integration or E2E changes to sit outside the normal harness validation story.

## Desired End State

- One explicit repository-wide taxonomy for unit, integration, local E2E, and CI-only system tests.
- Every durable integration or E2E suite is either represented in the harness selection rules or intentionally declared as an exception with documented ownership and invocation.

## Exit Criteria

- A contributor can change any integration or E2E file and get a deterministic `agent-report` result with named validation commands and affected suites.
- Workflow-only suites are either absorbed into the canonical harness or documented as bounded exceptions instead of parallel hidden paths.
