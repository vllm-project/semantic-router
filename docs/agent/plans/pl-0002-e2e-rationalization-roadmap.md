# E2E Rationalization Execution Plan

This document tracks the long-horizon loop for retiring the repository's open E2E and integration-test debt items.

## Goal

- Converge the repository on one explicit, harness-aligned E2E and integration testing model that contributors can resume from repo state alone.
- Preserve the closed `TD007` and `TD008` gains while retiring the remaining `TD009` naming debt and the new extensibility debt tracked in `TD010`.

## Scope

- Canonical docs and harness manifests that classify E2E and integration suites
- The Go profile-based `e2e/` runner, legacy `e2e/testing/` Python suites, and workflow-driven integration coverage
- Coverage ownership, profile inventory, and validation/reporting rules for affected-suite selection
- Framework-layer extension seams for profile registration, runner capabilities, service connections, shared stack composition, and reusable contract fixtures

## Exit Criteria

- Any durable integration or E2E file change yields a deterministic `make agent-report` or `make agent-e2e-affected` result with named suites or documented exceptions.
- `e2e/README.md`, `e2e/profiles/all/imports.go`, and `tools/agent/e2e-profile-map.yaml` share one mechanically checked runnable profile inventory.
- Heavy profiles and CLI/integration suites have explicit coverage ownership, with shared smoke behavior separated from environment-specific assertions.
- Runner-level cluster/image requirements and testcase service connections are declared through profile metadata and `ServiceConfig`, not profile-name switches or fixed local ports.
- `TD009` and `TD010` can be retired or materially narrowed in the same change that closes the underlying gaps.

## Task List

- [x] `E001` Inventory every durable integration and E2E entrypoint across the Go harness, Python suites, and CI workflows.
  - Done when the repo has one canonical matrix for unit, integration, local E2E, and CI-only system tests, including owner and invocation.
- [x] `E002` Define the canonical profile inventory and naming contract.
  - Done when `e2e/README.md`, `e2e/profiles/all/imports.go`, and `tools/agent/e2e-profile-map.yaml` are synchronized and naming drift is either fixed or explicitly mapped.
- [x] `E003` Pull workflow-driven integration suites into the harness selection story.
  - Done when each durable workflow-only suite is either modeled in manifests/reporting or documented as a bounded exception with ownership.
- [x] `E004` Publish a coverage ownership matrix for shared smoke, cross-profile contracts, and environment-specific assertions.
  - Done when each profile's `GetTestCases()` list and each CLI/integration suite can be justified by a named contract instead of inherited repetition.
- [x] `E005` Reduce repeated shared router testcases in heavy profiles and expensive environments.
  - Done when common baseline behavior runs through a deliberate smoke path or parameterized matrix and expensive profiles focus on unique semantics.
- [x] `E006` Refactor legacy CLI and integration suites to separate common startup/setup cost from distinct assertions.
  - Done when the Python integration path no longer repeats the full `vllm-sr serve` bootstrap for each small contract check.
- [x] `E007` Add mechanical drift checks and close-out guidance for E2E inventory and coverage ownership.
  - Done when validation/reporting can catch profile inventory drift and plan/debt/docs stay aligned as the workstream progresses.
- [x] `E008` Retire or materially narrow `TD007`, `TD008`, and `TD009`.
  - Done when the debt register reflects the reduced scope instead of carrying the same umbrella gaps.
- [x] `E009` Introduce registry-driven profile metadata for runner capabilities and runnable inventory.
  - Done when adding a profile no longer requires a constructor switch in `cmd/e2e/main.go`, and runner-level capabilities are declared next to the profile instead of in core runner name branches.
- [x] `E010` Decouple testcase service connections from fixed local port ownership.
  - Done when shared testcase helpers allocate ephemeral local ports by default and profiles describe the service-side port instead of hard-coding `8080:*` local mappings.
- [x] `E011` Extract composable stack modules for the repeated semantic-router/envoy/ai-gateway deployment flows.
  - Done when the main gateway-based profiles reuse shared stack helpers instead of carrying near-duplicate setup/teardown scripts.
- [x] `E012` Add higher-level typed fixtures and clients for common router and Responses API contracts.
  - Done when new contract suites can be added without open-coded port-forward, URL-building, and ad-hoc `http.Client` boilerplate in each testcase.

## Current Loop

- Loop status: completed on 2026-03-08.
- Completed in this loop: registry-driven profile metadata, `e2e/profiles/all/imports.go` as the runnable profile inventory, runner capability extraction, ephemeral local port allocation for shared testcase connections, `e2e/pkg/stacks/gateway` as the reusable gateway deployment module, `e2e/pkg/fixtures` as the typed session/client layer, and the `response-api-shared` rename that retires the last public/internal naming mismatch.
- Residual follow-up: none in the E2E framework rationalization workstream; future E2E changes should extend the stack and fixture layers instead of reintroducing copied setup scripts or ad-hoc HTTP wiring.

## Decision Log

- This plan was the long-horizon execution loop for `TD007`, `TD008`, `TD009`, and `TD010`; it remains in-repo as the resumable record of how the E2E harness converged.
- The final loop retired the residual naming and extensibility debt in the same change that introduced shared stack modules and typed fixtures, so the workstream no longer depends on an open follow-up debt item.
- The first loop prioritizes taxonomy, inventory, and harness integration because affected-suite selection has to become deterministic before coverage can be safely reduced.
- Workflow-only suites may remain temporarily only if the repo documents them as bounded exceptions with clear ownership and invocation, rather than leaving them as hidden parallel paths.
- `workflow_suite_rules`, `task-matrix.yaml`, and `agent_resolution.py` now carry workflow-driven integration suites through the same selection path as Go E2E profiles.
- `agent_ci_validation.py` now checks runnable profile inventory drift between `e2e/README.md`, `e2e/profiles/all/imports.go`, and `tools/agent/e2e-profile-map.yaml`.
- `ai-gateway` now owns the full baseline router contract, while heavy profiles reuse `testmatrix.RouterSmoke` plus profile-specific checks and the CLI integration suite reuses one shared serve lifecycle per contract bundle.
- `e2e/pkg/framework/profile_registry.go` and `e2e/profiles/all/imports.go` now own runner capabilities like GPU enablement and extra local images, removing those concerns from `runner.go` name switches.
- Shared testcase service connections now treat fixed local ports as framework-owned ephemeral resources, while profiles declare only the service-side port unless they explicitly need a stable local port.
- `e2e/pkg/stacks/gateway` is now the default extension seam for semantic-router/envoy/ai-gateway profiles, and the baseline `ai-gateway`, `routing-strategies`, `authz-rbac`, `multi-endpoint`, `streaming`, and `response-api*` profiles all compose through that module instead of copying the same lifecycle logic.
- `e2e/pkg/fixtures` now owns reusable service sessions plus typed chat/responses clients, and the baseline chat testcase plus the full Responses API contract suite use that layer instead of open-coded port-forward and raw `http.Client` setup.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-007-e2e-integration-surfaces-split-across-frameworks.md](../tech-debt/td-007-e2e-integration-surfaces-split-across-frameworks.md)
- [../tech-debt/td-008-e2e-profile-matrix-shared-router-coverage-ownership.md](../tech-debt/td-008-e2e-profile-matrix-shared-router-coverage-ownership.md)
- [../tech-debt/td-009-e2e-profile-inventory-and-documentation-drift.md](../tech-debt/td-009-e2e-profile-inventory-and-documentation-drift.md)
- [../tech-debt/td-010-e2e-framework-extension-paths.md](../tech-debt/td-010-e2e-framework-extension-paths.md)
- [../adr/README.md](../adr/README.md) (no dedicated ADR yet)
