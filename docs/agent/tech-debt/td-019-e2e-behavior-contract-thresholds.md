# TD019: Behavior Contract E2E Tests Still Encode Report-Only Thresholds

## Status

Closed

## Scope

`e2e/testcases/**`, shared testcase helpers, and the `ai-gateway` baseline router contract

## Summary

Several user-visible router behavior tests still behave like dashboards or benchmark probes rather than acceptance tests. They report accuracy or success-rate metrics, but only fail on `0%` accuracy, on "no successful requests", or on transport-level setup errors. That means obvious regressions in classification, routing fallback, plugin execution, or stress behavior can still pass the canonical `ai-gateway` contract. The repository already treats these testcases as baseline router coverage, so the current threshold model understates real regressions and weakens feature-gate signal.

## Evidence

- [e2e/pkg/testmatrix/testcases.go](../../../e2e/pkg/testmatrix/testcases.go)
- [e2e/testcases/domain_classify.go](../../../e2e/testcases/domain_classify.go)
- [e2e/testcases/plugin_config_variations.go](../../../e2e/testcases/plugin_config_variations.go)
- [e2e/testcases/plugin_chain_execution.go](../../../e2e/testcases/plugin_chain_execution.go)
- [e2e/testcases/decision_fallback.go](../../../e2e/testcases/decision_fallback.go)
- [e2e/testcases/chat_completions_progressive_stress.go](../../../e2e/testcases/chat_completions_progressive_stress.go)
- [e2e/testcases/mcp_http_classification.go](../../../e2e/testcases/mcp_http_classification.go)
- [e2e/testcases/mcp_probability_distribution.go](../../../e2e/testcases/mcp_probability_distribution.go)

## Why It Matters

- Feature-gate runs can pass even when the router is visibly misclassifying requests or degrading under stress.
- Acceptance and observability concerns are mixed together, so it is unclear which tests define product behavior and which are merely collecting telemetry.
- Contributors receive a false sense of safety from a green baseline contract that does not encode the intended behavioral floor.
- Weak thresholds make later refactors harder because regressions are discovered by manual log reading instead of executable contracts.

## Desired End State

- Every behavior-sensitive testcase in the baseline contract has an explicit, meaningful failure threshold.
- Benchmark or reporting-oriented cases are named and tracked as such instead of masquerading as acceptance coverage.
- Shared testcase helpers separate metric collection from pass/fail assertions so the acceptance policy is easy to audit and reuse.
- Harness docs and checklists explicitly reject `0%`-only thresholds for routing, classification, plugin, or API behavior contracts.

## Exit Criteria

- Satisfied on 2026-03-18: the `ai-gateway` contract cases for routing, classification, cache, plugin execution, fallback behavior, and stress validation now fail on explicit floors instead of only at `0%`.
- Satisfied on 2026-03-18: stress and progressive-load cases now declare executable success-rate budgets instead of acting as report-only probes.
- Satisfied on 2026-03-18: shared testcase helper logic now centralizes threshold evaluation so new baseline cases can reuse the same acceptance contract.
- Satisfied on 2026-03-18: harness docs and local rules state that report-only tests do not satisfy the repository's E2E acceptance bar.

## Resolution

- [e2e/pkg/testcases/acceptance_contracts.go](../../../e2e/pkg/testcases/acceptance_contracts.go) now defines the baseline `ai-gateway` floors and the shared acceptance evaluators for contract-style router cases.
- [e2e/pkg/testcases/registry.go](../../../e2e/pkg/testcases/registry.go) now wraps the named baseline contract cases at registration time so testcase files can keep publishing metrics while the shared seam owns threshold enforcement.
- Baseline contract cases continue to publish structured `SetDetails` maps, and the shared acceptance seam annotates those details with the minimum accepted rate before failing runs that fall below the conservative floor.
- Progressive stress reporting now gets per-stage and overall minimum success rates injected by the shared acceptance seam, with failures surfaced as executable test errors instead of console-only summaries.
- Validation note: `go test ./pkg/testcases/...`, `make agent-validate`, and `make agent-ci-gate ...` passed after the shared seam landed. A cold-start `make e2e-test E2E_PROFILE=ai-gateway E2E_VERBOSE=true` run timed out during Helm install because the profile had to anonymously download required Hugging Face models without `HF_TOKEN`.
