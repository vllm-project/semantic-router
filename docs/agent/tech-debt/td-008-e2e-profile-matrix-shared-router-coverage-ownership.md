# TD008: E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership

## Status

Closed

## Scope

test matrix efficiency and coverage design

## Summary

Many heavy profiles previously reran the same shared router assertions without an explicit coverage ownership model. The E2E matrix now separates baseline router coverage, smoke subsets, and profile-specific contracts, so this gap is retired.

## Evidence

- [e2e/README.md](../../../e2e/README.md)
- [e2e/profiles/ai-gateway/profile.go](../../../e2e/profiles/ai-gateway/profile.go)
- [e2e/profiles/dynamic-config/profile.go](../../../e2e/profiles/dynamic-config/profile.go)
- [e2e/profiles/istio/profile.go](../../../e2e/profiles/istio/profile.go)
- [e2e/profiles/llm-d/profile.go](../../../e2e/profiles/llm-d/profile.go)
- [e2e/profiles/production-stack/profile.go](../../../e2e/profiles/production-stack/profile.go)
- [e2e/pkg/testmatrix/testcases.go](../../../e2e/pkg/testmatrix/testcases.go)
- [e2e/testcases/aibrix_control_plane_health.go](../../../e2e/testcases/aibrix_control_plane_health.go)
- [e2e/testcases/llmd_inference_gateway_health.go](../../../e2e/testcases/llmd_inference_gateway_health.go)
- [e2e/testing/vllm-sr-cli/cli_test_base.py](../../../e2e/testing/vllm-sr-cli/cli_test_base.py)
- [e2e/testing/vllm-sr-cli/test_integration.py](../../../e2e/testing/vllm-sr-cli/test_integration.py)

## Why It Matters

- Many heavy profiles rerun the same shared router assertions. Today `chat-completions-request` is wired into nine profiles, the two chat stress cases into six profiles, and `domain-classify` into six profiles.
- Some profiles explicitly reuse shared router cases rather than asserting profile-specific semantics. `llm-d`, for example, currently reuses shared router testcases and expects llm-d-specific HA or traffic behavior to be covered elsewhere.
- The CLI integration suite shows the same pattern at a smaller scale: several methods each repeat the full `vllm-sr serve` startup path instead of separating common setup cost from distinct contract checks.
- Without a durable coverage ownership map, expensive environments spend time re-verifying baseline behavior while unique environment contracts remain under-specified.

## Desired End State

- A coverage matrix that distinguishes baseline router smoke, cross-profile contract tests, and environment-specific assertions.
- Shared behaviors run in one small baseline path or a parameterized matrix, while expensive profiles focus on the semantics that only they can validate.

## Exit Criteria

- Satisfied on 2026-03-08: each profile's `GetTestCases()` list is explainable in terms of unique contract coverage rather than inherited habit.
- Satisfied on 2026-03-08: repeated shared testcases are reduced to deliberate smoke subsets and baseline contract groups with clear purpose.
- Satisfied on 2026-03-08: environment-specific profiles now add assertions for their own semantics instead of mostly replaying generic router cases.

## Resolution

- `e2e/README.md` now publishes a coverage ownership matrix that distinguishes the baseline router contract from heavy-profile-specific assertions.
- `e2e/pkg/testmatrix/testcases.go` defines `BaselineRouterContract` and `RouterSmoke`, and heavy profiles now compose their testcase lists from those shared groups instead of replaying the full baseline suite.
- `aibrix` and `llm-d` now own profile-specific health checks, while the Python CLI integration suite reuses a shared serve lifecycle and validates multiple contracts per startup.
