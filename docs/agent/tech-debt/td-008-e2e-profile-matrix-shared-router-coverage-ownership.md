# TD008: E2E Profile Matrix Repeats Shared Router Testcases Without Clear Coverage Ownership

## Status

Open

## Scope

test matrix efficiency and coverage design

## Summary

Many heavy profiles rerun the same shared router assertions, but the repo does not yet make baseline coverage ownership explicit.

## Evidence

- [e2e/README.md](../../../e2e/README.md)
- [e2e/profiles/ai-gateway/profile.go](../../../e2e/profiles/ai-gateway/profile.go)
- [e2e/profiles/dynamic-config/profile.go](../../../e2e/profiles/dynamic-config/profile.go)
- [e2e/profiles/istio/profile.go](../../../e2e/profiles/istio/profile.go)
- [e2e/profiles/llm-d/profile.go](../../../e2e/profiles/llm-d/profile.go)
- [e2e/profiles/production-stack/profile.go](../../../e2e/profiles/production-stack/profile.go)
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

- Each profile's `GetTestCases()` list is explainable in terms of unique contract coverage rather than inherited habit.
- Repeated shared testcases are reduced to a deliberate smoke subset or a single parameterized multi-profile matrix with clear purpose.
- Environment-specific profiles add assertions for their own semantics instead of mostly replaying generic router cases.
