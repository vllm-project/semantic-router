# TD010: E2E Framework Extension Paths Still Rely on Script-Style Stack Composition and Low-Level Test Fixtures

## Status

Closed

## Scope

framework architecture and contributor extensibility

## Summary

The E2E framework previously relied on runner name switches, script-style repeated stack composition, and low-level testcase plumbing that made new profiles and contracts expensive to add. The framework now exposes registry-driven metadata, shared gateway stack modules, and typed fixtures, so this gap is retired.

## Evidence

- [e2e/pkg/framework/profile_registry.go](../../../e2e/pkg/framework/profile_registry.go)
- [e2e/pkg/framework/runner.go](../../../e2e/pkg/framework/runner.go)
- [e2e/pkg/framework/types.go](../../../e2e/pkg/framework/types.go)
- [e2e/pkg/stacks/gateway/stack.go](../../../e2e/pkg/stacks/gateway/stack.go)
- [e2e/pkg/fixtures/session.go](../../../e2e/pkg/fixtures/session.go)
- [e2e/pkg/fixtures/chat.go](../../../e2e/pkg/fixtures/chat.go)
- [e2e/pkg/fixtures/response_api.go](../../../e2e/pkg/fixtures/response_api.go)
- [e2e/testcases/common.go](../../../e2e/testcases/common.go)
- [e2e/testcases/chat_completions_request.go](../../../e2e/testcases/chat_completions_request.go)
- [e2e/testcases/response_api_basic.go](../../../e2e/testcases/response_api_basic.go)
- [e2e/testcases/response_api_error_handling.go](../../../e2e/testcases/response_api_error_handling.go)
- [e2e/testcases/response_api_conversation_chaining.go](../../../e2e/testcases/response_api_conversation_chaining.go)
- [e2e/testcases/response_api_edge_cases.go](../../../e2e/testcases/response_api_edge_cases.go)
- [e2e/profiles/ai-gateway/profile.go](../../../e2e/profiles/ai-gateway/profile.go)
- [e2e/profiles/authz-rbac/profile.go](../../../e2e/profiles/authz-rbac/profile.go)
- [e2e/profiles/multi-endpoint/profile.go](../../../e2e/profiles/multi-endpoint/profile.go)
- [e2e/profiles/response-api/profile.go](../../../e2e/profiles/response-api/profile.go)
- [e2e/profiles/routing-strategies/profile.go](../../../e2e/profiles/routing-strategies/profile.go)
- [e2e/profiles/streaming/profile.go](../../../e2e/profiles/streaming/profile.go)
- [e2e/README.md](../../../e2e/README.md)

## Why It Matters

- New profile onboarding previously required edits in multiple runner switch points and repeated setup/teardown scripts for the same semantic-router/envoy/ai-gateway topology.
- Common contract tests previously open-coded port-forward sessions, URL assembly, and ad-hoc `http.Client` wiring, which made testcase authorship harder than necessary.
- This kept the framework usable but not elegant, and raised the cost of extending the E2E stack as new features landed.

## Desired End State

- Adding a new profile uses registry metadata and shared stack modules instead of editing runner name switches.
- Gateway-based profiles compose through reusable stack modules instead of copying deployment scripts.
- Common router and Responses API contracts extend through typed fixtures and clients instead of low-level per-test plumbing.

## Exit Criteria

- Satisfied on 2026-03-08: runner behavior is metadata-driven, not dependent on profile-name switches in `cmd/e2e/main.go` or `runner.go`.
- Satisfied on 2026-03-08: the main gateway-based profiles share one stack module instead of duplicated setup and teardown flows.
- Satisfied on 2026-03-08: common chat and Responses API contracts extend through typed fixtures instead of ad-hoc port-forward and raw `http.Client` plumbing.

## Resolution

- `e2e/pkg/framework/profile_registry.go` and `e2e/profiles/all/imports.go` now own runner capabilities like GPU enablement and extra local images, removing those concerns from `runner.go` name switches.
- `e2e/pkg/stacks/gateway/stack.go` now owns the default semantic-router/envoy/ai-gateway lifecycle, and the main gateway-based profiles compose through that module instead of duplicating setup, verification, and teardown code.
- `e2e/pkg/fixtures` now provides reusable service sessions plus typed chat and Responses API clients, while `e2e/testcases/common.go` wraps that layer for compatibility.
