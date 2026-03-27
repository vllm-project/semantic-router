# TD037: Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI

## Status

Open

## Scope

`src/vllm-sr/cli/**`, `e2e/pkg/framework/**`, `e2e/pkg/fixtures/**`, `e2e/pkg/testmatrix/testcases.go`, `e2e/profiles/**`, `e2e/testing/vllm-sr-cli/**`, `.github/workflows/integration-test-*.yml`, `.github/workflows/test-and-build.yml`, `tools/agent/e2e-profile-map.yaml`, `tools/agent/repo-manifest.yaml`, and `tools/make/{docker,e2e}.mk`

## Summary

The desired dev integration story is `make vllm-sr-dev`, then `vllm-sr` owns environment bring-up for either `docker` or `k8s` backed by Kind, and both environments exercise the same shared `dashboard` and `core routing` suites. The current repository still splits those responsibilities across multiple frameworks. `vllm-sr` owns local Docker lifecycle, but Kind cluster creation, local image loading, and most Kubernetes-facing test setup still live in the Go E2E runner. The shared router and dashboard testcase groups also remain Kubernetes-coupled through port-forward and service-session fixtures, while the default CI topology still fans out across several environment-specific profiles and standalone integration workflows.

## Evidence

- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/k8s_backend.py](../../../src/vllm-sr/cli/k8s_backend.py)
- [tools/make/docker.mk](../../../tools/make/docker.mk)
- [tools/make/e2e.mk](../../../tools/make/e2e.mk)
- [e2e/pkg/framework/runner.go](../../../e2e/pkg/framework/runner.go)
- [e2e/pkg/framework/runner_lifecycle.go](../../../e2e/pkg/framework/runner_lifecycle.go)
- [e2e/pkg/cluster/kind.go](../../../e2e/pkg/cluster/kind.go)
- [e2e/pkg/fixtures/session.go](../../../e2e/pkg/fixtures/session.go)
- [e2e/testcases/common.go](../../../e2e/testcases/common.go)
- [e2e/pkg/testmatrix/testcases.go](../../../e2e/pkg/testmatrix/testcases.go)
- [e2e/profiles/ai-gateway/profile.go](../../../e2e/profiles/ai-gateway/profile.go)
- [e2e/profiles/dashboard/profile.go](../../../e2e/profiles/dashboard/profile.go)
- [.github/workflows/integration-test-k8s.yml](../../../.github/workflows/integration-test-k8s.yml)
- [.github/workflows/integration-test-memory.yml](../../../.github/workflows/integration-test-memory.yml)
- [.github/workflows/integration-test-vllm-sr-cli.yml](../../../.github/workflows/integration-test-vllm-sr-cli.yml)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [tools/agent/repo-manifest.yaml](../../../tools/agent/repo-manifest.yaml)

## Why It Matters

- Contributors still need to reason about at least three integration control planes: `vllm-sr` Docker lifecycle, Go E2E Kind lifecycle, and workflow-specific integration entrypoints.
- Expensive PR matrices continue to spend quota on environment-specific profiles even when the intended durable dev contract is only `docker` or `k8s(kubernetes)` plus shared `dashboard` and `core routing` coverage.
- The current shared testcase groups are not actually environment-agnostic, so the repository cannot yet express one reusable integration contract that runs against both local Docker and Kind-backed Kubernetes.
- As long as environment ownership and suite ownership stay split, removing profiles such as `aibrix` or `llm-d` from the default path remains a CI-only trim instead of a coherent architecture simplification.

## Desired End State

- `make vllm-sr-dev` produces the local artifacts consumed by one `vllm-sr`-owned integration workflow.
- `vllm-sr serve --target docker|k8s` is the primary dev environment entrypoint, and the `k8s` target owns Kind cluster lifecycle, local image loading, and the kubernetes baseline deployment defaults for development.
- `dashboard` and `core routing` become explicit shared integration suites that can run against either Docker-local or Kind-backed Kubernetes through one environment-agnostic session or endpoint abstraction.
- Environment-specific profiles that are not part of that default dev contract either move to targeted or nightly coverage or are retired entirely.
- CI and harness manifests describe the same reduced default PR surface instead of keeping a broad environment matrix as the steady-state baseline.

## Exit Criteria

- `vllm-sr` can launch and tear down both supported dev environments, including Kind-backed Kubernetes, without delegating cluster lifecycle to the Go E2E runner.
- The shared `dashboard` and `core routing` suites no longer require Kubernetes-specific port-forward fixtures and can run against both Docker and Kind targets through one transport abstraction.
- The default PR integration path is reduced to the supported dev environments and shared suites, while non-core environment profiles are removed or isolated behind targeted triggers or nightly/manual workflows.
- `tools/agent/e2e-profile-map.yaml`, `tools/agent/repo-manifest.yaml`, workflow definitions, and contributor docs all describe the same reduced integration topology.
