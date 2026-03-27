# TD037: Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI

## Status

Open

## Scope

`src/vllm-sr/cli/**`, `e2e/pkg/framework/**`, `e2e/pkg/fixtures/**`, `e2e/pkg/testmatrix/testcases.go`, `e2e/profiles/**`, `e2e/testing/vllm-sr-cli/**`, `.github/workflows/integration-test-*.yml`, `.github/workflows/test-and-build.yml`, `tools/agent/e2e-profile-map.yaml`, `tools/agent/repo-manifest.yaml`, and `tools/make/{docker,e2e}.mk`

## Summary

The desired dev integration story is `make vllm-sr-dev`, then `vllm-sr` owns environment bring-up for either `docker` or `k8s` backed by Kind, and both environments exercise the same shared `dashboard` and `core routing` suites. The repository has already reduced the default Kubernetes PR baseline to `kubernetes` and `dashboard`, and the public baseline profile key now uses `kubernetes` even though the implementation still lives under the `ai-gateway` subtree. The remaining gap is that the repository still splits environment ownership, shared-suite transport, and workflow selection across multiple control planes. `vllm-sr` owns local Docker lifecycle, but Kind cluster creation, local image loading, and most Kubernetes-facing test setup still live in the Go E2E runner. The shared router and dashboard testcase groups also remain Kubernetes-coupled through port-forward and service-session fixtures, while the workflow and harness layers still retain environment-specific profiles and standalone integration workflows outside the intended `docker | k8s(kind)` regression model.

## Current Gap Inventory

1. `src/vllm-sr/cli/k8s_backend.py` still treats `--target k8s` as a Helm deployment against an existing cluster. Kind lifecycle, kubeconfig discovery, and local image load still sit in `e2e/pkg/cluster/kind.go` and `e2e/pkg/framework/{runner,runner_lifecycle}.go`.
2. The shared `dashboard` and `core routing` contracts are named in `e2e/pkg/testmatrix/testcases.go`, but the execution path still depends on Kubernetes-specific types and port-forward fixtures through `e2e/pkg/testcases/registry.go`, `e2e/pkg/fixtures/session.go`, and `e2e/testcases/common.go`.
3. `.github/workflows/integration-test-k8s.yml` now defaults to `kubernetes` and `dashboard`, but it still targets `istio`, `aibrix`, `llm-d`, `production-stack`, `dynamic-config`, `routing-strategies`, `multi-endpoint`, `authz-rbac`, `streaming`, and `ml-model-selection` when their files change.
4. `tools/agent/e2e-profile-map.yaml` and `tools/agent/repo-manifest.yaml` still carry those non-core environment profiles in the canonical harness inventory instead of expressing a fully reduced default dev regression topology.
5. `.github/workflows/integration-test-vllm-sr-cli.yml` and `.github/workflows/integration-test-memory.yml` still run as separate workflow-driven integration entrypoints instead of one `vllm-sr`-owned environment-and-suite model.

## Evidence

- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/k8s_backend.py](../../../src/vllm-sr/cli/k8s_backend.py)
- [tools/make/docker.mk](../../../tools/make/docker.mk)
- [tools/make/e2e.mk](../../../tools/make/e2e.mk)
- [e2e/pkg/framework/runner.go](../../../e2e/pkg/framework/runner.go)
- [e2e/pkg/framework/runner_lifecycle.go](../../../e2e/pkg/framework/runner_lifecycle.go)
- [e2e/pkg/cluster/kind.go](../../../e2e/pkg/cluster/kind.go)
- [e2e/pkg/fixtures/session.go](../../../e2e/pkg/fixtures/session.go)
- [e2e/pkg/testcases/registry.go](../../../e2e/pkg/testcases/registry.go)
- [e2e/testcases/common.go](../../../e2e/testcases/common.go)
- [e2e/pkg/testmatrix/testcases.go](../../../e2e/pkg/testmatrix/testcases.go)
- [e2e/profiles/ai-gateway/profile.go](../../../e2e/profiles/ai-gateway/profile.go)
- [e2e/profiles/dashboard/profile.go](../../../e2e/profiles/dashboard/profile.go)
- [.github/workflows/ci-changes.yml](../../../.github/workflows/ci-changes.yml)
- [.github/workflows/integration-test-k8s.yml](../../../.github/workflows/integration-test-k8s.yml)
- [.github/workflows/integration-test-memory.yml](../../../.github/workflows/integration-test-memory.yml)
- [.github/workflows/integration-test-vllm-sr-cli.yml](../../../.github/workflows/integration-test-vllm-sr-cli.yml)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [tools/agent/repo-manifest.yaml](../../../tools/agent/repo-manifest.yaml)

## Why It Matters

- Contributors still need to reason about at least three integration control planes: `vllm-sr` Docker lifecycle, Go E2E Kind lifecycle, and workflow-specific integration entrypoints.
- Expensive PR and workflow surfaces still spend quota on environment-specific coverage even after the baseline was trimmed to `kubernetes` and `dashboard`, because the non-core profiles and standalone workflows still exist as steady-state trigger paths.
- The current shared testcase groups are not actually environment-agnostic, so the repository cannot yet express one reusable integration contract that runs against both local Docker and Kind-backed Kubernetes through `vllm-sr`.
- As long as environment ownership and suite ownership stay split, removing profiles such as `aibrix` or `llm-d` from the default path remains a matrix trim instead of a coherent architecture simplification.

## Desired End State

- `make vllm-sr-dev` produces the local artifacts consumed by one `vllm-sr`-owned integration workflow.
- `vllm-sr serve --target docker|k8s` is the primary dev environment entrypoint, and the `k8s` target owns Kind cluster lifecycle, local image loading, and the `kubernetes` baseline deployment defaults for development while still being free to reuse the existing `ai-gateway` implementation paths internally.
- `dashboard` and `core routing` become explicit shared integration suites that can run against either Docker-local or Kind-backed Kubernetes through one environment-agnostic session or endpoint abstraction.
- Environment-specific profiles that are not part of that default dev contract are removed from the steady-state PR path and either retired entirely or kept only as documented manual/nightly exceptions.
- CI and harness manifests describe the same reduced default PR surface instead of keeping a broad environment matrix or parallel workflow-only regression model as the steady-state baseline.

## Exit Criteria

- `vllm-sr` can launch and tear down both supported dev environments, including Kind-backed Kubernetes, without delegating cluster lifecycle to the Go E2E runner.
- The shared `dashboard` and `core routing` suites no longer require Kubernetes-specific port-forward fixtures and can run against both Docker and Kind targets through one transport abstraction.
- The default PR integration path is reduced to the supported dev environments and shared suites, while non-core environment profiles are removed from the steady-state PR matrix and any retained exceptions are explicitly documented as manual/nightly-only.
- `tools/agent/e2e-profile-map.yaml`, `tools/agent/repo-manifest.yaml`, workflow definitions, and contributor docs all describe the same reduced integration topology.
