# TD037: Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI

## Status

Closed

## Scope

`src/vllm-sr/cli/**`, `e2e/pkg/framework/**`, `e2e/pkg/fixtures/**`, `e2e/pkg/testmatrix/testcases.go`, `e2e/profiles/**`, `e2e/testing/vllm-sr-cli/**`, `.github/workflows/integration-test-*.yml`, `.github/workflows/test-and-build.yml`, `tools/agent/e2e-profile-map.yaml`, `tools/agent/repo-manifest.yaml`, and `tools/make/{docker,e2e}.mk`

## Summary

The repository now converges on one default dev integration story: `make vllm-sr-dev`, then `vllm-sr` owns docker or Kind-backed k8s bring-up, and both environments run the same shared `kubernetes` and `dashboard` suites through the `e2e/testing/vllm-sr-cli` transport harness. The Kind lifecycle no longer needs the Go E2E runner for the steady-state default path, the shared suites no longer depend on Kubernetes-only testcase fixtures, the k8s path now ships the same Envoy listener plane and public OpenAI surface as docker, and the workflow or manifest layer now treats non-core profiles and the memory integration workflow as explicit manual or nightly exceptions instead of part of the default PR topology.

## Evidence

- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/config_translator.py](../../../src/vllm-sr/cli/config_translator.py)
- [src/vllm-sr/cli/k8s_backend.py](../../../src/vllm-sr/cli/k8s_backend.py)
- [src/vllm-sr/cli/kind_cluster.py](../../../src/vllm-sr/cli/kind_cluster.py)
- [src/vllm-sr/cli/templates/envoy.template.yaml](../../../src/vllm-sr/cli/templates/envoy.template.yaml)
- [tools/make/docker.mk](../../../tools/make/docker.mk)
- [deploy/helm/semantic-router/templates/configmap.yaml](../../../deploy/helm/semantic-router/templates/configmap.yaml)
- [deploy/helm/semantic-router/templates/envoy-deployment.yaml](../../../deploy/helm/semantic-router/templates/envoy-deployment.yaml)
- [deploy/helm/semantic-router/templates/envoy-service.yaml](../../../deploy/helm/semantic-router/templates/envoy-service.yaml)
- [e2e/testing/vllm-sr-cli/cli_test_base.py](../../../e2e/testing/vllm-sr-cli/cli_test_base.py)
- [e2e/testing/vllm-sr-cli/test_shared_suites.py](../../../e2e/testing/vllm-sr-cli/test_shared_suites.py)
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

## Retirement Notes

- `src/vllm-sr/cli/k8s_backend.py` now defaults `--target k8s` onto a managed Kind context, while [kind_cluster.py](../../../src/vllm-sr/cli/kind_cluster.py) owns cluster creation, deletion, and local image load for the steady-state dev path.
- `src/vllm-sr/cli/config_translator.py` and [deploy/helm/semantic-router/templates/configmap.yaml](../../../deploy/helm/semantic-router/templates/configmap.yaml) now carry the canonical router config through a raw override path, so the k8s dev runtime no longer inherits chart-default router state that diverges from the shared-suite config under test.
- `src/vllm-sr/cli/k8s_backend.py`, [src/vllm-sr/cli/templates/envoy.template.yaml](../../../src/vllm-sr/cli/templates/envoy.template.yaml), and the Helm Envoy templates now give the k8s target the same listener-plane ownership as docker, including a dedicated `semantic-router-envoy` service that fronts the public OpenAI-compatible port.
- `e2e/testing/vllm-sr-cli/cli_test_base.py` now writes the same smoke-style minimal canonical config used to suppress router-owned mmBERT defaults, and [e2e/testing/vllm-sr-cli/test_shared_suites.py](../../../e2e/testing/vllm-sr-cli/test_shared_suites.py) now treats `/v1/models` as the shared public readiness contract while also waiting for the k8s `llm-katan` fixture service itself to expose `/v1/models` before issuing chat traffic.
- `.github/workflows/integration-test-vllm-sr-cli.yml` and `.github/workflows/integration-test-k8s.yml` now both run the same shared suites, with `docker` and `k8s` as the only steady-state default dev environments.
- `tools/agent/e2e-profile-map.yaml` now keeps non-core profiles in `manual_profile_rules`, and `.github/workflows/integration-test-memory.yml` is retained as a manual/nightly exception instead of part of the default PR matrix.
- `tools/agent/repo-manifest.yaml`, [environments.md](../environments.md), [testing-strategy.md](../testing-strategy.md), and [playbooks/e2e-selection.md](../playbooks/e2e-selection.md) now describe the same reduced docker|k8s shared-suite topology.

## Validation

- `python -m unittest e2e/testing/vllm-sr-cli/test_unit_runtime_topology.py`
- `pytest src/vllm-sr/tests/test_k8s_dev_backend.py`
- `RUN_INTEGRATION_TESTS=true VLLM_SR_TEST_TARGET=k8s VLLM_SR_SHARED_SUITE=kubernetes CONTAINER_RUNTIME=docker PYTHONUNBUFFERED=1 python e2e/testing/vllm-sr-cli/run_cli_tests.py --verbose --integration --pattern shared`
- `RUN_INTEGRATION_TESTS=true VLLM_SR_TEST_TARGET=k8s VLLM_SR_SHARED_SUITE=dashboard CONTAINER_RUNTIME=docker PYTHONUNBUFFERED=1 python e2e/testing/vllm-sr-cli/run_cli_tests.py --verbose --integration --pattern shared`
