# TD004: Python CLI and Kubernetes Workflow Separation

## Status

Closed

## Scope

environment orchestration and user workflow

## Summary

Current source now provides a first-class Kubernetes deployment target through
the same Python CLI command family. `DeploymentBackend` defines a shared
deployment interface, `K8sBackend` implements Helm/kubectl deployment,
teardown, status, logs, dashboard URL lookup, and running-state checks, and
`vllm-sr serve/status/logs/stop/dashboard` all accept `--target k8s`.
`test_deployment_backend.py` covers target resolution, K8s backend helpers, Helm
value translation, Docker backend parity, and CLI target routing.

A 2026-05-25 source recheck found the original TD004 wording stale enough to
close this item. The remaining integration-environment gap is narrower and is
already tracked by TD037: `--target k8s` deploys to an existing cluster, while
Kind lifecycle, kubeconfig discovery, local image loading, and shared-suite
topology are still outside the CLI-owned deployment target. That residual work
should continue under TD037 rather than keeping TD004 open.

The same recheck also narrowed CLI runtime command structure while preserving
the public command contract. Runtime serve help now lives in `runtime_help.py`,
algorithm and platform mutation live in `runtime_config_mutation.py`, and
KB/runtime path helpers live in sibling modules. `runtime.py` is now below the
structure warning threshold.

## Evidence

- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/deployment_backend.py](../../../src/vllm-sr/cli/deployment_backend.py)
- [src/vllm-sr/cli/k8s_backend.py](../../../src/vllm-sr/cli/k8s_backend.py)
- [src/vllm-sr/cli/docker_backend.py](../../../src/vllm-sr/cli/docker_backend.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/commands/runtime_help.py](../../../src/vllm-sr/cli/commands/runtime_help.py)
- [src/vllm-sr/cli/commands/runtime_config_mutation.py](../../../src/vllm-sr/cli/commands/runtime_config_mutation.py)
- [src/vllm-sr/cli/commands/runtime_kb.py](../../../src/vllm-sr/cli/commands/runtime_kb.py)
- [src/vllm-sr/cli/commands/runtime_paths.py](../../../src/vllm-sr/cli/commands/runtime_paths.py)
- [docs/agent/environments.md](../environments.md)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [src/vllm-sr/tests/test_deployment_backend.py](../../../src/vllm-sr/tests/test_deployment_backend.py)
- [docs/agent/tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md](td-037-dev-integration-env-ownership-and-shared-suite-topology.md)

## Why It Matters

- This item is closed because the shared CLI command family now has a Kubernetes
  deployment target.
- Local developer Kind lifecycle and shared-suite topology still matter, but
  they are tracked by TD037.

## Desired End State

- The CLI and environment management model expose a more consistent experience
  across local and Kubernetes workflows.
- Environment differences are treated as deployment backends, not separate
  product surfaces.

## Exit Criteria

- Kubernetes deployment and lifecycle management have a coherent path within
  the shared CLI deployment target.
- Residual local Kind/image integration work is tracked separately under TD037.
