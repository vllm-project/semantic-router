# TD004: Python CLI and Kubernetes Workflow Separation

## Status

Closed

## Scope

environment orchestration and user workflow

## Summary

The Python CLI now treats Docker and Kubernetes as deployment backends behind one lifecycle surface instead of forcing users onto separate orchestration models. `vllm-sr serve`, `status`, `logs`, `dashboard`, and `stop` all dispatch through the shared `DeploymentBackend` seam, with `DockerBackend` as the default local target and `K8sBackend` selected through `--target k8s`. The CLI docs and regression tests now make that contract explicit, so this workflow split no longer needs to stay open as active debt.

## Evidence

- [src/vllm-sr/cli/deployment_backend.py](../../../src/vllm-sr/cli/deployment_backend.py)
- [src/vllm-sr/cli/docker_backend.py](../../../src/vllm-sr/cli/docker_backend.py)
- [src/vllm-sr/cli/k8s_backend.py](../../../src/vllm-sr/cli/k8s_backend.py)
- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/tests/test_deployment_backend.py](../../../src/vllm-sr/tests/test_deployment_backend.py)
- [docs/agent/environments.md](../environments.md)
- [src/vllm-sr/README.md](../../../src/vllm-sr/README.md)

## Why It Matters

- The Python CLI is strongly oriented around local container lifecycle and does not provide a comparable first-class orchestration path for Kubernetes environments.
- This creates an environment split where local users and Kubernetes users learn different control surfaces and config flows.
- It also makes it harder to provide one consistent product story across local dev, cluster deployment, and dashboard operations.

## Desired End State

- The CLI and environment management model expose a more consistent experience across local and Kubernetes workflows.
- Environment differences are treated as deployment backends, not separate product surfaces.

## Exit Criteria

- Kubernetes deployment and lifecycle management have a coherent path within the shared CLI or a clearly unified orchestration interface.
- Users do not need to mentally switch between unrelated environment management models for common operations.

## Retirement Notes

- `cli/deployment_backend.py` now owns the shared deployment-target contract instead of leaving Docker and Kubernetes as unrelated user workflows.
- `cli/commands/runtime.py` routes `serve`, `status`, `logs`, `dashboard`, and `stop` through `_build_backend(...)`, so the user-facing verbs stay stable while the backend changes.
- `src/vllm-sr/tests/test_deployment_backend.py` now covers both the default Docker path and the Kubernetes path for the shared lifecycle commands.
- `docs/agent/environments.md` and `src/vllm-sr/README.md` both describe Kubernetes as the same CLI surface selected with `--target k8s`.

## Validation

- `pytest src/vllm-sr/tests/test_deployment_backend.py`
