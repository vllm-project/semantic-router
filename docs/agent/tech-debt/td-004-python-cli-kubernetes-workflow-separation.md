# TD004: Python CLI and Kubernetes Workflow Separation

## Status

Open

## Scope

environment orchestration and user workflow

## Summary

The Python CLI is strongly oriented around local container lifecycle and does not provide a comparable first-class orchestration path for Kubernetes environments.

## Evidence

- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [docs/agent/environments.md](../environments.md)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)

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
