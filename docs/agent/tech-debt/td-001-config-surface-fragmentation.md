# TD001: Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard

## Status

Open

## Scope

configuration architecture

## Summary

The same conceptual router configuration is represented in multiple schemas and translated between them.

## Evidence

- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/parser.py](../../../src/vllm-sr/cli/parser.py)
- [src/vllm-sr/cli/validator.py](../../../src/vllm-sr/cli/validator.py)
- [src/vllm-sr/cli/merger.py](../../../src/vllm-sr/cli/merger.py)
- [src/semantic-router/pkg/dsl/emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)

## Why It Matters

- The same conceptual router configuration is represented in multiple schemas and translated between them.
- A single config feature can require synchronized edits in Go router config, Python CLI models, merge/translation logic, dashboard editing UI, and Kubernetes CRD paths.
- This increases drift risk and makes feature delivery slower and less reliable.

## Desired End State

- One canonical config contract with thinner adapters for CLI, dashboard, and Kubernetes deployment.
- Translation layers exist only where representation changes are unavoidable.

## Exit Criteria

- Adding a config feature no longer requires parallel structural changes across several independent schemas for the common path.
- Router, CLI, dashboard, and operator paths share a clearer single source of truth for config shape.
