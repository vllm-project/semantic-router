# TD002: Config Portability Gap Between Local Docker and Kubernetes Deployments

## Status

Open

## Scope

environment and deployment configuration

## Summary

Local Docker startup, repo config examples, and Kubernetes deployment paths do not share one portable config story.

## Evidence

- [src/vllm-sr/cli/templates/config.template.yaml](../../../src/vllm-sr/cli/templates/config.template.yaml)
- [src/vllm-sr/cli/templates/router-defaults.yaml](../../../src/vllm-sr/cli/templates/router-defaults.yaml)
- [config/config.yaml](../../../config/config.yaml)
- [deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local](../../../deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local)
- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go)

## Why It Matters

- Local Docker startup, repo config examples, and Kubernetes/operator deployment paths do not share one portable config story.
- The `config/` directory mixes legacy and environment-specific examples that are not consistently reusable across local and Kubernetes flows.
- Kubernetes mode currently needs special validation and loading behavior instead of looking like the same config model deployed differently.

## Desired End State

- A clearer split between canonical portable config, environment overlays, and legacy examples.
- Local Docker, AMD, and Kubernetes paths can consume the same conceptual config with predictable adapters.

## Exit Criteria

- The primary local and Kubernetes workflows can start from the same canonical config structure or a formally defined overlay system.
- Legacy-only examples are either retired or explicitly isolated from the default path.
