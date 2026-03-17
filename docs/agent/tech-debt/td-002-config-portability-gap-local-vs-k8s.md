# TD002: Config Portability Gap Between Local Docker and Kubernetes Deployments

## Status

Closed

## Scope

environment and deployment configuration

## Summary

The primary local, AMD, Helm, operator, onboarding, and DSL workflows now share the same canonical v0.3 config concepts. Operator wrapping and backend discovery remain documented adapter layers, not a second steady-state config model.

## Evidence

- [src/vllm-sr/cli/templates/config.template.yaml](../../../src/vllm-sr/cli/templates/config.template.yaml)
- [deploy/amd/config.yaml](../../../deploy/amd/config.yaml)
- [deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml](../../../deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/installation/k8s/operator.md](../../../website/docs/installation/k8s/operator.md)

## Why It Matters

- The old portability story mixed canonical config with environment-specific wrappers, which made Kubernetes look like a separate user-facing schema.
- A portable contract only works if the repo clearly distinguishes between canonical config and adapter metadata such as operator wrapping or backend discovery.

## Desired End State

- A clearer split between canonical portable config, environment adapters, and compatibility-only legacy files.
- Local Docker, AMD, Helm, and Kubernetes paths all start from the same conceptual config, with operator-specific wrapping documented as an adapter rather than a second config model.

## Exit Criteria

- The primary local and Kubernetes workflows can start from the same canonical config structure or a formally defined overlay system.
- Legacy-only examples are either retired or explicitly isolated from the default path.
- Local helper commands and docs no longer imply that `.vllm-sr/router-defaults.yaml` is part of the normal portability story, and any auxiliary files under `.vllm-sr/` are clearly documented as references rather than config sources.

All exit criteria are satisfied as of 2026-03-14. The remaining environment-specific differences are explicit deployment adapters (`spec.config`, `spec.vllmEndpoints`, Helm chart values wrapping), not a portability debt in the config contract itself.
