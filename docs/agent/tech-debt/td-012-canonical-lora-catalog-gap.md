# TD012: Canonical v0.3 Routing Contract Still Lacks a LoRA Catalog Surface

## Status

Closed

## Scope

canonical routing model catalog versus runtime-only LoRA adapter support

## Summary

The canonical v0.3 routing contract now owns the LoRA adapter catalog under `routing.modelCards[].loras`, and every active surface that exposes `decision.modelRefs[].lora_name` now resolves against that same catalog. The repo no longer relies on runtime-only `model_config[*].loras` as the only end-to-end path.

## Evidence

- [src/semantic-router/pkg/config/decision_config.go](../../../src/semantic-router/pkg/config/decision_config.go)
- [src/semantic-router/pkg/config/model_config_types.go](../../../src/semantic-router/pkg/config/model_config_types.go)
- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go)
- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/canonical_export.go](../../../src/semantic-router/pkg/config/canonical_export.go)
- [src/semantic-router/pkg/config/reference_config_contract_test.go](../../../src/semantic-router/pkg/config/reference_config_contract_test.go)
- [src/semantic-router/pkg/config/reference_config_routing_surface_test.go](../../../src/semantic-router/pkg/config/reference_config_routing_surface_test.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/config_migration.py](../../../src/vllm-sr/cli/config_migration.py)
- [src/vllm-sr/cli/validator.py](../../../src/vllm-sr/cli/validator.py)
- [src/vllm-sr/tests/test_config_migrate.py](../../../src/vllm-sr/tests/test_config_migrate.py)
- [src/semantic-router/pkg/dsl/compiler_models.go](../../../src/semantic-router/pkg/dsl/compiler_models.go)
- [src/semantic-router/pkg/dsl/routing_contract.go](../../../src/semantic-router/pkg/dsl/routing_contract.go)
- [src/semantic-router/pkg/dsl/validator.go](../../../src/semantic-router/pkg/dsl/validator.go)
- [src/semantic-router/pkg/dsl/routing_contract_test.go](../../../src/semantic-router/pkg/dsl/routing_contract_test.go)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/configPageSupport.ts](../../../dashboard/frontend/src/pages/configPageSupport.ts)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/operator/controllers/backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
- [deploy/operator/controllers/semanticrouter_controller.go](../../../deploy/operator/controllers/semanticrouter_controller.go)
- [deploy/operator/controllers/semanticrouter_controller_test.go](../../../deploy/operator/controllers/semanticrouter_controller_test.go)
- [config/config.yaml](../../../config/config.yaml)
- [config/decision/single/domain-computer-science.yaml](../../../config/decision/single/domain-computer-science.yaml)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/proposals/unified-config-contract-v0-3.md](../../../website/docs/proposals/unified-config-contract-v0-3.md)

## Why It Matters

- Canonical config is now enforced as the public source of truth, so any capability that still only exists in runtime-only or legacy model-config fields becomes impossible to express in the supported steady-state workflow.
- Keeping `lora_name` visible on decision model refs without a corresponding canonical catalog causes confusion in docs, examples, and dashboard/DSL editing flows.
- The repo can no longer honestly ship exhaustive public config assets while this seam stays implicit.

## Desired End State

- Canonical `routing.modelCards` or another explicit public v0.3 model-catalog surface owns the list of available LoRA adapters for each logical model.
- Decision `modelRefs[].lora_name` is either fully supported through that canonical catalog across CLI/router/dashboard/DSL/operator surfaces or removed from the public contract.

## Exit Criteria

- Canonical config has one explicit, documented LoRA catalog field with parser, validator, CLI, dashboard, and DSL support.
- `config/config.yaml` and `config/decision/` can include LoRA examples without relying on runtime-only legacy fields.
- The reference-config contract test no longer needs to exclude `lora_name`.

## Retirement Notes

- Canonical routing config now exposes `routing.modelCards[].loras`, and the router parser/normalizer validates `decision.modelRefs[].lora_name` against that catalog.
- CLI migration converts legacy flat `model_config[*].loras`, `vllm_endpoints`, and `provider_profiles` into canonical `routing/providers` structures instead of leaving LoRA support behind in migration-only runtime fields.
- DSL compile/decompile/validation now carries model-catalog LoRA entries as part of the routing-owned surface.
- Dashboard decision editing preserves and emits `lora_name`, while the operator's `spec.vllmEndpoints[].loras` adapter now projects into canonical `routing.modelCards[].loras`.
