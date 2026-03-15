# TD012: Canonical v0.3 Routing Contract Still Lacks a LoRA Catalog Surface

## Status

Open

## Scope

canonical routing model catalog versus runtime-only LoRA adapter support

## Summary

The runtime router and several compatibility/public surfaces still understand `decision.modelRefs[].lora_name` and `model_config[*].loras`, but the canonical v0.3 public contract has no steady-state place to declare those LoRA adapters under `version/listeners/providers/routing/global`. As a result, user-facing canonical assets now avoid LoRA examples, the exhaustive reference-config contract intentionally excludes `lora_name`, and the repo still carries an exposed feature that the supported canonical config cannot describe end to end.

## Evidence

- [src/semantic-router/pkg/config/decision_config.go](../../../src/semantic-router/pkg/config/decision_config.go)
- [src/semantic-router/pkg/config/model_config_types.go](../../../src/semantic-router/pkg/config/model_config_types.go)
- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go)
- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/reference_config_contract_test.go](../../../src/semantic-router/pkg/config/reference_config_contract_test.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml](../../../deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml)
- [config/decision/single/domain-computer-science.yaml](../../../config/decision/single/domain-computer-science.yaml)

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
