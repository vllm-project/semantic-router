package config

import (
	"fmt"
	"strings"
)

// validateExternalModelContracts checks the external[] models: unique names
// and the reasoning settings each one declares.
func validateExternalModelContracts(cfg *RouterConfig) error {
	if err := validateExternalModelNames(cfg.ExternalModels); err != nil {
		return err
	}
	return validateExternalModelReasoningContracts(cfg)
}

func validateExternalModelReasoningContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for i := range cfg.ExternalModels {
		external := &cfg.ExternalModels[i]
		path := fmt.Sprintf("global.model_catalog.external[%d]", i)
		if external.Name != "" {
			path = fmt.Sprintf("global.model_catalog.external[%q]", external.Name)
		}
		if err := validateExternalModelReasoningSyntax(external, path); err != nil {
			return err
		}
	}
	return nil
}

func validateExternalModelReasoningSyntax(
	external *ExternalModelConfig,
	path string,
) error {
	if external == nil || external.Reasoning == nil {
		return nil
	}
	reasoning := external.Reasoning
	if strings.TrimSpace(reasoning.Family) == "" {
		return fmt.Errorf("%s.reasoning.family is required", path)
	}
	if strings.TrimSpace(reasoning.Family) != reasoning.Family {
		return fmt.Errorf("%s.reasoning.family must not contain surrounding whitespace", path)
	}
	if reasoning.UseReasoning == nil {
		return fmt.Errorf("%s.reasoning.use_reasoning is required", path)
	}
	if strings.TrimSpace(reasoning.ReasoningEffort) != reasoning.ReasoningEffort {
		return fmt.Errorf("%s.reasoning.reasoning_effort must not contain surrounding whitespace", path)
	}
	return nil
}

func validateExternalModelNames(models []ExternalModelConfig) error {
	externalNames := make(map[string]struct{})
	for _, external := range models {
		if external.Name == "" {
			continue
		}
		if _, exists := externalNames[external.Name]; exists {
			return fmt.Errorf("global.model_catalog.external: duplicate name %q", external.Name)
		}
		externalNames[external.Name] = struct{}{}
	}
	return nil
}

// validateReasoningSetting checks a chat model's reasoning setting against
// its reasoning family.
func validateReasoningSetting(cfg *RouterConfig, path string, reasoning *ExternalModelReasoningConfig) error {
	family, exists := cfg.ReasoningFamilies[reasoning.Family]
	if !exists {
		return fmt.Errorf("%s family %q is not configured", path, reasoning.Family)
	}
	if reasoning.UseReasoning == nil {
		return fmt.Errorf("%s use_reasoning is required", path)
	}
	effort := reasoning.ReasoningEffort
	if !*reasoning.UseReasoning {
		if !externalReasoningFamilyCanDisable(&family) {
			return fmt.Errorf("%s cannot disable always-on family %q", path, reasoning.Family)
		}
		if effort != "" {
			return fmt.Errorf("%s reasoning_effort cannot be set while reasoning is disabled", path)
		}
		return nil
	}
	if effort == "" {
		return nil
	}
	if family.Type == ReasoningFamilyTypeReasoningMode || family.Type == ReasoningFamilyTypeChatTemplateKwargs {
		return fmt.Errorf("%s reasoning_effort cannot be used with mode-only family %q", path, reasoning.Family)
	}
	if !reasoningFamilyAllowsLevel(&family, effort) {
		return fmt.Errorf("%s reasoning_effort %q is not supported by family %q", path, effort, reasoning.Family)
	}
	return nil
}

func externalReasoningFamilyCanDisable(family *ReasoningFamilyConfig) bool {
	if family == nil {
		return false
	}
	if len(family.Modes) > 0 && !reasoningFamilyModeAllowed(family, ReasoningModeDisabled) {
		return false
	}
	if family.Type == ReasoningFamilyTypeTopLevelReasoningEffort {
		return family.Disabled != ""
	}
	return reasoningFamilyCanDisableConfig(family)
}
