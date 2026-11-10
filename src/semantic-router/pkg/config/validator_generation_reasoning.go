package config

import (
	"fmt"
)

// validateReasoningSetting checks a chat model's reasoning setting against
// its reasoning family.
func validateReasoningSetting(cfg *RouterConfig, path string, reasoning *GenerationReasoning) error {
	family, exists := cfg.ReasoningFamilies[reasoning.Family]
	if !exists {
		return fmt.Errorf("%s family %q is not configured", path, reasoning.Family)
	}
	if reasoning.UseReasoning == nil {
		return fmt.Errorf("%s use_reasoning is required", path)
	}
	effort := reasoning.ReasoningEffort
	if !*reasoning.UseReasoning {
		if !generationReasoningCanDisable(&family) {
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

func generationReasoningCanDisable(family *ReasoningFamilyConfig) bool {
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
