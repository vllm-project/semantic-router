package config

import (
	"fmt"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

const (
	ReasoningModeEnabled  = "enabled"
	ReasoningModeDisabled = "disabled"
	ReasoningModeAdaptive = "adaptive"
)

func validateModelRefReasoningControl(
	cfg *RouterConfig,
	decisionName string,
	index int,
	modelRef ModelRef,
) error {
	path := fmt.Sprintf("decision '%s', modelRefs[%d] (%s)", decisionName, index, modelRef.Model)
	effort, mode, err := validateReasoningControlSyntax(modelRef, path)
	if err != nil {
		return err
	}

	family := cfg.GetModelReasoningFamily(modelRef.Model)
	useReasoning := modelRef.UseReasoning != nil && *modelRef.UseReasoning
	if family == nil {
		return validateFamilylessReasoningControl(cfg, path, mode)
	}
	if err := validateReasoningModeSelection(family, path, mode, useReasoning); err != nil {
		return err
	}
	if !useReasoning {
		return validateDisabledModelReasoningControl(cfg, path, modelRef.Model, family, mode, effort)
	}
	return validateEnabledModelReasoningControl(cfg, path, modelRef.Model, family, mode, effort)
}

func validateReasoningControlSyntax(modelRef ModelRef, path string) (string, string, error) {
	effort := strings.TrimSpace(modelRef.ReasoningEffort)
	if effort != modelRef.ReasoningEffort {
		return "", "", fmt.Errorf("%s: reasoning_effort must not contain surrounding whitespace", path)
	}
	mode := strings.TrimSpace(modelRef.ReasoningMode)
	if mode != modelRef.ReasoningMode {
		return "", "", fmt.Errorf("%s: reasoning_mode must not contain surrounding whitespace", path)
	}
	if mode != "" && !isSupportedReasoningMode(mode) {
		return "", "", fmt.Errorf("%s: reasoning_mode %q must be enabled, disabled, or adaptive", path, mode)
	}
	return effort, mode, nil
}

func validateFamilylessReasoningControl(cfg *RouterConfig, path, mode string) error {
	// Custom models without a family retain the legacy pass-through contract:
	// use_reasoning and reasoning_effort may still describe an upstream that
	// the operator owns. reasoning_mode requires a known projection.
	if mode != "" && cfg.EffectiveModelRegistry != nil {
		return fmt.Errorf("%s: reasoning_mode requires a reasoning family", path)
	}
	return nil
}

func validateReasoningModeSelection(
	family *ReasoningFamilyConfig,
	path string,
	mode string,
	useReasoning bool,
) error {
	if mode == "" {
		return nil
	}
	if !reasoningFamilyModeAllowed(family, mode) {
		return fmt.Errorf("%s: reasoning_mode %q is not supported by the model's reasoning family", path, mode)
	}
	if useReasoning != (mode != ReasoningModeDisabled) {
		return fmt.Errorf("%s: use_reasoning conflicts with reasoning_mode %q", path, mode)
	}
	return nil
}

func validateDisabledModelReasoningControl(
	cfg *RouterConfig,
	path string,
	model string,
	family *ReasoningFamilyConfig,
	mode string,
	effort string,
) error {
	if !reasoningFamilyCanDisableConfig(family) {
		return fmt.Errorf("%s: use_reasoning=false is not supported by this always-on reasoning family", path)
	}
	if effort != "" {
		return fmt.Errorf("%s: reasoning_effort cannot be set while reasoning is disabled", path)
	}
	return validateProviderReasoningControl(cfg, path, model, family, false, mode, "")
}

func validateEnabledModelReasoningControl(
	cfg *RouterConfig,
	path string,
	model string,
	family *ReasoningFamilyConfig,
	mode string,
	effort string,
) error {
	if effort == "" {
		return validateProviderReasoningControl(cfg, path, model, family, true, mode, "")
	}
	if family.Type == ReasoningFamilyTypeReasoningMode || family.Type == ReasoningFamilyTypeChatTemplateKwargs {
		return fmt.Errorf("%s: reasoning_effort cannot be used with this mode-only reasoning family", path)
	}
	if !reasoningFamilyAllowsLevel(family, effort) {
		return fmt.Errorf("%s: reasoning_effort %q is not supported by the model's reasoning family", path, effort)
	}
	return validateProviderReasoningControl(cfg, path, model, family, true, mode, effort)
}

func validateProviderReasoningControl(
	cfg *RouterConfig,
	path string,
	model string,
	family *ReasoningFamilyConfig,
	useReasoning bool,
	mode string,
	effort string,
) error {
	params, ok := cfg.ModelConfig[model]
	if !ok || len(params.PreferredEndpoints) == 0 {
		return nil
	}
	effectiveMode, effectiveEffort := effectiveProviderReasoningValues(family, useReasoning, mode, effort)
	for _, endpoint := range params.PreferredEndpoints {
		profile, exists := cfg.ProviderProfiles[endpoint]
		if !exists {
			continue
		}
		if err := validateProviderProfileReasoningControl(
			profile, path, family, useReasoning, effectiveMode, effectiveEffort,
		); err != nil {
			return err
		}
	}
	return nil
}

func effectiveProviderReasoningValues(
	family *ReasoningFamilyConfig,
	useReasoning bool,
	mode string,
	effort string,
) (string, string) {
	if mode == "" {
		mode = ReasoningModeDisabled
		if useReasoning {
			mode = family.DefaultMode
		}
	}
	if useReasoning && effort == "" {
		effort = family.Default
	}
	return mode, effort
}

func validateProviderProfileReasoningControl(
	profile ProviderProfile,
	path string,
	family *ReasoningFamilyConfig,
	useReasoning bool,
	effectiveMode string,
	effectiveEffort string,
) error {
	transport, err := profile.ResolveReasoningTransport()
	if err != nil {
		return fmt.Errorf("%s: cannot resolve provider %q reasoning transport: %w", path, profile.Type, err)
	}
	if !transport.SupportsFamilyType(family.Type) {
		return fmt.Errorf(
			"%s: provider %q reasoning transport %q cannot project family type %q",
			path, profile.Type, transport, family.Type,
		)
	}
	if !useReasoning && family.ActivationParameter != "" && family.Disabled == "" &&
		transport == modelcatalog.ReasoningTransportTopLevelEffort {
		return fmt.Errorf(
			"%s: provider %q reasoning transport %q cannot project the model's independent disabled switch",
			path, profile.Type, transport,
		)
	}
	if len(profile.ReasoningModes) > 0 && !reasoningValueAllowed(profile.ReasoningModes, effectiveMode) {
		return fmt.Errorf(
			"%s: reasoning_mode %q is not supported by provider %q for this model",
			path, effectiveMode, profile.Type,
		)
	}
	if effectiveEffort != "" && len(profile.ReasoningEfforts) > 0 &&
		!reasoningValueAllowed(profile.ReasoningEfforts, effectiveEffort) {
		return fmt.Errorf(
			"%s: reasoning_effort %q is not supported by provider %q for this model",
			path, effectiveEffort, profile.Type,
		)
	}
	return nil
}

func reasoningValueAllowed(values []string, candidate string) bool {
	for _, value := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func reasoningFamilyCanDisableConfig(family *ReasoningFamilyConfig) bool {
	if family == nil {
		return false
	}
	return family.ActivationParameter != "" || family.Disabled != "" ||
		reasoningFamilyModeAllowed(family, ReasoningModeDisabled)
}

func isSupportedReasoningMode(mode string) bool {
	switch mode {
	case ReasoningModeEnabled, ReasoningModeDisabled, ReasoningModeAdaptive:
		return true
	default:
		return false
	}
}

func reasoningFamilyModeAllowed(family *ReasoningFamilyConfig, mode string) bool {
	for _, supported := range family.Modes {
		if supported == mode {
			return true
		}
	}
	return false
}

func reasoningFamilyAllowsLevel(family *ReasoningFamilyConfig, level string) bool {
	for _, supported := range family.Levels {
		if supported == level {
			return true
		}
	}
	return false
}
