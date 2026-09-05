package config

import (
	"fmt"
	"sort"
	"strings"
)

func validateReasoningFamilyContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}

	familyNames := make([]string, 0, len(cfg.ReasoningFamilies))
	for familyName := range cfg.ReasoningFamilies {
		familyNames = append(familyNames, familyName)
	}
	sort.Strings(familyNames)

	for _, familyName := range familyNames {
		if err := validateReasoningFamilyContract(familyName, cfg.ReasoningFamilies[familyName]); err != nil {
			return err
		}
	}

	return nil
}

func validateReasoningFamilyContract(name string, family ReasoningFamilyConfig) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("providers.defaults.reasoning_families: family name must not be empty")
	}
	if err := validateReasoningFamilyType(name, family.Type); err != nil {
		return err
	}
	if strings.TrimSpace(family.Parameter) == "" {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].parameter must not be empty",
			name,
		)
	}
	if err := validateReasoningFamilyActivationParameter(name, family); err != nil {
		return err
	}
	if family.Type == ReasoningFamilyTypeTopLevelReasoningEffort && family.Parameter != "reasoning_effort" {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].parameter must be %q for type %s",
			name,
			"reasoning_effort",
			ReasoningFamilyTypeTopLevelReasoningEffort,
		)
	}
	return validateReasoningFamilyLevels(name, family)
}

func validateReasoningFamilyType(name, familyType string) error {
	switch familyType {
	case ReasoningFamilyTypeChatTemplateKwargs,
		ReasoningFamilyTypeReasoningEffort,
		ReasoningFamilyTypeTopLevelReasoningEffort:
		return nil
	default:
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].type: unsupported value %q (supported: %s, %s, %s)",
			name,
			familyType,
			ReasoningFamilyTypeChatTemplateKwargs,
			ReasoningFamilyTypeReasoningEffort,
			ReasoningFamilyTypeTopLevelReasoningEffort,
		)
	}
}

func validateReasoningFamilyActivationParameter(name string, family ReasoningFamilyConfig) error {
	if family.ActivationParameter == "" {
		return nil
	}
	if strings.TrimSpace(family.ActivationParameter) == "" {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].activation_parameter must not be blank",
			name,
		)
	}
	if family.ActivationParameter == family.Parameter {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].activation_parameter must differ from parameter",
			name,
		)
	}
	if family.Type != ReasoningFamilyTypeReasoningEffort {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].activation_parameter requires type %s",
			name,
			ReasoningFamilyTypeReasoningEffort,
		)
	}
	return nil
}

func validateReasoningFamilyLevels(name string, family ReasoningFamilyConfig) error {
	if len(family.Levels) == 0 {
		if family.Default != "" || family.Disabled != "" {
			return fmt.Errorf(
				"providers.defaults.reasoning_families[%q].levels must be set when default or disabled is set",
				name,
			)
		}
		// Legacy/custom families did not declare a finite level set. Keep those
		// valid while every built-in family materializes the complete contract.
		return nil
	}

	seen := make(map[string]struct{}, len(family.Levels))
	for _, level := range family.Levels {
		if strings.TrimSpace(level) == "" {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].levels must not contain an empty value", name)
		}
		if _, exists := seen[level]; exists {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].levels contains duplicate %q", name, level)
		}
		seen[level] = struct{}{}
	}
	if _, ok := seen[family.Default]; !ok {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].default %q must be listed in levels", name, family.Default)
	}
	if family.Disabled != "" {
		if _, ok := seen[family.Disabled]; !ok {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].disabled %q must be listed in levels", name, family.Disabled)
		}
	}
	return nil
}
