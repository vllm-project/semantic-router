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
		family := cfg.ReasoningFamilies[familyName]
		if strings.TrimSpace(familyName) == "" {
			return fmt.Errorf("providers.defaults.reasoning_families: family name must not be empty")
		}
		switch family.Type {
		case ReasoningFamilyTypeChatTemplateKwargs,
			ReasoningFamilyTypeReasoningEffort,
			ReasoningFamilyTypeTopLevelReasoningEffort:
		default:
			return fmt.Errorf(
				"providers.defaults.reasoning_families[%q].type: unsupported value %q (supported: %s, %s, %s)",
				familyName,
				family.Type,
				ReasoningFamilyTypeChatTemplateKwargs,
				ReasoningFamilyTypeReasoningEffort,
				ReasoningFamilyTypeTopLevelReasoningEffort,
			)
		}
		if strings.TrimSpace(family.Parameter) == "" {
			return fmt.Errorf(
				"providers.defaults.reasoning_families[%q].parameter must not be empty",
				familyName,
			)
		}
		if family.Type == ReasoningFamilyTypeTopLevelReasoningEffort && family.Parameter != "reasoning_effort" {
			return fmt.Errorf(
				"providers.defaults.reasoning_families[%q].parameter must be %q for type %s",
				familyName,
				"reasoning_effort",
				ReasoningFamilyTypeTopLevelReasoningEffort,
			)
		}
		if err := validateReasoningFamilyLevels(familyName, family); err != nil {
			return err
		}
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
