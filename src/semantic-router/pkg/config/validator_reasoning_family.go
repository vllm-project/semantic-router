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
			return fmt.Errorf("reasoning_families: family name must not be empty")
		}
		switch family.Type {
		case ReasoningFamilyTypeChatTemplateKwargs,
			ReasoningFamilyTypeReasoningEffort,
			ReasoningFamilyTypeTopLevelReasoningEffort:
		default:
			return fmt.Errorf(
				"reasoning_families[%q].type: unsupported value %q (supported: %s, %s, %s)",
				familyName,
				family.Type,
				ReasoningFamilyTypeChatTemplateKwargs,
				ReasoningFamilyTypeReasoningEffort,
				ReasoningFamilyTypeTopLevelReasoningEffort,
			)
		}
		if strings.TrimSpace(family.Parameter) == "" {
			return fmt.Errorf(
				"reasoning_families[%q].parameter must not be empty",
				familyName,
			)
		}
		if family.Type == ReasoningFamilyTypeTopLevelReasoningEffort && family.Parameter != "reasoning_effort" {
			return fmt.Errorf(
				"reasoning_families[%q].parameter must be %q for type %s",
				familyName,
				"reasoning_effort",
				ReasoningFamilyTypeTopLevelReasoningEffort,
			)
		}
	}

	return nil
}
