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
	if err := validateReasoningFamilyEffortFlags(name, family); err != nil {
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

func validateReasoningFamilyEffortFlags(name string, family ReasoningFamilyConfig) error {
	if len(family.EffortFlags) == 0 {
		return nil
	}
	if family.Type != ReasoningFamilyTypeReasoningEffort {
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].effort_flags requires type %s",
			name,
			ReasoningFamilyTypeReasoningEffort,
		)
	}
	if family.ActivationParameter == "" {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags requires activation_parameter", name)
	}
	levels := make(map[string]struct{}, len(family.Levels))
	for _, level := range family.Levels {
		levels[level] = struct{}{}
	}
	seenParameters := make(map[string]struct{}, len(family.EffortFlags))
	for effort, parameter := range family.EffortFlags {
		if err := validateReasoningFamilyEffortFlag(name, family, levels, seenParameters, effort, parameter); err != nil {
			return err
		}
		seenParameters[parameter] = struct{}{}
	}
	activeLevelCount := len(family.Levels)
	if family.Disabled != "" {
		if _, ok := levels[family.Disabled]; ok {
			activeLevelCount--
		}
	}
	if activeLevelCount-len(family.EffortFlags) > 1 {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags leaves multiple effort levels indistinguishable by omission", name)
	}
	return nil
}

func validateReasoningFamilyEffortFlag(
	name string,
	family ReasoningFamilyConfig,
	levels map[string]struct{},
	seenParameters map[string]struct{},
	effort string,
	parameter string,
) error {
	if _, ok := levels[effort]; !ok {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags key %q must be listed in levels", name, effort)
	}
	if strings.TrimSpace(parameter) == "" {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags[%q] must not be blank", name, effort)
	}
	if parameter == family.Parameter || parameter == family.ActivationParameter {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags[%q] must differ from parameter and activation_parameter", name, effort)
	}
	if _, exists := seenParameters[parameter]; exists {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].effort_flags contains duplicate parameter %q", name, parameter)
	}
	return nil
}

func validateReasoningFamilyType(name, familyType string) error {
	switch familyType {
	case ReasoningFamilyTypeChatTemplateKwargs,
		ReasoningFamilyTypeReasoningEffort,
		ReasoningFamilyTypeReasoningMode,
		ReasoningFamilyTypeTopLevelReasoningEffort:
		return nil
	default:
		return fmt.Errorf(
			"providers.defaults.reasoning_families[%q].type: unsupported value %q (supported: %s, %s, %s, %s)",
			name,
			familyType,
			ReasoningFamilyTypeChatTemplateKwargs,
			ReasoningFamilyTypeReasoningEffort,
			ReasoningFamilyTypeReasoningMode,
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
		if family.Type == ReasoningFamilyTypeReasoningEffort ||
			family.Type == ReasoningFamilyTypeTopLevelReasoningEffort {
			return fmt.Errorf(
				"providers.defaults.reasoning_families[%q].levels must be set for effort-based reasoning",
				name,
			)
		}
		return validateReasoningModes(name, family)
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
	if family.Default != "" {
		if _, ok := seen[family.Default]; !ok {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].default %q must be listed in levels", name, family.Default)
		}
	}
	return validateReasoningModes(name, family)
}

func validateReasoningModes(name string, family ReasoningFamilyConfig) error {
	if len(family.Modes) == 0 {
		// Inline operator families may predate or intentionally omit the mode
		// contract. A completely omitted pair is valid; built-in families are
		// checked strictly before generation and always materialize both fields.
		if family.DefaultMode == "" {
			return nil
		}
		return fmt.Errorf("providers.defaults.reasoning_families[%q].modes must not be empty when default_mode is set", name)
	}
	seen := make(map[string]struct{}, len(family.Modes))
	for _, mode := range family.Modes {
		switch mode {
		case "enabled", "disabled", "adaptive":
		default:
			return fmt.Errorf("providers.defaults.reasoning_families[%q].modes contains unsupported mode %q", name, mode)
		}
		if _, exists := seen[mode]; exists {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].modes contains duplicate %q", name, mode)
		}
		seen[mode] = struct{}{}
	}
	if family.DefaultMode == "" {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].default_mode must not be empty when modes are set", name)
	}
	if _, ok := seen[family.DefaultMode]; !ok {
		return fmt.Errorf("providers.defaults.reasoning_families[%q].default_mode %q must be listed in modes", name, family.DefaultMode)
	}
	if family.Disabled != "" {
		if _, ok := seen[ReasoningModeDisabled]; !ok {
			return fmt.Errorf("providers.defaults.reasoning_families[%q].disabled requires disabled to be listed in modes", name)
		}
	}
	return nil
}
