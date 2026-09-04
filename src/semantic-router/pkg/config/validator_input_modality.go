package config

import (
	"fmt"
	"slices"
	"strings"
)

func validateInputModalityContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.InputModalityRules))
	for i, rule := range cfg.InputModalityRules {
		if err := ValidateInputModalityRuleContract(rule); err != nil {
			return fmt.Errorf("routing.signals.input_modality[%d]: %w", i, err)
		}
		normalizedName := strings.ToLower(rule.Name)
		if _, exists := seen[normalizedName]; exists {
			return fmt.Errorf("routing.signals.input_modality[%d]: duplicate name %q", i, rule.Name)
		}
		seen[normalizedName] = struct{}{}
	}
	return nil
}

// ValidateInputModalityRuleContract validates a single input_modality rule.
// Exported so the DSL compiler and validator share the exact same contract.
func ValidateInputModalityRuleContract(rule InputModalityRule) error {
	trimmedName := strings.TrimSpace(rule.Name)
	if trimmedName == "" {
		return fmt.Errorf("name is required")
	}
	if trimmedName != rule.Name {
		return fmt.Errorf("name must not contain surrounding whitespace")
	}
	if !slices.Contains(SupportedInputModalities(), rule.Modality) {
		return fmt.Errorf(
			"modality must be one of %s, got %q",
			strings.Join(SupportedInputModalities(), ", "),
			rule.Modality,
		)
	}
	return nil
}
