package config

import "fmt"

func validateAdvancedToolFilteringConfig(cfg *RouterConfig) error {
	if cfg == nil || cfg.Tools.AdvancedFiltering == nil {
		return nil
	}

	advanced := cfg.Tools.AdvancedFiltering
	if !advanced.Enabled {
		return nil
	}

	for _, field := range []struct {
		name  string
		value *int
	}{
		{name: "candidate_pool_size", value: advanced.CandidatePoolSize},
		{name: "min_lexical_overlap", value: advanced.MinLexicalOverlap},
	} {
		if err := validateAdvancedToolFilteringNonNegativeInt(field.name, field.value); err != nil {
			return err
		}
	}

	for _, field := range []struct {
		name  string
		value *float32
	}{
		{name: "min_combined_score", value: advanced.MinCombinedScore},
		{name: "category_confidence_threshold", value: advanced.CategoryConfidenceThreshold},
	} {
		if err := validateAdvancedToolFilteringUnitFloat(field.name, field.value); err != nil {
			return err
		}
	}

	for _, field := range []struct {
		name  string
		value *float32
	}{
		{"embed", advanced.Weights.Embed},
		{"lexical", advanced.Weights.Lexical},
		{"tag", advanced.Weights.Tag},
		{"name", advanced.Weights.Name},
		{"category", advanced.Weights.Category},
	} {
		if err := validateAdvancedToolFilteringUnitFloat("weights."+field.name, field.value); err != nil {
			return err
		}
	}

	return nil
}

func validateAdvancedToolFilteringNonNegativeInt(name string, value *int) error {
	if value == nil || *value >= 0 {
		return nil
	}
	return fmt.Errorf("tools.advanced_filtering.%s must be >= 0", name)
}

func validateAdvancedToolFilteringUnitFloat(name string, value *float32) error {
	if value == nil || (*value >= 0.0 && *value <= 1.0) {
		return nil
	}
	return fmt.Errorf("tools.advanced_filtering.%s must be between 0.0 and 1.0", name)
}
