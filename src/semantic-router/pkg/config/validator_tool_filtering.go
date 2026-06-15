package config

import (
	"fmt"
	"strings"
)

func validateAdvancedToolFilteringConfig(cfg *RouterConfig) error {
	if cfg == nil || cfg.Tools.AdvancedFiltering == nil {
		return nil
	}
	advanced := cfg.Tools.AdvancedFiltering
	if !advanced.Enabled {
		return nil
	}
	if err := validateAdvancedToolFilteringIntFields(advanced); err != nil {
		return err
	}
	if err := validateAdvancedToolFilteringCoreFloats(advanced); err != nil {
		return err
	}
	if err := validateToolFilteringWeightFloats(advanced.Weights); err != nil {
		return err
	}
	if err := validateRetrievalStrategyValue(advanced.RetrievalStrategy); err != nil {
		return err
	}
	return validateHybridHistorySubconfig(advanced.HybridHistory)
}

func validateAdvancedToolFilteringIntFields(advanced *AdvancedToolFilteringConfig) error {
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
	return nil
}

func validateAdvancedToolFilteringCoreFloats(advanced *AdvancedToolFilteringConfig) error {
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
	return nil
}

func validateToolFilteringWeightFloats(weights ToolFilteringWeights) error {
	for _, field := range []struct {
		name  string
		value *float32
	}{
		{"embed", weights.Embed},
		{"lexical", weights.Lexical},
		{"tag", weights.Tag},
		{"name", weights.Name},
		{"category", weights.Category},
	} {
		if err := validateAdvancedToolFilteringUnitFloat("weights."+field.name, field.value); err != nil {
			return err
		}
	}
	return nil
}

func validateRetrievalStrategyValue(strategy string) error {
	s := strings.TrimSpace(strings.ToLower(strategy))
	if s == "" {
		return nil
	}
	if s == ToolRetrievalStrategyWeighted || s == ToolRetrievalStrategyHybridHistory {
		return nil
	}
	return fmt.Errorf("tools.advanced_filtering.retrieval_strategy must be %q or %q", ToolRetrievalStrategyWeighted, ToolRetrievalStrategyHybridHistory)
}

func validateHybridHistorySubconfig(h *HybridHistoryToolRetrievalConfig) error {
	if h == nil {
		return nil
	}
	for _, field := range []struct {
		name  string
		value *int
	}{
		{"history_horizon", h.HistoryHorizon},
		{"min_history_steps", h.MinHistorySteps},
	} {
		if err := validateAdvancedToolFilteringNonNegativeInt("hybrid_history."+field.name, field.value); err != nil {
			return err
		}
	}
	for _, field := range []struct {
		name  string
		value *float32
	}{
		{"history_confidence_threshold", h.HistoryConfidenceThreshold},
		{"weight_semantic", h.WeightSemantic},
		{"weight_history_transition", h.WeightHistoryTransition},
		{"weight_decision_prior", h.WeightDecisionPrior},
		{"repetition_penalty_strength", h.RepetitionPenaltyStrength},
	} {
		if err := validateAdvancedToolFilteringUnitFloat("hybrid_history."+field.name, field.value); err != nil {
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
