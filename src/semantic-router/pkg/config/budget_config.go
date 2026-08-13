package config

import "fmt"

// BudgetConfig configures algorithm.budget: a token/cost/wall-time ceiling
// for one Looper execution (issue #2861). All fields are optional; zero
// means unlimited for that dimension.
type BudgetConfig struct {
	MaxPromptTokens     int64   `yaml:"max_prompt_tokens,omitempty"`
	MaxCompletionTokens int64   `yaml:"max_completion_tokens,omitempty"`
	MaxTotalTokens      int64   `yaml:"max_total_tokens,omitempty"`
	MaxEstimatedCost    float64 `yaml:"max_estimated_cost,omitempty"`
	MaxWallTimeMs       int64   `yaml:"max_wall_time_ms,omitempty"`
}

// ValidateBudgetConfig validates algorithm.budget against the algorithm type
// it is attached to. cfg is nil-safe: an unset budget is always valid.
func ValidateBudgetConfig(normalizedType string, cfg *BudgetConfig) error {
	if cfg == nil {
		return nil
	}
	if !IsLooperAlgorithmType(normalizedType) {
		return fmt.Errorf(
			"algorithm.budget is only supported for algorithm types %v (got %q); "+
				"this type does not execute through Looper, so a configured budget would never be enforced",
			SupportedLooperAlgorithmTypes(), normalizedType,
		)
	}
	if err := validateBudgetLimit("max_prompt_tokens", float64(cfg.MaxPromptTokens)); err != nil {
		return err
	}
	if err := validateBudgetLimit("max_completion_tokens", float64(cfg.MaxCompletionTokens)); err != nil {
		return err
	}
	if err := validateBudgetLimit("max_total_tokens", float64(cfg.MaxTotalTokens)); err != nil {
		return err
	}
	if err := validateBudgetLimit("max_estimated_cost", cfg.MaxEstimatedCost); err != nil {
		return err
	}
	if err := validateBudgetLimit("max_wall_time_ms", float64(cfg.MaxWallTimeMs)); err != nil {
		return err
	}
	if cfg.MaxPromptTokens > 0 && cfg.MaxTotalTokens > 0 && cfg.MaxPromptTokens > cfg.MaxTotalTokens {
		return fmt.Errorf("algorithm.budget: max_prompt_tokens (%d) cannot exceed max_total_tokens (%d)", cfg.MaxPromptTokens, cfg.MaxTotalTokens)
	}
	if cfg.MaxCompletionTokens > 0 && cfg.MaxTotalTokens > 0 && cfg.MaxCompletionTokens > cfg.MaxTotalTokens {
		return fmt.Errorf("algorithm.budget: max_completion_tokens (%d) cannot exceed max_total_tokens (%d)", cfg.MaxCompletionTokens, cfg.MaxTotalTokens)
	}
	return nil
}

func validateBudgetLimit(field string, value float64) error {
	if value < 0 {
		return fmt.Errorf("algorithm.budget: %s cannot be negative, got %v", field, value)
	}
	return nil
}
