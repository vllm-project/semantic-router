package config

import "fmt"

// validateSessionTokenBudgetContracts is the RouterConfig-scoped entry point
// registered in the contract-validator list.
func validateSessionTokenBudgetContracts(cfg *RouterConfig) error {
	return validateSessionTokenBudget(cfg.SessionTokenBudget)
}

// validateSessionTokenBudget validates the session_token_budget config block.
//
// An empty/disabled block is always valid (enforcement is opt-in). When set:
// budget_tokens must be >= 0, every explicit threshold must be > 0, and the
// explicit thresholds must be non-decreasing in stage order
// (shape_tools <= compress <= downgrade <= terminate) so the graduated ladder
// escalates correctly. Zero thresholds are left to the sessionbudget defaults
// and are skipped by the ascending check.
func validateSessionTokenBudget(cfg SessionTokenBudgetConfig) error {
	if cfg.BudgetTokens < 0 {
		return fmt.Errorf("session_token_budget.budget_tokens must be >= 0, got %d", cfg.BudgetTokens)
	}
	if cfg.Enabled && cfg.BudgetTokens == 0 {
		return fmt.Errorf(
			"session_token_budget.budget_tokens must be > 0 when session_token_budget.enabled is true " +
				"(enforcement would otherwise silently no-op)",
		)
	}

	t := cfg.Thresholds
	stages := []struct {
		name  string
		value float64
	}{
		{"shape_tools", t.ShapeTools},
		{"compress", t.Compress},
		{"downgrade", t.Downgrade},
		{"terminate", t.Terminate},
	}
	for _, s := range stages {
		if s.value < 0 {
			return fmt.Errorf("session_token_budget.thresholds.%s must not be negative (0 = use default), got %v", s.name, s.value)
		}
	}

	// Explicit (non-zero) thresholds must be ascending in stage order. Equal
	// adjacent thresholds are allowed and intentionally collapse the lower stage
	// (the higher-priority stage wins in sessionbudget.Evaluate).
	prevName, prev := "", 0.0
	for _, s := range stages {
		if s.value == 0 {
			continue
		}
		if prev != 0 && s.value < prev {
			return fmt.Errorf(
				"session_token_budget.thresholds must be ascending: %s (%v) is below %s (%v)",
				s.name, s.value, prevName, prev,
			)
		}
		prevName, prev = s.name, s.value
	}
	return nil
}
