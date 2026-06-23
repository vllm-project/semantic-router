package config

// SessionTokenBudgetConfig configures runtime per-session token-budget
// enforcement (WRP Opportunity 5). It is opt-in and tri-state: enforcement is a
// complete no-op unless Enabled is true AND BudgetTokens > 0.
//
// When active, the router compares each active session's cumulative token usage
// (prompt + completion, across turns) against BudgetTokens and escalates through
// a graduated response ladder rather than a binary deny. See pkg/sessionbudget
// for the stage model.
type SessionTokenBudgetConfig struct {
	// Enabled activates session token-budget evaluation after session-transition
	// fields are populated. Defaults to false (disabled).
	Enabled bool `yaml:"enabled,omitempty"`

	// BudgetTokens is the static per-session token ceiling (the "expected
	// budget"). Must be > 0 for enforcement to run. The percentile / fingerprint
	// budget prior from the vision paper is deferred to a follow-up.
	BudgetTokens int64 `yaml:"budget_tokens,omitempty"`

	// Thresholds set the over-budget ratio at which each ladder stage fires.
	// Zero fields fall back to the sessionbudget defaults.
	Thresholds SessionTokenBudgetThresholds `yaml:"thresholds,omitempty"`
}

// SessionTokenBudgetThresholds are ascending multipliers of BudgetTokens at
// which each graduated-response stage activates. A zero field means "use the
// sessionbudget default" for that stage.
type SessionTokenBudgetThresholds struct {
	ShapeTools float64 `yaml:"shape_tools,omitempty"`
	Compress   float64 `yaml:"compress,omitempty"`
	Downgrade  float64 `yaml:"downgrade,omitempty"`
	Terminate  float64 `yaml:"terminate,omitempty"`
}
