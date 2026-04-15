package config

// SessionAwareSelectionConfig configures stay-versus-switch routing for
// multi-turn sessions using runtime-derived session facts and lookup-table
// priors sourced from router replay.
type SessionAwareSelectionConfig struct {
	// FallbackMethod is used when session context is unavailable or insufficient.
	FallbackMethod string `yaml:"fallback_method,omitempty"`

	// MinTurnsBeforeSwitch suppresses switching early in a conversation.
	MinTurnsBeforeSwitch int `yaml:"min_turns_before_switch,omitempty"`

	// StayBias adds a baseline preference for keeping the current session model.
	StayBias float64 `yaml:"stay_bias,omitempty"`

	// QualityGapMultiplier scales lookup-table quality-gap estimates when
	// evaluating a potential switch to another model.
	QualityGapMultiplier float64 `yaml:"quality_gap_multiplier,omitempty"`

	// HandoffPenaltyWeight scales replay-derived switch penalties between models.
	HandoffPenaltyWeight float64 `yaml:"handoff_penalty_weight,omitempty"`

	// RemainingTurnWeight increases the value of staying on the current model
	// when the conversation is expected to continue for more turns.
	RemainingTurnWeight float64 `yaml:"remaining_turn_weight,omitempty"`
}
