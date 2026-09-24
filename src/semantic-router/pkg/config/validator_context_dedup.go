package config

import "fmt"

// validateContextDedupPlugin enforces the deduplication action's configuration
// contract. The action has no external dependency, so an enabled policy needs
// nothing beyond a well-formed shape.
func validateContextDedupPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ContextDedupPluginConfig,
) error {
	scope := fmt.Sprintf("decision %q plugins[%d] (%s)", decisionName, index, pluginType)
	checks := []func() error{
		func() error { return validateContextDedupNormalization(typed, scope) },
		func() error { return validateContextDedupFailureMode(typed, scope) },
		func() error { return validateContextDedupLimits(typed, scope) },
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

// ValidateContextDedupPluginConfig validates a standalone policy payload.
func ValidateContextDedupPluginConfig(typed *ContextDedupPluginConfig) error {
	if typed == nil {
		return fmt.Errorf("context_dedup configuration is required")
	}
	return validateContextDedupPlugin("preview", 0, DecisionPluginContextDedup, typed)
}

func validateContextDedupNormalization(typed *ContextDedupPluginConfig, scope string) error {
	switch typed.EffectiveNormalization() {
	case ContextDedupNormalizationExact, ContextDedupNormalizationWhitespace:
		return nil
	default:
		return fmt.Errorf("%s: normalization must be %s or %s",
			scope, ContextDedupNormalizationExact, ContextDedupNormalizationWhitespace)
	}
}

func validateContextDedupFailureMode(typed *ContextDedupPluginConfig, scope string) error {
	switch typed.EffectiveFailureMode() {
	case ContextDedupFailureOpen, ContextDedupFailureClosed:
		return nil
	default:
		return fmt.Errorf("%s: failure_mode must be %s or %s",
			scope, ContextDedupFailureOpen, ContextDedupFailureClosed)
	}
}

func validateContextDedupLimits(typed *ContextDedupPluginConfig, scope string) error {
	if typed.Limits == nil {
		return nil
	}
	bounds := []struct {
		field string
		value int
		max   int
	}{
		{"limits.max_history_turns", typed.Limits.MaxHistoryTurns, maxContextDedupHistoryTurns},
		{"limits.max_history_bytes", typed.Limits.MaxHistoryBytes, maxContextDedupHistoryBytes},
		{"limits.max_segment_turns", typed.Limits.MaxSegmentTurns, maxContextDedupSegmentTurns},
		{"limits.timeout_ms", typed.Limits.TimeoutMs, maxContextDedupTimeoutMs},
	}
	for _, bound := range bounds {
		if bound.value == 0 {
			continue
		}
		if bound.value < 0 {
			return fmt.Errorf("%s: %s must be positive", scope, bound.field)
		}
		if bound.value > bound.max {
			return fmt.Errorf("%s: %s cannot exceed %d", scope, bound.field, bound.max)
		}
	}
	// Only an explicit segment bound is checked against the history bound;
	// an omitted one is derived from it.
	limits := typed.EffectiveLimits()
	if typed.Limits.MaxSegmentTurns > 0 && limits.MaxSegmentTurns > limits.MaxHistoryTurns {
		return fmt.Errorf("%s: limits.max_segment_turns cannot exceed limits.max_history_turns (%d)",
			scope, limits.MaxHistoryTurns)
	}
	return nil
}
