package config

import (
	"fmt"
	"strings"
)

func validateSessionContracts(cfg *RouterConfig) error {
	for i, rule := range cfg.SessionRules {
		if err := validateSessionRule(rule); err != nil {
			return fmt.Errorf("routing.signals.session[%d]: %w", i, err)
		}
	}
	return nil
}

func validateSessionRule(rule SessionRule) error {
	if strings.TrimSpace(rule.Name) == "" {
		return fmt.Errorf("name cannot be empty")
	}
	if !IsSupportedSessionFact(rule.Fact) {
		return fmt.Errorf("fact %q is unsupported", rule.Fact)
	}
	if rule.Predicate == nil {
		return fmt.Errorf("predicate is required")
	}
	if err := validateNumericPredicate(rule.Predicate); err != nil {
		return fmt.Errorf("predicate: %w", err)
	}

	switch NormalizeSessionFact(rule.Fact) {
	case SessionFactQualityGap, SessionFactHandoffPenalty:
		if strings.TrimSpace(rule.CandidateModel) == "" {
			return fmt.Errorf("candidate_model is required for fact %q", rule.Fact)
		}
	}
	return nil
}

func validateSessionAwareAlgorithmConfig(cfg *SessionAwareSelectionConfig) error {
	if cfg == nil {
		return fmt.Errorf("configuration cannot be nil")
	}
	if cfg.MinTurnsBeforeSwitch < 0 {
		return fmt.Errorf("min_turns_before_switch must be >= 0")
	}
	for _, field := range []struct {
		name  string
		value float64
	}{
		{name: "stay_bias", value: cfg.StayBias},
		{name: "quality_gap_multiplier", value: cfg.QualityGapMultiplier},
		{name: "handoff_penalty_weight", value: cfg.HandoffPenaltyWeight},
		{name: "remaining_turn_weight", value: cfg.RemainingTurnWeight},
	} {
		if field.value < 0 {
			return fmt.Errorf("%s must be >= 0", field.name)
		}
	}
	if cfg.FallbackMethod != "" && !IsSupportedDecisionAlgorithmType(cfg.FallbackMethod) {
		return fmt.Errorf("fallback_method %q is unsupported", cfg.FallbackMethod)
	}
	if strings.EqualFold(cfg.FallbackMethod, "session_aware") {
		return fmt.Errorf("fallback_method cannot be session_aware")
	}
	return nil
}

func validateNumericPredicate(predicate *NumericPredicate) error {
	if predicate == nil {
		return fmt.Errorf("cannot be nil")
	}
	if predicate.GT == nil && predicate.GTE == nil && predicate.LT == nil && predicate.LTE == nil {
		return fmt.Errorf("at least one of gt, gte, lt, lte is required")
	}
	if predicate.GT != nil && predicate.GTE != nil {
		return fmt.Errorf("cannot set both gt and gte")
	}
	if predicate.LT != nil && predicate.LTE != nil {
		return fmt.Errorf("cannot set both lt and lte")
	}
	return nil
}
