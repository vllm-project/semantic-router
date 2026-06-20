package config

import (
	"fmt"
	"strings"
)

func validateRouterLearningConfig(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateSessionAwareLearningConfig(cfg.RouterLearning.Adaptations.SessionAware); err != nil {
		return err
	}
	for _, decision := range cfg.Decisions {
		if err := validateDecisionAdaptationsConfig(decision.Name, decision.Adaptations); err != nil {
			return err
		}
	}
	return nil
}

func validateSessionAwareLearningConfig(cfg SessionAwareLearningConfig) error {
	if err := validateSessionAwareScope(
		"global.router.learning.adaptations.session_aware.scope",
		cfg.Scope,
	); err != nil {
		return err
	}

	for key, value := range cfg.Identity.Headers {
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf("global.router.learning.adaptations.session_aware.identity.headers contains an empty key")
		}
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("global.router.learning.adaptations.session_aware.identity.headers.%s cannot be empty", key)
		}
	}

	return validateSessionAwareTuning(
		"global.router.learning.adaptations.session_aware.tuning",
		cfg.Tuning,
	)
}

func validateSessionAwareScope(field string, scope string) error {
	trimmed := strings.TrimSpace(scope)
	switch trimmed {
	case "", RouterLearningScopeConversation, RouterLearningScopeSession:
		return nil
	default:
		return fmt.Errorf("%s must be %q or %q, got %q",
			field,
			RouterLearningScopeConversation,
			RouterLearningScopeSession,
			scope,
		)
	}
}

func validateSessionAwareTuning(prefix string, tuning SessionAwareLearningTuning) error {
	if err := validateOptionalNonNegativeIntFields([]optionalNonNegativeIntField{
		{prefix + ".idle_timeout_seconds", tuning.IdleTimeoutSeconds},
		{prefix + ".min_turns_before_switch", tuning.MinTurnsBeforeSwitch},
	}); err != nil {
		return err
	}
	if err := validateOptionalNonNegativeFloatFields([]optionalNonNegativeFloatField{
		{prefix + ".switch_margin", tuning.SwitchMargin},
		{prefix + ".cache_weight", tuning.CacheWeight},
		{prefix + ".handoff_penalty", tuning.HandoffPenalty},
		{prefix + ".handoff_penalty_weight", tuning.HandoffPenaltyWeight},
		{prefix + ".switch_history_weight", tuning.SwitchHistoryWeight},
	}); err != nil {
		return err
	}
	if tuning.MaxCacheCostMultiplier != nil && *tuning.MaxCacheCostMultiplier < 1 {
		return fmt.Errorf("%s.max_cache_cost_multiplier must be >= 1, got %v", prefix, *tuning.MaxCacheCostMultiplier)
	}
	return nil
}

func validateDecisionAdaptationsConfig(decisionName string, cfg DecisionAdaptationsConfig) error {
	if cfg.SessionAware == nil {
		return nil
	}
	switch strings.TrimSpace(cfg.SessionAware.Mode) {
	case "", DecisionAdaptationModeApply, DecisionAdaptationModeObserve, DecisionAdaptationModeBypass:
	default:
		return fmt.Errorf("decision '%s': adaptations.session_aware.mode must be %q, %q, or %q, got %q",
			decisionName,
			DecisionAdaptationModeApply,
			DecisionAdaptationModeObserve,
			DecisionAdaptationModeBypass,
			cfg.SessionAware.Mode,
		)
	}
	if err := validateSessionAwareScope(
		fmt.Sprintf("decision '%s': adaptations.session_aware.scope", decisionName),
		cfg.SessionAware.Scope,
	); err != nil {
		return err
	}
	if err := validateSessionAwareTuning(
		fmt.Sprintf("decision '%s': adaptations.session_aware.tuning", decisionName),
		cfg.SessionAware.Tuning,
	); err != nil {
		return err
	}
	return nil
}

func isSessionAwareSelectionConfigConfigured(cfg SessionAwareSelectionConfig) bool {
	return cfg.BaseMethod != "" || anyTrue(
		cfg.IdleTimeoutSeconds != nil,
		cfg.MinTurnsBeforeSwitch != nil,
		cfg.SwitchMargin != nil,
		cfg.StayBias != nil,
		cfg.ToolLoopHardLock != nil,
		cfg.ContextPortabilityHardLock != nil,
		cfg.DecisionDriftReset != nil,
		cfg.ToolLoopStayBias != nil,
		cfg.PrefixCacheWeight != nil,
		cfg.HandoffPenaltyWeight != nil,
		cfg.DefaultHandoffPenalty != nil,
		cfg.QualityGapMultiplier != nil,
		cfg.MaxCacheCostMultiplier != nil,
		cfg.SwitchHistoryWeight != nil,
		cfg.RemainingTurnPriorWeight != nil,
		cfg.RemainingTurnPriorHorizon != nil,
		cfg.MinRemainingTurnPriorSamples != nil,
	)
}

func isModelSwitchGateConfigured(cfg ModelSwitchGateConfig) bool {
	return cfg.Enabled ||
		cfg.Mode != "" ||
		cfg.MinSwitchAdvantage != 0 ||
		cfg.DefaultHandoffPenalty != 0 ||
		cfg.CacheWarmthWeight != 0
}

func isLookupTableConfigConfigured(cfg LookupTableConfig) bool {
	return cfg.Enabled ||
		cfg.StoragePath != "" ||
		cfg.AutoSaveInterval != "" ||
		cfg.PopulateFromReplay ||
		cfg.PopulateInterval != "" ||
		len(cfg.QualityGaps) > 0 ||
		len(cfg.HandoffPenalties) > 0 ||
		len(cfg.RemainingTurnPriors) > 0
}

func anyTrue(values ...bool) bool {
	for _, value := range values {
		if value {
			return true
		}
	}
	return false
}
