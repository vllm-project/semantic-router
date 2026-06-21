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
	if err := validateBanditLearningConfig(cfg.RouterLearning.Adaptations.Bandit); err != nil {
		return err
	}
	if err := validateEloLearningConfig(cfg.RouterLearning.Adaptations.Elo); err != nil {
		return err
	}
	if err := validatePersonalizationLearningConfig(cfg.RouterLearning.Adaptations.Personalization); err != nil {
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

func validateBanditLearningConfig(cfg BanditLearningConfig) error {
	switch strings.TrimSpace(cfg.Algorithm) {
	case "", RouterLearningBanditAlgorithmLinUCB, RouterLearningBanditAlgorithmLinearThompson:
	default:
		return fmt.Errorf("global.router.learning.adaptations.bandit.algorithm must be %q or %q, got %q",
			RouterLearningBanditAlgorithmLinUCB,
			RouterLearningBanditAlgorithmLinearThompson,
			cfg.Algorithm,
		)
	}
	if err := validateLearningScope(
		"global.router.learning.adaptations.bandit.scope",
		cfg.Scope,
	); err != nil {
		return err
	}
	if err := validateLearningGoals(
		"global.router.learning.adaptations.bandit.goals",
		cfg.Goals,
	); err != nil {
		return err
	}
	return validateBanditLearningTuning(
		"global.router.learning.adaptations.bandit.tuning",
		cfg.Tuning,
	)
}

func validateEloLearningConfig(cfg EloLearningConfig) error {
	if err := validateLearningScope(
		"global.router.learning.adaptations.elo.scope",
		cfg.Scope,
	); err != nil {
		return err
	}
	if cfg.InitialRating != nil && *cfg.InitialRating < 0 {
		return fmt.Errorf("global.router.learning.adaptations.elo.initial_rating must be >= 0, got %v", *cfg.InitialRating)
	}
	if cfg.KFactor != nil && *cfg.KFactor <= 0 {
		return fmt.Errorf("global.router.learning.adaptations.elo.k_factor must be > 0, got %v", *cfg.KFactor)
	}
	return nil
}

func validatePersonalizationLearningConfig(cfg PersonalizationLearningConfig) error {
	return validateLearningScope(
		"global.router.learning.adaptations.personalization.scope",
		cfg.Scope,
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

func validateLearningScope(field string, scope string) error {
	trimmed := strings.TrimSpace(scope)
	switch trimmed {
	case "", RouterLearningScopeDecision, RouterLearningScopeConversation, RouterLearningScopeSession:
		return nil
	default:
		return fmt.Errorf("%s must be %q, %q, or %q, got %q",
			field,
			RouterLearningScopeDecision,
			RouterLearningScopeConversation,
			RouterLearningScopeSession,
			scope,
		)
	}
}

func validateLearningGoals(prefix string, goals map[string]float64) error {
	for goal, weight := range goals {
		trimmed := strings.TrimSpace(goal)
		if trimmed == "" {
			return fmt.Errorf("%s contains an empty goal name", prefix)
		}
		switch trimmed {
		case "quality", "cost", "latency":
		default:
			return fmt.Errorf("%s.%s is not supported; supported goals are quality, cost, and latency", prefix, goal)
		}
		if weight < 0 {
			return fmt.Errorf("%s.%s must be >= 0, got %v", prefix, goal, weight)
		}
	}
	return nil
}

func validateBanditLearningTuning(prefix string, tuning BanditLearningTuning) error {
	if tuning.ExplorationBudget != nil && (*tuning.ExplorationBudget < 0 || *tuning.ExplorationBudget > 1) {
		return fmt.Errorf("%s.exploration_budget must be between 0 and 1, got %v", prefix, *tuning.ExplorationBudget)
	}
	return nil
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
	if cfg.SessionAware != nil {
		if err := validateDecisionSessionAwareAdaptation(decisionName, *cfg.SessionAware); err != nil {
			return err
		}
	}
	if cfg.Bandit != nil {
		if err := validateDecisionBanditAdaptation(decisionName, *cfg.Bandit); err != nil {
			return err
		}
	}
	if cfg.Elo != nil {
		if err := validateDecisionAdaptationMode(decisionName, "elo", cfg.Elo.Mode); err != nil {
			return err
		}
	}

	if cfg.Personalization != nil {
		if err := validateDecisionAdaptationMode(decisionName, "personalization", cfg.Personalization.Mode); err != nil {
			return err
		}
	}

	return nil
}

func validateDecisionSessionAwareAdaptation(decisionName string, cfg DecisionSessionAwareAdaptationConfig) error {
	if err := validateDecisionAdaptationMode(decisionName, "session_aware", cfg.Mode); err != nil {
		return err
	}
	if err := validateSessionAwareScope(
		fmt.Sprintf("decision '%s': adaptations.session_aware.scope", decisionName),
		cfg.Scope,
	); err != nil {
		return err
	}
	return validateSessionAwareTuning(
		fmt.Sprintf("decision '%s': adaptations.session_aware.tuning", decisionName),
		cfg.Tuning,
	)
}

func validateDecisionBanditAdaptation(decisionName string, cfg DecisionBanditAdaptationConfig) error {
	if err := validateDecisionAdaptationMode(decisionName, "bandit", cfg.Mode); err != nil {
		return err
	}
	if err := validateLearningScope(
		fmt.Sprintf("decision '%s': adaptations.bandit.scope", decisionName),
		cfg.Scope,
	); err != nil {
		return err
	}
	if err := validateLearningGoals(
		fmt.Sprintf("decision '%s': adaptations.bandit.goals", decisionName),
		cfg.Goals,
	); err != nil {
		return err
	}
	return validateBanditLearningTuning(
		fmt.Sprintf("decision '%s': adaptations.bandit.tuning", decisionName),
		cfg.Tuning,
	)
}

func validateDecisionAdaptationMode(decisionName string, adaptationName string, mode string) error {
	switch strings.TrimSpace(mode) {
	case "", DecisionAdaptationModeApply, DecisionAdaptationModeObserve, DecisionAdaptationModeBypass:
		return nil
	default:
		return fmt.Errorf("decision '%s': adaptations.%s.mode must be %q, %q, or %q, got %q",
			decisionName,
			adaptationName,
			DecisionAdaptationModeApply,
			DecisionAdaptationModeObserve,
			DecisionAdaptationModeBypass,
			mode,
		)
	}
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

func isEloSelectionConfigConfigured(cfg EloSelectionConfig) bool {
	return cfg.InitialRating != 0 ||
		cfg.KFactor != 0 ||
		cfg.CategoryWeighted ||
		cfg.DecayFactor != 0 ||
		cfg.MinComparisons != 0 ||
		cfg.CostScalingFactor != 0 ||
		cfg.StoragePath != "" ||
		cfg.AutoSaveInterval != ""
}

func anyTrue(values ...bool) bool {
	for _, value := range values {
		if value {
			return true
		}
	}
	return false
}
