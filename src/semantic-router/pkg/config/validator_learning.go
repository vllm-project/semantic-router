package config

import (
	"fmt"
	"strings"
)

func validateRouterLearningConfig(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateRouterLearningAdaptationConfig(cfg.RouterLearning.Adaptation); err != nil {
		return err
	}
	if err := validateRouterLearningProtectionConfig(cfg.RouterLearning.Protection); err != nil {
		return err
	}
	for _, decision := range cfg.Decisions {
		if err := validateDecisionAdaptationsConfig(decision.Name, decision.Adaptations); err != nil {
			return err
		}
	}
	return nil
}

func validateRouterLearningAdaptationConfig(cfg RouterLearningAdaptationConfig) error {
	if err := validateLearningCandidateSet(
		"global.router.learning.adaptation.candidate_set",
		cfg.CandidateSet,
	); err != nil {
		return err
	}
	switch strings.TrimSpace(cfg.Strategy) {
	case "", RouterLearningStrategyRoutingSampling:
	default:
		return fmt.Errorf(
			"global.router.learning.adaptation.strategy must be %q, got %q",
			RouterLearningStrategyRoutingSampling,
			cfg.Strategy,
		)
	}
	return nil
}

func validateRouterLearningProtectionConfig(cfg RouterLearningProtectionConfig) error {
	if err := validateProtectionScope(
		"global.router.learning.protection.scope",
		cfg.Scope,
	); err != nil {
		return err
	}

	if cfg.Identity.Headers.Session != nil && strings.TrimSpace(*cfg.Identity.Headers.Session) == "" {
		return fmt.Errorf("global.router.learning.protection.identity.headers.session cannot be empty")
	}
	if cfg.Identity.Headers.Conversation != nil && strings.TrimSpace(*cfg.Identity.Headers.Conversation) == "" {
		return fmt.Errorf("global.router.learning.protection.identity.headers.conversation cannot be empty")
	}

	return validateProtectionTuning(
		"global.router.learning.protection.tuning",
		cfg.Tuning,
	)
}

func validateProtectionScope(field string, scope string) error {
	switch strings.TrimSpace(scope) {
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

func validateProtectionTuning(prefix string, tuning RouterLearningProtectionTuning) error {
	if err := validateOptionalNonNegativeIntFields([]optionalNonNegativeIntField{
		{prefix + ".idle_timeout_seconds", tuning.IdleTimeoutSeconds},
		{prefix + ".min_turns_before_switch", tuning.MinTurnsBeforeSwitch},
	}); err != nil {
		return err
	}
	if err := validateOptionalNonNegativeFloatFields([]optionalNonNegativeFloatField{
		{prefix + ".switch_margin", tuning.SwitchMargin},
		{prefix + ".stability_weight", tuning.StabilityWeight},
	}); err != nil {
		return err
	}
	return nil
}

func validateDecisionAdaptationsConfig(decisionName string, cfg DecisionAdaptationsConfig) error {
	if err := validateDecisionAdaptationMode(decisionName, "adaptations", cfg.Mode); err != nil {
		return err
	}
	decisionMode := cfg.EffectiveMode()
	if cfg.Adaptation != nil {
		if err := validateDecisionAdaptationMode(decisionName, "adaptations.adaptation", cfg.Adaptation.Mode); err != nil {
			return err
		}
		if err := validateDecisionComponentModeBoundary(
			decisionName,
			"adaptations.adaptation",
			decisionMode,
			cfg.Adaptation.Mode,
		); err != nil {
			return err
		}
		if err := validateLearningCandidateSet(
			fmt.Sprintf("decision '%s': adaptations.adaptation.candidate_set", decisionName),
			cfg.Adaptation.CandidateSet,
		); err != nil {
			return err
		}
	}
	if cfg.Protection != nil {
		if err := validateDecisionAdaptationMode(decisionName, "adaptations.protection", cfg.Protection.Mode); err != nil {
			return err
		}
		if err := validateDecisionComponentModeBoundary(
			decisionName,
			"adaptations.protection",
			decisionMode,
			cfg.Protection.Mode,
		); err != nil {
			return err
		}
		if err := validateOptionalNonNegativeFloatFields([]optionalNonNegativeFloatField{
			{fmt.Sprintf("decision '%s': adaptations.protection.stability_weight", decisionName), cfg.Protection.StabilityWeight},
			{fmt.Sprintf("decision '%s': adaptations.protection.switch_margin", decisionName), cfg.Protection.SwitchMargin},
		}); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionComponentModeBoundary(
	decisionName string,
	field string,
	decisionMode string,
	componentMode string,
) error {
	componentMode = strings.TrimSpace(componentMode)
	if componentMode == "" {
		return nil
	}
	switch decisionMode {
	case DecisionAdaptationModeBypass:
		if componentMode != DecisionAdaptationModeBypass {
			return fmt.Errorf(
				"decision '%s': %s.mode cannot be %q when adaptations.mode is %q",
				decisionName,
				field,
				componentMode,
				DecisionAdaptationModeBypass,
			)
		}
	case DecisionAdaptationModeObserve:
		if componentMode == DecisionAdaptationModeApply {
			return fmt.Errorf(
				"decision '%s': %s.mode cannot be %q when adaptations.mode is %q",
				decisionName,
				field,
				DecisionAdaptationModeApply,
				DecisionAdaptationModeObserve,
			)
		}
	}
	return nil
}

func validateLearningCandidateSet(field string, candidateSet string) error {
	switch strings.TrimSpace(candidateSet) {
	case "", RouterLearningCandidateSetDecision, RouterLearningCandidateSetTier, RouterLearningCandidateSetGlobal:
		return nil
	default:
		return fmt.Errorf(
			"%s must be %q, %q, or %q, got %q",
			field,
			RouterLearningCandidateSetDecision,
			RouterLearningCandidateSetTier,
			RouterLearningCandidateSetGlobal,
			candidateSet,
		)
	}
}

func validateDecisionAdaptationMode(decisionName string, field string, mode string) error {
	switch strings.TrimSpace(mode) {
	case "", DecisionAdaptationModeApply, DecisionAdaptationModeObserve, DecisionAdaptationModeBypass:
		return nil
	default:
		return fmt.Errorf("decision '%s': %s.mode must be %q, %q, or %q, got %q",
			decisionName,
			field,
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
