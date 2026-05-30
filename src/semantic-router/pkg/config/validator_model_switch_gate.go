package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	modelSwitchGateModeShadow  = "shadow"
	modelSwitchGateModeEnforce = "enforce"
)

// validateModelSwitchGate validates the model_switch_gate config block.
// Empty mode is accepted (runtime defaults to shadow); typos like "enforced"
// are rejected so a misconfigured gate fails at boot rather than silently
// degrading to shadow. Negative weights/penalties are rejected because they
// would invert the gate's stay-vs-switch math.
func validateModelSwitchGate(cfg ModelSwitchGateConfig) error {
	switch cfg.Mode {
	case "", modelSwitchGateModeShadow, modelSwitchGateModeEnforce:
	default:
		return fmt.Errorf(
			"model_switch_gate.mode must be %q or %q (empty defaults to %q), got %q",
			modelSwitchGateModeShadow, modelSwitchGateModeEnforce, modelSwitchGateModeShadow, cfg.Mode,
		)
	}
	if cfg.MinSwitchAdvantage < 0 {
		return fmt.Errorf("model_switch_gate.min_switch_advantage must be >= 0, got %v", cfg.MinSwitchAdvantage)
	}
	if cfg.DefaultHandoffPenalty < 0 {
		return fmt.Errorf("model_switch_gate.default_handoff_penalty must be >= 0, got %v", cfg.DefaultHandoffPenalty)
	}
	if cfg.CacheWarmthWeight < 0 {
		return fmt.Errorf("model_switch_gate.cache_warmth_weight must be >= 0, got %v", cfg.CacheWarmthWeight)
	}
	return nil
}

func validateSessionAwareSelectionConfig(cfg SessionAwareSelectionConfig) error {
	if err := validateSessionAwareTimingConfig(cfg); err != nil {
		return err
	}
	if err := validateSessionAwareCostConfig(cfg); err != nil {
		return err
	}
	return validateSessionAwareContextConfig(cfg)
}

func validateSessionAwareTimingConfig(cfg SessionAwareSelectionConfig) error {
	if cfg.IdleTimeoutSeconds < 0 {
		return fmt.Errorf("session_aware.idle_timeout_seconds must be >= 0, got %d", cfg.IdleTimeoutSeconds)
	}
	if cfg.MinTurnsBeforeSwitch < 0 {
		return fmt.Errorf("session_aware.min_turns_before_switch must be >= 0, got %d", cfg.MinTurnsBeforeSwitch)
	}
	if cfg.SwitchMargin < 0 {
		return fmt.Errorf("session_aware.switch_margin must be >= 0, got %v", cfg.SwitchMargin)
	}
	if cfg.StayBias < 0 {
		return fmt.Errorf("session_aware.stay_bias must be >= 0, got %v", cfg.StayBias)
	}
	if cfg.ToolLoopStayBias < 0 {
		return fmt.Errorf("session_aware.tool_loop_stay_bias must be >= 0, got %v", cfg.ToolLoopStayBias)
	}
	return nil
}

func validateSessionAwareCostConfig(cfg SessionAwareSelectionConfig) error {
	if cfg.PrefixCacheWeight < 0 {
		return fmt.Errorf("session_aware.prefix_cache_weight must be >= 0, got %v", cfg.PrefixCacheWeight)
	}
	if cfg.HandoffPenaltyWeight < 0 {
		return fmt.Errorf("session_aware.handoff_penalty_weight must be >= 0, got %v", cfg.HandoffPenaltyWeight)
	}
	if cfg.DefaultHandoffPenalty < 0 {
		return fmt.Errorf("session_aware.default_handoff_penalty must be >= 0, got %v", cfg.DefaultHandoffPenalty)
	}
	if cfg.QualityGapMultiplier < 0 {
		return fmt.Errorf("session_aware.quality_gap_multiplier must be >= 0, got %v", cfg.QualityGapMultiplier)
	}
	if cfg.MaxCacheCostMultiplier < 0 {
		return fmt.Errorf("session_aware.max_cache_cost_multiplier must be >= 0, got %v", cfg.MaxCacheCostMultiplier)
	}
	if cfg.SwitchHistoryWeight < 0 {
		return fmt.Errorf("session_aware.switch_history_weight must be >= 0, got %v", cfg.SwitchHistoryWeight)
	}
	return nil
}

func validateSessionAwareContextConfig(cfg SessionAwareSelectionConfig) error {
	if cfg.ContextPortabilityWeight < 0 {
		return fmt.Errorf("session_aware.context_portability_weight must be >= 0, got %v", cfg.ContextPortabilityWeight)
	}
	if cfg.MinSwitchContextPortability < 0 || cfg.MinSwitchContextPortability > 1 {
		return fmt.Errorf("session_aware.min_switch_context_portability must be between 0 and 1, got %v", cfg.MinSwitchContextPortability)
	}
	if cfg.ProviderStatePenalty < 0 {
		return fmt.Errorf("session_aware.provider_state_penalty must be >= 0, got %v", cfg.ProviderStatePenalty)
	}
	return nil
}

// warnModelSwitchGateEnforceWithoutCostSignals emits a startup warning when
// enforce is configured but neither lookup_tables nor a default handoff penalty
// is in place. In that situation the gate's stay cost has no real evidence and
// would be unjustifiably optimistic — the warning surfaces this without
// blocking boot, since "no handoff cost" is a legitimate config in some setups.
func warnModelSwitchGateEnforceWithoutCostSignals(cfg ModelSelectionConfig) {
	gate := cfg.ModelSwitchGate
	if !gate.Enabled || gate.Mode != modelSwitchGateModeEnforce {
		return
	}
	if cfg.LookupTables.Enabled || gate.DefaultHandoffPenalty > 0 {
		return
	}
	logging.Warnf(
		"model_switch_gate: enforce mode configured without lookup_tables.enabled or a non-zero default_handoff_penalty. " +
			"The gate will treat handoff cost as zero and may switch models too eagerly. " +
			"Either enable lookup_tables or set a non-zero default_handoff_penalty.",
	)
}
