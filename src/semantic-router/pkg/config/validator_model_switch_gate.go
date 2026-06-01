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
	intFields := []optionalNonNegativeIntField{
		{"session_aware.idle_timeout_seconds", cfg.IdleTimeoutSeconds},
		{"session_aware.min_turns_before_switch", cfg.MinTurnsBeforeSwitch},
		{"session_aware.min_remaining_turn_prior_samples", cfg.MinRemainingTurnPriorSamples},
	}
	if err := validateOptionalNonNegativeIntFields(intFields); err != nil {
		return err
	}
	if cfg.RemainingTurnPriorHorizon != nil && *cfg.RemainingTurnPriorHorizon <= 0 {
		return fmt.Errorf(
			"session_aware.remaining_turn_prior_horizon must be > 0, got %d",
			*cfg.RemainingTurnPriorHorizon,
		)
	}

	floatFields := []optionalNonNegativeFloatField{
		{"session_aware.switch_margin", cfg.SwitchMargin},
		{"session_aware.stay_bias", cfg.StayBias},
		{"session_aware.tool_loop_stay_bias", cfg.ToolLoopStayBias},
		{"session_aware.prefix_cache_weight", cfg.PrefixCacheWeight},
		{"session_aware.handoff_penalty_weight", cfg.HandoffPenaltyWeight},
		{"session_aware.default_handoff_penalty", cfg.DefaultHandoffPenalty},
		{"session_aware.quality_gap_multiplier", cfg.QualityGapMultiplier},
		{"session_aware.switch_history_weight", cfg.SwitchHistoryWeight},
		{"session_aware.remaining_turn_prior_weight", cfg.RemainingTurnPriorWeight},
	}
	if err := validateOptionalNonNegativeFloatFields(floatFields); err != nil {
		return err
	}
	if cfg.MaxCacheCostMultiplier != nil && *cfg.MaxCacheCostMultiplier < 1 {
		return fmt.Errorf(
			"session_aware.max_cache_cost_multiplier must be >= 1, got %v",
			*cfg.MaxCacheCostMultiplier,
		)
	}
	return nil
}

type optionalNonNegativeIntField struct {
	name  string
	value *int
}

func validateOptionalNonNegativeIntFields(fields []optionalNonNegativeIntField) error {
	for _, field := range fields {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("%s must be >= 0, got %d", field.name, *field.value)
		}
	}
	return nil
}

type optionalNonNegativeFloatField struct {
	name  string
	value *float64
}

func validateOptionalNonNegativeFloatFields(fields []optionalNonNegativeFloatField) error {
	for _, field := range fields {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("%s must be >= 0, got %v", field.name, *field.value)
		}
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
