package config

import (
	"fmt"
	"strings"
)

// validateHistoryResetPlugin enforces the reset action's configuration
// contract. Structural checks apply to every policy; the trigger requirements
// apply only once a policy is enabled, so a disabled example stays valid
// without a complete trigger.
func validateHistoryResetPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *HistoryResetPluginConfig,
) error {
	scope := fmt.Sprintf("decision %q plugins[%d] (%s)", decisionName, index, pluginType)
	checks := []func() error{
		func() error { return validateHistoryResetScope(typed, scope) },
		func() error { return validateHistoryResetFailureMode(typed, scope) },
		func() error { return validateHistoryResetLimits(typed, scope) },
		func() error { return validateHistoryResetTrigger(typed, scope) },
		func() error { return validateHistoryResetRecovery(typed, scope) },
		func() error { return validateHistoryResetEnablement(typed, scope) },
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

// ValidateHistoryResetPluginConfig validates a standalone policy payload.
func ValidateHistoryResetPluginConfig(typed *HistoryResetPluginConfig) error {
	if typed == nil {
		return fmt.Errorf("history_reset configuration is required")
	}
	return validateHistoryResetPlugin("preview", 0, DecisionPluginHistoryReset, typed)
}

func validateHistoryResetScope(typed *HistoryResetPluginConfig, scope string) error {
	if typed.EffectiveScope() != HistoryResetScopeEligibleHistory {
		return fmt.Errorf("%s: scope must be %s", scope, HistoryResetScopeEligibleHistory)
	}
	return nil
}

func validateHistoryResetFailureMode(typed *HistoryResetPluginConfig, scope string) error {
	switch typed.EffectiveFailureMode() {
	case HistoryResetFailureOpen, HistoryResetFailureClosed:
		return nil
	default:
		return fmt.Errorf("%s: failure_mode must be %s or %s",
			scope, HistoryResetFailureOpen, HistoryResetFailureClosed)
	}
}

func validateHistoryResetLimits(typed *HistoryResetPluginConfig, scope string) error {
	if typed.Limits == nil {
		return nil
	}
	bounds := []struct {
		field string
		value int
		max   int
	}{
		{"limits.max_history_turns", typed.Limits.MaxHistoryTurns, maxHistoryResetHistoryTurns},
		{"limits.max_history_bytes", typed.Limits.MaxHistoryBytes, maxHistoryResetHistoryBytes},
		{"limits.timeout_ms", typed.Limits.TimeoutMs, maxHistoryResetTimeoutMs},
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
	return nil
}

// validateHistoryResetTrigger checks the trigger's own shape. Whether the named
// rule is declared is checked per recipe by validateHistoryResetTriggerReferences.
func validateHistoryResetTrigger(typed *HistoryResetPluginConfig, scope string) error {
	if typed.Trigger == nil {
		return nil
	}
	if strings.TrimSpace(typed.Trigger.Signal) == "" {
		return fmt.Errorf("%s: trigger.signal is required when trigger is configured", scope)
	}
	confidence, present := typed.EffectiveMinConfidence()
	if present && (confidence <= 0 || confidence > 1) {
		return fmt.Errorf("%s: trigger.min_confidence must be greater than 0 and at most 1", scope)
	}
	return nil
}

func validateHistoryResetRecovery(typed *HistoryResetPluginConfig, scope string) error {
	if !typed.RequiresRecovery() {
		return nil
	}
	switch strings.TrimSpace(typed.Recovery.Store) {
	case "redis", "valkey", "response_cache":
	default:
		return fmt.Errorf("%s: recovery.store must be redis, valkey, or response_cache", scope)
	}
	if typed.Recovery.TTLSeconds < 0 ||
		typed.Recovery.MaxBytesPerRequest < 0 ||
		typed.Recovery.MaxTotalBytes < 0 ||
		typed.Recovery.MaxRetrievals < 0 {
		return fmt.Errorf("%s: recovery limits cannot be negative", scope)
	}
	return nil
}

// validateHistoryResetEnablement rejects an enabled policy that cannot be
// executed safely: it needs an explicit trigger, an acceptance threshold, and
// the result contracts it trusts. Disabled policies are accepted without
// those, so the contract and its examples stay usable.
func validateHistoryResetEnablement(typed *HistoryResetPluginConfig, scope string) error {
	if !typed.IsEnabled() {
		return nil
	}
	if typed.Trigger == nil {
		return fmt.Errorf("%s: trigger is required when enabled is true", scope)
	}
	if _, present := typed.EffectiveMinConfidence(); !present {
		return fmt.Errorf("%s: trigger.min_confidence is required when enabled is true", scope)
	}
	if len(typed.Trigger.AcceptedVersions) == 0 {
		return fmt.Errorf(
			"%s: trigger.accepted_versions is required when enabled is true", scope)
	}
	for index, version := range typed.Trigger.AcceptedVersions {
		if strings.TrimSpace(version) == "" {
			return fmt.Errorf(
				"%s: trigger.accepted_versions[%d] cannot be empty", scope, index)
		}
	}
	return nil
}

// validateDecisionContextRecoveryAgreement rejects a decision whose context
// actions ask for different recovery stores. Removal and compression share one
// request-level store, budget, and retrieval tool, so the router cannot honour
// two stores; silently preferring one plugin's setting would leave the other's
// content unreachable.
func validateDecisionContextRecoveryAgreement(decision *Decision) error {
	compression := decision.GetContextCompressionConfig()
	reset := decision.GetHistoryResetConfig()
	// Only actions that are enabled and enable recovery share a store; a
	// disabled action's recovery block is validated on its own and constrains
	// nothing else.
	if compression == nil || !compression.Enabled || compression.Recovery == nil || !compression.Recovery.Enabled ||
		!reset.IsEnabled() || !reset.RequiresRecovery() {
		return nil
	}
	compressionStore := strings.TrimSpace(compression.Recovery.Store)
	resetStore := strings.TrimSpace(reset.Recovery.Store)
	if !strings.EqualFold(compressionStore, resetStore) {
		return fmt.Errorf(
			"decision %q: context_compression recovery.store %q and history_reset recovery.store %q must match",
			decision.Name,
			compressionStore,
			resetStore,
		)
	}
	return nil
}

// validateHistoryResetTriggerReferences resolves each enabled policy's trigger
// against the topic-continuity rules the selected recipe declares. Only a
// declared rule is evaluated for a request, so a trigger naming anything else,
// including a signal of another family with the same name, could never supply
// evidence and is rejected rather than left to preserve every request.
func validateHistoryResetTriggerReferences(cfg *RouterConfig) error {
	if cfg == nil || cfg.RoutingScope == "" {
		// Unscoped configuration cannot resolve recipe-local references; the
		// per-recipe validation pass performs this check.
		return nil
	}
	for index := range cfg.Decisions {
		decision := &cfg.Decisions[index]
		reset := decision.GetHistoryResetConfig()
		if !reset.IsEnabled() || reset.Trigger == nil {
			continue
		}
		name := strings.TrimSpace(reset.Trigger.Signal)
		if _, declared := cfg.TopicContinuityRule(name); declared {
			continue
		}
		return fmt.Errorf(
			"decision %q: history_reset trigger.signal %s(%q) is not declared in this recipe",
			decision.Name,
			SignalTypeTopicContinuity,
			name,
		)
	}
	return nil
}
