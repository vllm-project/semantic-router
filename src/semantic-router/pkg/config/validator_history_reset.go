package config

import (
	"fmt"
	"strings"
)

// validateHistoryResetPlugin enforces the reset action's configuration
// contract. Structural checks apply to every policy; trigger resolution and
// runtime capability checks apply only once a policy is enabled, so a disabled
// example can reference a signal family that is not registered yet.
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

// validateHistoryResetTrigger checks the trigger's own shape. Whether the
// referenced family exists is an enablement concern handled separately.
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
// executed safely. An enabled policy needs an explicit trigger and acceptance
// threshold, and the topic-continuity signal family must be registered; until
// #3342 registers it, no configuration can enable live removal. Disabled
// policies are accepted so the contract, examples, and round trips are usable
// before that dependency lands.
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
	if !HistoryResetTriggerFamilyRegistered() {
		return fmt.Errorf(
			"%s: %s: no %q signal family is registered, so an enabled history_reset policy cannot resolve a trigger",
			scope,
			HistoryResetTriggerUnavailable,
			HistoryResetTriggerSignalType,
		)
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
	if compression == nil || compression.Recovery == nil || !compression.Recovery.Enabled ||
		!reset.RequiresRecovery() {
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
// against the signals the selected recipe actually declares, using the same
// rules as a decision rule leaf. Registering the topic-continuity family
// globally is not enough: the named signal must exist in this scope and belong
// to that family, so a valid name pointing at an unrelated signal cannot
// authorize removal.
func validateHistoryResetTriggerReferences(cfg *RouterConfig) error {
	if cfg == nil || cfg.RoutingScope == "" {
		// Unscoped configuration cannot resolve recipe-local references; the
		// per-recipe validation pass performs this check.
		return nil
	}
	declared := projectionDeclaredSignals(cfg)
	for index := range cfg.Decisions {
		decision := &cfg.Decisions[index]
		reset := decision.GetHistoryResetConfig()
		if !reset.IsEnabled() || reset.Trigger == nil {
			continue
		}
		name := strings.TrimSpace(reset.Trigger.Signal)
		if projectionInputDeclared(declared, HistoryResetTriggerSignalType, name) {
			continue
		}
		return fmt.Errorf(
			"decision %q: history_reset trigger.signal %s(%q) is not declared in this recipe",
			decision.Name,
			HistoryResetTriggerSignalType,
			name,
		)
	}
	return nil
}
