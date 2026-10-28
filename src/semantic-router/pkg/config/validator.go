package config

import (
	"fmt"
	"strings"
)

// Public values of global.router.protocol.lossy_features.
const (
	ProtocolLossyReject              = "reject"
	ProtocolLossyAllowWithDiagnostic = "allow_with_diagnostic"
)

func validateRoutingStrategy(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return cfg.Strategy.Validate()
}

func validateProtocolTranslationPolicy(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	switch cfg.ProtocolLossyFeatures {
	case "", ProtocolLossyReject, ProtocolLossyAllowWithDiagnostic:
		return nil
	default:
		return fmt.Errorf("global.router.protocol.lossy_features must be %q or %q, got %q",
			ProtocolLossyReject, ProtocolLossyAllowWithDiagnostic, cfg.ProtocolLossyFeatures)
	}
}

// prompt_guard backend validation lives in validator_prompt_guard.go.

func validateModelSelectionConfig(cfg *RouterConfig) error {
	if err := validatePromptGuardStaticContracts(cfg); err != nil {
		return err
	}
	if isSessionAwareSelectionConfigConfigured(cfg.ModelSelection.SessionAware) {
		return fmt.Errorf("global.router.model_selection.session_aware is no longer supported; use global.router.learning.protection")
	}
	if isModelSwitchGateConfigured(cfg.ModelSelection.ModelSwitchGate) {
		return fmt.Errorf("global.router.model_selection.model_switch_gate is no longer supported; use global.router.learning.protection.tuning")
	}
	if isLookupTableConfigConfigured(cfg.ModelSelection.LookupTables) {
		return fmt.Errorf("global.router.model_selection.lookup_tables has moved to future Router Learning experience; remove lookup_tables from public config")
	}
	if method := strings.TrimSpace(cfg.ModelSelection.Method); removedGlobalLearningSelector(method) {
		return fmt.Errorf("global.router.model_selection.%s is no longer supported; use global.router.learning.adaptation", method)
	}
	if isEloSelectionConfigConfigured(cfg.ModelSelection.Elo) {
		return fmt.Errorf("global.router.model_selection.elo is no longer supported; use global.router.learning.adaptation")
	}
	return nil
}
