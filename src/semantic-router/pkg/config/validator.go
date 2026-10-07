package config

import (
	"fmt"
	"strings"
)

func validateRoutingStrategy(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return cfg.Strategy.Validate()
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
