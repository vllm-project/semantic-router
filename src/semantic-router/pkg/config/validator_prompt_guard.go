package config

import "fmt"

// validPromptGuardProtocols is the set of recognized PromptGuardConfig.Protocol values.
var validPromptGuardProtocols = map[string]bool{
	PromptGuardProtocolHTTPChat:     true,
	PromptGuardProtocolHTTPClassify: true,
}

// validatePromptGuardBackend validates the prompt_guard backend selection and
// that the selected backend is actually wired up.
func validatePromptGuardBackend(cfg *RouterConfig) error {
	if err := validatePromptGuardBackendConfig(&cfg.PromptGuard); err != nil {
		return err
	}
	return validatePromptGuardWiring(cfg)
}

// validatePromptGuardBackendConfig validates the prompt_guard backend selection:
// variant (local) and protocol (remote) are mutually exclusive, and each must
// name a recognized value.
func validatePromptGuardBackendConfig(cfg *PromptGuardConfig) error {
	if cfg.Variant != "" && cfg.Protocol != "" {
		return fmt.Errorf("prompt_guard: variant %q and protocol %q are mutually exclusive - "+
			"variant selects a local model, protocol selects a remote one", cfg.Variant, cfg.Protocol)
	}
	if err := cfg.ClassifierOnErrorConfig.ValidateOnError(); err != nil {
		return fmt.Errorf("prompt_guard.%w", err)
	}
	if cfg.Protocol != "" {
		if !validPromptGuardProtocols[cfg.Protocol] {
			return fmt.Errorf("prompt_guard.protocol: unrecognized value %q, must be one of: %s, %s",
				cfg.Protocol, PromptGuardProtocolHTTPChat, PromptGuardProtocolHTTPClassify)
		}
		return nil
	}
	if !validPromptGuardVariants[cfg.Variant] {
		return fmt.Errorf("prompt_guard.variant: unrecognized value %q, must be one of: %s, %s",
			cfg.Variant, PromptGuardVariantCandle, PromptGuardVariantMmBERT32K)
	}
	return nil
}

// validatePromptGuardWiring rejects a remote prompt_guard backend that is not
// fully wired up.
//
// Every field checked here is one IsPromptGuardEnabled() requires for a
// protocol backend. When one is missing that helper just returns false, which
// drops the jailbreak signal from the dispatch set - so the guardrail silently
// never runs and on_error: block becomes a no-op, the exact fail-open it exists
// to prevent. Failing config load instead makes the misconfiguration visible.
//
// jailbreak_mapping_path is deliberately NOT required here even though
// IsPromptGuardEnabled() also needs it. It is the same class of fail-open, but
// the operator path reaches it through a serialization bug rather than an
// author's mistake: the CRD drops `enabled: false` (omitempty on a bool) so the
// router falls back to its default `Enabled: true`, while
// CanonicalPromptGuardModule emits `jailbreak_mapping_path: ""` (no omitempty)
// and blanks the default path. Requiring it here turns that into a hard startup
// failure for every operator deployment. Tracked separately - fixing it means
// changing how the operator serializes those two fields, not adding a check.
func validatePromptGuardWiring(cfg *RouterConfig) error {
	if !cfg.PromptGuard.Enabled || cfg.PromptGuard.Protocol == "" {
		return nil
	}

	guardrail := cfg.FindExternalModelByRole(ModelRoleGuardrail)
	if guardrail == nil {
		return fmt.Errorf(
			"prompt_guard.protocol %q requires an entry in external_models with model_role: %s",
			cfg.PromptGuard.Protocol, ModelRoleGuardrail)
	}
	if guardrail.ModelEndpoint.Address == "" {
		return fmt.Errorf(
			"external_models entry with model_role: %s is missing llm_endpoint.address, required by prompt_guard.protocol %q",
			ModelRoleGuardrail, cfg.PromptGuard.Protocol)
	}
	if guardrail.ModelName == "" {
		return fmt.Errorf(
			"external_models entry with model_role: %s is missing llm_model_name, required by prompt_guard.protocol %q",
			ModelRoleGuardrail, cfg.PromptGuard.Protocol)
	}
	return nil
}
