package config

import "fmt"

// validatePIIModelBackendContracts runs the PII backend and on_error checks at
// config load, so a remote PII backend that is mixed with the local selector, or
// an unknown on_error value, is rejected before any classifier is built. Window
// geometry is static; provider and input budget checks need recipe bindings.
func validatePIIModelBackendContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("PII model configuration is nil")
	}
	model := &cfg.PIIModel
	if err := model.Window.validateGeometry(); err != nil {
		return fmt.Errorf("classifier.pii.%w", err)
	}
	if err := model.ClassifierOnErrorConfig.ValidateOnError(); err != nil {
		return fmt.Errorf("classifier.pii.%w", err)
	}
	if model.Backend == nil {
		return nil
	}
	if model.UseMmBERT32K {
		return fmt.Errorf("classifier.pii: backend is mutually exclusive with use_mmbert_32k")
	}
	if model.Backend.Protocol != RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf("classifier.pii.backend.protocol %q is not supported by the PII consumer", model.Backend.Protocol)
	}
	if _, err := ResolveRemoteClassifierBackend(
		cfg,
		model.Backend,
		ModelRoleClassification,
		RemoteClassifierContractTokenSpans,
	); err != nil {
		return fmt.Errorf("classifier.pii: %w", err)
	}
	return nil
}
