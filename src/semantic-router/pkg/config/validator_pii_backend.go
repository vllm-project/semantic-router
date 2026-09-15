package config

// validatePIIModelBackendContracts runs the PII backend and on_error checks at
// config load, so a remote PII backend that is mixed with the local selector, or
// an unknown on_error value, is rejected before any classifier is built.
func validatePIIModelBackendContracts(cfg *RouterConfig) error {
	return ValidatePIIModelBackend(cfg)
}
