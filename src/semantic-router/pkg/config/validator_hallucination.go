package config

import (
	"fmt"
	"net/url"
	"strings"
)

const (
	// HallucinationBackendCandle runs the in-process Candle token classifier (default).
	HallucinationBackendCandle = "candle"
	// HallucinationBackendEndpoint calls a generative span detector behind an
	// OpenAI-compatible server (e.g. vLLM).
	HallucinationBackendEndpoint = "endpoint"
)

// NormalizedBackend returns the trimmed, lower-cased detector backend, defaulting
// to candle when unset. Read sites should use this instead of comparing the raw
// Backend field so an unset or differently-cased value resolves deterministically.
func (c *HallucinationModelConfig) NormalizedBackend() string {
	backend := strings.ToLower(strings.TrimSpace(c.Backend))
	if backend == "" {
		return HallucinationBackendCandle
	}
	return backend
}

// validateHallucinationContracts validates the hallucination detector backend
// selection during config validation.
func validateHallucinationContracts(cfg *RouterConfig) error {
	return ValidateHallucinationBackend(&cfg.HallucinationMitigation.HallucinationModel)
}

// ValidateHallucinationBackend validates the detector backend selection. An unset
// backend defaults to candle (resolved via NormalizedBackend at read time); an
// unknown value is rejected, and the endpoint backend requires an absolute http(s)
// endpoint plus a model_id. This is a pure check and does not mutate the config.
func ValidateHallucinationBackend(cfg *HallucinationModelConfig) error {
	if cfg == nil {
		return nil
	}

	switch cfg.NormalizedBackend() {
	case HallucinationBackendCandle:
		return nil
	case HallucinationBackendEndpoint:
		// endpoint backend requirements are validated below
	default:
		return fmt.Errorf("hallucination detector backend %q is not supported; use %q or %q",
			cfg.Backend, HallucinationBackendCandle, HallucinationBackendEndpoint)
	}

	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		return fmt.Errorf("hallucination detector endpoint is required when backend is %q", HallucinationBackendEndpoint)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("hallucination detector endpoint %q is not a valid URL: %w", endpoint, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("hallucination detector endpoint %q must be an absolute http(s) URL", endpoint)
	}
	if parsed.Host == "" {
		return fmt.Errorf("hallucination detector endpoint %q must include a host", endpoint)
	}
	if strings.TrimSpace(cfg.ModelID) == "" {
		return fmt.Errorf("hallucination detector model_id is required when backend is %q", HallucinationBackendEndpoint)
	}
	return nil
}
