package config

import (
	"fmt"
	"strings"
)

// The hallucination detector's backend is selected the way every other span
// consumer's is: a deployment plus a recipe binding with an adapter and the
// token_spans.v1 contract. The parser refuses the retired `backend: endpoint`
// shorthand; `vllm-sr config migrate` rewrites it into that binding. The
// backend scalar below is what the binding projection records for the
// detector's consumers; nothing selects a code path from user input.
const (
	// HallucinationBackendLocal names the built-in runtime's detector (default).
	HallucinationBackendLocal = ModelRuntimeProvider
	// HallucinationBackendEndpoint marks a detector the projection bound to an
	// http deployment.
	HallucinationBackendEndpoint = "endpoint"
)

// NormalizedBackend returns the trimmed, lower-cased backend token, defaulting
// to the local runtime detector when unset. It describes how the detector is
// served, for display and for the model-download gates.
func (c *HallucinationModelConfig) NormalizedBackend() string {
	backend := strings.ToLower(strings.TrimSpace(c.Backend))
	if backend == "" {
		return HallucinationBackendLocal
	}
	return backend
}

// validateHallucinationContracts validates the backend scalar during config
// validation, so an unusable value fails at load rather than when the binding
// plan is compiled.
func validateHallucinationContracts(cfg *RouterConfig) error {
	return ValidateHallucinationBackend(&cfg.HallucinationMitigation.HallucinationModel)
}

// ValidateHallucinationBackend accepts the local detector and the projection's
// endpoint marker; any other value is rejected. A remote detector is a
// hallucination_detector binding.
func ValidateHallucinationBackend(cfg *HallucinationModelConfig) error {
	if cfg == nil {
		return nil
	}
	switch cfg.NormalizedBackend() {
	case HallucinationBackendLocal, HallucinationBackendEndpoint:
		return nil
	default:
		return fmt.Errorf("hallucination detector backend %q is not supported; use %q, or bind hallucination_detector to a deployment",
			cfg.Backend, HallucinationBackendLocal)
	}
}
