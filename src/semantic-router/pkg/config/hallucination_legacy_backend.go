package config

import (
	"fmt"
	"net/url"
	"strings"
)

// The hallucination detector's backend is selected the way every other span
// consumer's is: a deployment plus a recipe binding with an adapter and the
// token_spans.v1 contract. The scalar form below predates that mechanism and
// keeps working as shorthand: `backend: endpoint` with `endpoint` and
// `model_id` desugars into an http deployment bound through the http_chat
// adapter, and `backend: candle` (or unset) is the local module default. Both
// resolve through the same binding plan; nothing at runtime reads the scalar
// to choose a code path.
const (
	// HallucinationBackendCandle names the in-process detector (default).
	HallucinationBackendCandle = "candle"
	// HallucinationBackendEndpoint is the legacy shorthand for a chat-based
	// remote detector behind an OpenAI-compatible server.
	HallucinationBackendEndpoint = "endpoint"
	// LegacyHallucinationEndpointModel is the external-model name the desugared
	// binding refers to. It is synthesized from the scalar endpoint and
	// model_id rather than looked up in the external catalog, so the legacy
	// endpoint URL, base path included, is used exactly as written.
	LegacyHallucinationEndpointModel = "hallucination_detector.legacy_endpoint"
)

// NormalizedBackend returns the trimmed, lower-cased legacy backend token,
// defaulting to candle when unset. It describes how the detector was
// configured, for display and for the model-download gates; backend selection
// itself goes through the binding plan.
func (c *HallucinationModelConfig) NormalizedBackend() string {
	backend := strings.ToLower(strings.TrimSpace(c.Backend))
	if backend == "" {
		return HallucinationBackendCandle
	}
	return backend
}

// LegacyHallucinationBinding desugars the scalar `backend: endpoint` form into
// the binding CompileModelBindings would otherwise expect the recipe to
// declare. The boolean is false for the candle default, where the module's
// canonical local binding applies. An unknown backend token or an unusable
// endpoint is a configuration error.
func LegacyHallucinationBinding(model *HallucinationModelConfig) (ModelBinding, ModelDeployment, bool, error) {
	if model == nil {
		return ModelBinding{}, ModelDeployment{}, false, nil
	}
	if err := ValidateHallucinationBackend(model); err != nil {
		return ModelBinding{}, ModelDeployment{}, false, err
	}
	if model.NormalizedBackend() != HallucinationBackendEndpoint {
		return ModelBinding{}, ModelDeployment{}, false, nil
	}
	binding := ModelBinding{Deployment: "hallucination_detector", Contract: RemoteClassifierContractTokenSpans, Adapter: RemoteClassifierProtocolHTTPChat}
	deployment := ModelDeployment{Provider: "http", ExternalModel: LegacyHallucinationEndpointModel}
	return binding, deployment, true, nil
}

// LegacyHallucinationExternalModel materializes the external model the
// desugared binding refers to: the endpoint URL as written (the chat adapter
// posts to <endpoint>/chat/completions) and the configured model_id.
func LegacyHallucinationExternalModel(model *HallucinationModelConfig) *ExternalModelConfig {
	return &ExternalModelConfig{
		Name:           LegacyHallucinationEndpointModel,
		ModelRole:      ModelRoleClassification,
		ModelName:      model.ModelID,
		ModelEndpoint:  ClassifierVLLMEndpoint{Address: strings.TrimRight(strings.TrimSpace(model.Endpoint), "/")},
		TimeoutSeconds: 10,
	}
}

// validateHallucinationContracts validates the legacy backend scalar during
// config validation, so an unusable value fails at load rather than when the
// binding plan is compiled.
func validateHallucinationContracts(cfg *RouterConfig) error {
	return ValidateHallucinationBackend(&cfg.HallucinationMitigation.HallucinationModel)
}

// ValidateHallucinationBackend validates the legacy backend scalar. An unset
// backend defaults to candle; an unknown value is rejected, and the endpoint
// form requires an absolute http(s) endpoint plus a model_id. This is a pure
// check and does not mutate the config.
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
		return fmt.Errorf("hallucination detector backend %q is not supported; use %q or %q, or bind hallucination_detector to a deployment",
			cfg.Backend, HallucinationBackendCandle, HallucinationBackendEndpoint)
	}
	endpoint := cfg.Endpoint
	if strings.TrimSpace(endpoint) == "" {
		return fmt.Errorf("hallucination detector endpoint is required when backend is %q", HallucinationBackendEndpoint)
	}
	// Reject surrounding whitespace: the runtime constructor uses the raw value,
	// so a padded endpoint would pass validation and then build an invalid URL.
	if endpoint != strings.TrimSpace(endpoint) {
		return fmt.Errorf("hallucination detector endpoint %q must not have leading or trailing whitespace", endpoint)
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
