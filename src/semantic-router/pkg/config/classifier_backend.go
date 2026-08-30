package config

import (
	"fmt"
	"strings"
)

// Remote classifier protocols describe how a built-in classifier is called.
// They are deliberately independent from the response contract.
const (
	RemoteClassifierProtocolHTTPClassify = "http_classify"
	RemoteClassifierProtocolHTTPChat     = "http_chat"
)

// Remote classifier contracts describe the semantic product returned by a
// remote classifier. Category is the first consumer and currently supports
// the complete label distribution contract.
const (
	RemoteClassifierContractLabelDistribution = "label_distribution"
)

const defaultRemoteClassifierTimeoutSeconds = 5

// RemoteClassifierBackend is the shared remote attachment contract for
// built-in classifier modules. A nil backend means that the module uses its
// existing local implementation. TimeoutSeconds is a pointer so omitted and
// an explicitly invalid zero value cannot be confused during validation.
type RemoteClassifierBackend struct {
	Protocol       string `yaml:"protocol" json:"protocol"`
	Contract       string `yaml:"contract,omitempty" json:"contract,omitempty"`
	Model          string `yaml:"model" json:"model"`
	TimeoutSeconds *int   `yaml:"timeout_seconds,omitempty" json:"timeout_seconds,omitempty"`
}

// UnmarshalYAML rejects stale or misspelled timeout fields instead of letting
// yaml.v2 silently discard them. The final public contract has one spelling:
// timeout_seconds.
func (b *RemoteClassifierBackend) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw map[interface{}]interface{}
	if err := unmarshal(&raw); err != nil {
		return err
	}
	for key := range raw {
		name, ok := key.(string)
		if !ok {
			return fmt.Errorf("backend contains a non-string field name %v", key)
		}
		switch name {
		case "protocol", "contract", "model", "timeout_seconds":
		default:
			return fmt.Errorf("backend: unsupported field %q (use timeout_seconds for the timeout)", name)
		}
	}
	type backendAlias RemoteClassifierBackend
	var decoded backendAlias
	if err := unmarshal(&decoded); err != nil {
		return err
	}
	*b = RemoteClassifierBackend(decoded)
	return nil
}

// EffectiveContract returns the configured contract or the consumer's
// default. The receiver is not mutated, preserving omitted-vs-explicit state
// for canonical round trips.
func (b *RemoteClassifierBackend) EffectiveContract(defaultContract string) string {
	if b == nil || strings.TrimSpace(b.Contract) == "" {
		return defaultContract
	}
	return b.Contract
}

// EffectiveTimeoutSeconds returns the shared HTTP classifier timeout default.
func (b *RemoteClassifierBackend) EffectiveTimeoutSeconds() int {
	if b == nil || b.TimeoutSeconds == nil {
		return defaultRemoteClassifierTimeoutSeconds
	}
	return *b.TimeoutSeconds
}

// Validate checks the fields common to every remote classifier attachment.
// Signal-specific validation is layered on top by the first consumer.
func (b *RemoteClassifierBackend) Validate() error {
	if b == nil {
		return nil
	}
	if strings.TrimSpace(b.Protocol) == "" {
		return fmt.Errorf("backend.protocol is required")
	}
	switch b.Protocol {
	case RemoteClassifierProtocolHTTPClassify, RemoteClassifierProtocolHTTPChat:
	default:
		return fmt.Errorf("backend.protocol: unsupported value %q", b.Protocol)
	}
	if strings.TrimSpace(b.Model) == "" {
		return fmt.Errorf("backend.model is required")
	}
	if b.Contract != "" {
		switch b.Contract {
		case RemoteClassifierContractLabelDistribution:
		default:
			return fmt.Errorf("backend.contract: unsupported value %q", b.Contract)
		}
	}
	if b.TimeoutSeconds != nil && *b.TimeoutSeconds <= 0 {
		return fmt.Errorf("backend.timeout_seconds must be greater than zero, got %d", *b.TimeoutSeconds)
	}
	return nil
}

// ResolveRemoteClassifierBackend validates a shared backend and resolves its
// explicitly named external model. The role assertion is kept separate from
// the transport protocol so a model cannot be selected merely because it is
// the first catalog entry with a matching role.
func ResolveRemoteClassifierBackend(
	cfg *RouterConfig,
	backend *RemoteClassifierBackend,
	expectedRole string,
	expectedContract string,
) (*ExternalModelConfig, error) {
	if backend == nil {
		return nil, nil
	}
	if err := backend.Validate(); err != nil {
		return nil, err
	}
	if expectedContract != "" && backend.EffectiveContract(expectedContract) != expectedContract {
		return nil, fmt.Errorf("backend.contract %q is incompatible; expected %q", backend.EffectiveContract(expectedContract), expectedContract)
	}
	external, err := findNamedExternalModel(cfg, backend.Model)
	if err != nil {
		return nil, err
	}
	if err := validateExternalClassifierModel(external, backend.Model, expectedRole); err != nil {
		return nil, err
	}
	return external, nil
}

func findNamedExternalModel(cfg *RouterConfig, name string) (*ExternalModelConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("backend.model %q cannot be resolved without router configuration", name)
	}
	var external *ExternalModelConfig
	for i := range cfg.ExternalModels {
		if cfg.ExternalModels[i].Name != name {
			continue
		}
		if external != nil {
			return nil, fmt.Errorf("backend.model %q is ambiguous: multiple entries use this external-catalog name", name)
		}
		external = &cfg.ExternalModels[i]
	}
	if external == nil {
		return nil, fmt.Errorf("backend.model %q is not declared in global.model_catalog.external[].name", name)
	}
	return external, nil
}

func validateExternalClassifierModel(external *ExternalModelConfig, name, expectedRole string) error {
	if expectedRole != "" && external.ModelRole != expectedRole {
		return fmt.Errorf("backend.model %q must use model_role %q, got %q", name, expectedRole, external.ModelRole)
	}
	if strings.TrimSpace(external.ModelName) == "" {
		return fmt.Errorf("external model %q requires llm_model_name", name)
	}
	if !validClassifierEndpoint(external.ModelEndpoint) {
		return fmt.Errorf("external model %q requires a valid llm_endpoint address and port", name)
	}
	endpointProtocol := strings.ToLower(strings.TrimSpace(external.ModelEndpoint.Protocol))
	if endpointProtocol != "" && endpointProtocol != "http" && endpointProtocol != "https" {
		return fmt.Errorf("external model %q llm_endpoint.protocol must be http or https", name)
	}
	return nil
}

func validClassifierEndpoint(endpoint ClassifierVLLMEndpoint) bool {
	if strings.TrimSpace(endpoint.Address) == "" {
		return false
	}
	return endpoint.Port >= 1 && endpoint.Port <= 65535
}

// ValidateCategoryModelBackend is the single category-facing entry point for
// local selector compatibility and remote backend validation. It is exported
// so construction paths that receive an already-built RouterConfig cannot
// bypass the same checks performed by config loading.
func ValidateCategoryModelBackend(cfg *RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("category model configuration is nil")
	}
	model := &cfg.CategoryModel
	if err := model.ValidateOnError(); err != nil {
		return fmt.Errorf("classifier.domain.%w", err)
	}
	if err := model.ValidateLocalVariant(); err != nil {
		return err
	}
	if model.Backend == nil {
		return nil
	}
	if model.Variant != "" || model.UseModernBERT || model.UseMmBERT32K {
		return fmt.Errorf("classifier.domain: backend is mutually exclusive with variant and legacy local selectors")
	}
	if model.Backend.Protocol != RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf("classifier.domain.backend.protocol %q is not supported by the category consumer", model.Backend.Protocol)
	}
	if _, err := ResolveRemoteClassifierBackend(
		cfg,
		model.Backend,
		ModelRoleClassification,
		RemoteClassifierContractLabelDistribution,
	); err != nil {
		return fmt.Errorf("classifier.domain: %w", err)
	}
	return nil
}
