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
// remote classifier, independently of the protocol used to fetch it. A
// consumer declares which contracts it can read; the contract chosen tells the
// runtime how to interpret the response, so the two are not interchangeable.
const (
	RemoteClassifierContractLabelDistribution = "label_distribution.v1"
	// RemoteClassifierContractScore carries a single continuous score for a
	// regression-style model, such as a query-difficulty scorer. It has no
	// label of its own; the consumer turns the score into a verdict.
	RemoteClassifierContractScore = "score.v1"
)

const RemoteClassifierDeadlineMs = 5000

// RemoteClassifierCircuitBreakerConfig controls per-backend circuit breaker
// behaviour. When Enabled is true, consecutive failures trip the breaker open,
// skipping requests for the configured duration. A single half-open probe then
// tests whether the backend has recovered. The breaker only decides whether to
// place the call; a skipped call still surfaces as the existing
// unresolved/timeout error so decision-tree Unknown, rules.on_unknown, and
// prompt-guard on_error keep their current meaning.
//
// All fields are optional. A zero or nil config leaves the breaker disabled.
type RemoteClassifierCircuitBreakerConfig struct {
	// Enabled switches the breaker on for this remote backend instance.
	Enabled bool `yaml:"enabled" json:"enabled"`
	// ConsecutiveFailures is the number of sequential failures that trip
	// the breaker open. Defaults to 5.
	ConsecutiveFailures *int `yaml:"consecutive_failures,omitempty" json:"consecutive_failures,omitempty"`
	// OpenIntervalMs is how long the breaker stays open before transitioning
	// to half-open. Defaults to 30000 (30 seconds).
	OpenIntervalMs *int `yaml:"open_interval_ms,omitempty" json:"open_interval_ms,omitempty"`
	// HalfOpenMaxRequests is the number of probe requests allowed in the
	// half-open state. Defaults to 1.
	HalfOpenMaxRequests *int `yaml:"half_open_max_requests,omitempty" json:"half_open_max_requests,omitempty"`
}

// EffectiveConsecutiveFailures returns the configured value or the default.
func (c *RemoteClassifierCircuitBreakerConfig) EffectiveConsecutiveFailures() int {
	if c == nil || c.ConsecutiveFailures == nil {
		return defaultCircuitBreakerConsecutiveFailures
	}
	return *c.ConsecutiveFailures
}

// EffectiveOpenInterval returns the configured open duration or the default.
func (c *RemoteClassifierCircuitBreakerConfig) EffectiveOpenInterval() int {
	if c == nil || c.OpenIntervalMs == nil {
		return defaultCircuitBreakerOpenIntervalMs
	}
	return *c.OpenIntervalMs
}

// EffectiveHalfOpenMaxRequests returns the configured probe count or the default.
func (c *RemoteClassifierCircuitBreakerConfig) EffectiveHalfOpenMaxRequests() int {
	if c == nil || c.HalfOpenMaxRequests == nil {
		return defaultCircuitBreakerHalfOpenMaxRequests
	}
	return *c.HalfOpenMaxRequests
}

const (
	defaultCircuitBreakerConsecutiveFailures   = 10
	defaultCircuitBreakerOpenIntervalMs        = 30000
	defaultCircuitBreakerHalfOpenMaxRequests   = 1
)

// RemoteClassifierBackend is the shared remote attachment contract for
// built-in classifier modules. A nil backend means that the module uses its
// existing local implementation. DeadlineMs is a pointer so omitted and
// an explicitly invalid zero value cannot be confused during validation.
type RemoteClassifierBackend struct {
	Protocol        string                              `yaml:"protocol" json:"protocol"`
	Contract        string                              `yaml:"contract,omitempty" json:"contract,omitempty"`
	Model           string                              `yaml:"model" json:"model"`
	DeadlineMs      *int                                `yaml:"deadline_ms,omitempty" json:"deadline_ms,omitempty"`
	CircuitBreaker  *RemoteClassifierCircuitBreakerConfig `yaml:"circuit_breaker,omitempty" json:"circuit_breaker,omitempty"`
}

// UnmarshalYAML rejects stale or misspelled deadline fields instead of letting
// yaml.v2 silently discard them. The final public contract has one spelling:
// deadline_ms.
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
		case "protocol", "contract", "model", "deadline_ms", "circuit_breaker":
		default:
			return fmt.Errorf("backend: unsupported field %q (use deadline_ms for the request deadline, circuit_breaker for circuit breaker config)", name)
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

// EffectiveDeadlineMs returns the shared HTTP classifier deadline default.
func (b *RemoteClassifierBackend) EffectiveDeadlineMs() int {
	if b == nil || b.DeadlineMs == nil {
		return RemoteClassifierDeadlineMs
	}
	return *b.DeadlineMs
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
		case RemoteClassifierContractLabelDistribution, RemoteClassifierContractScore:
		default:
			return fmt.Errorf("backend.contract: unsupported value %q", b.Contract)
		}
	}
	if b.DeadlineMs != nil && *b.DeadlineMs <= 0 {
		return fmt.Errorf("backend.deadline_ms must be greater than zero, got %d", *b.DeadlineMs)
	}
	if err := b.validateCircuitBreaker(); err != nil {
		return err
	}
	return nil
}

// validateCircuitBreaker checks circuit breaker thresholds are positive when enabled.
func (b *RemoteClassifierBackend) validateCircuitBreaker() error {
	if b.CircuitBreaker == nil || !b.CircuitBreaker.Enabled {
		return nil
	}
	if b.CircuitBreaker.ConsecutiveFailures != nil && *b.CircuitBreaker.ConsecutiveFailures <= 0 {
		return fmt.Errorf("backend.circuit_breaker.consecutive_failures must be positive when enabled, got %d", *b.CircuitBreaker.ConsecutiveFailures)
	}
	if b.CircuitBreaker.OpenIntervalMs != nil && *b.CircuitBreaker.OpenIntervalMs <= 0 {
		return fmt.Errorf("backend.circuit_breaker.open_interval_ms must be positive when enabled, got %d", *b.CircuitBreaker.OpenIntervalMs)
	}
	if b.CircuitBreaker.HalfOpenMaxRequests != nil && *b.CircuitBreaker.HalfOpenMaxRequests <= 0 {
		return fmt.Errorf("backend.circuit_breaker.half_open_max_requests must be positive when enabled, got %d", *b.CircuitBreaker.HalfOpenMaxRequests)
	}
	return nil
}

// ResolveRemoteClassifierBackend validates a shared backend and resolves its
// explicitly named external model. The role assertion is kept separate from
// the transport protocol so a model cannot be selected merely because it is
// the first catalog entry with a matching role.
//
// A consumer passes every contract it can read. Passing exactly one keeps that
// contract available as the default for an omitted `contract:` field, because
// there is nothing to guess. A consumer that accepts several cannot default -
// see requireDeclaredContract.
func ResolveRemoteClassifierBackend(
	cfg *RouterConfig,
	backend *RemoteClassifierBackend,
	expectedRole string,
	expectedContracts ...string,
) (*ExternalModelConfig, error) {
	if backend == nil {
		return nil, nil
	}
	if err := backend.Validate(); err != nil {
		return nil, err
	}
	if err := requireDeclaredContract(backend, expectedContracts); err != nil {
		return nil, err
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

// requireDeclaredContract checks a backend's contract against the contracts a
// consumer declared it can read.
//
// With one declared contract the field may be omitted: the single possibility
// is the default. With several, omitting it would leave the runtime guessing
// which response shape to expect - and guessing wrong surfaces per request
// rather than at config load - so an explicit value is required.
func requireDeclaredContract(backend *RemoteClassifierBackend, declared []string) error {
	if len(declared) == 0 {
		return nil
	}
	if len(declared) == 1 {
		if got := backend.EffectiveContract(declared[0]); got != declared[0] {
			return fmt.Errorf("backend.contract %q is incompatible; expected %q", got, declared[0])
		}
		return nil
	}
	contract := strings.TrimSpace(backend.Contract)
	if contract == "" {
		return fmt.Errorf(
			"backend.contract is required because this signal reads more than one contract; set it to one of: %s",
			strings.Join(declared, ", "))
	}
	for _, candidate := range declared {
		if contract == candidate {
			return nil
		}
	}
	return fmt.Errorf(
		"backend.contract %q is incompatible; expected one of: %s",
		contract, strings.Join(declared, ", "))
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
