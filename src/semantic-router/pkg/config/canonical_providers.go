package config

import (
	"reflect"
	"strings"
)

// CanonicalProviders holds deployment bindings and provider defaults.
type CanonicalProviders struct {
	Defaults CanonicalProviderDefaults `yaml:"defaults,omitempty"`
	Models   []CanonicalProviderModel  `yaml:"models,omitempty"`
}

// CanonicalProviderDefaults groups provider-wide defaults separately from
// per-model access bindings.
type CanonicalProviderDefaults struct {
	DefaultModel           string `yaml:"model,omitempty"`
	DefaultReasoningEffort string `yaml:"reasoning_effort,omitempty"`
	// TLS is the trust of every https backend whose own tls names no CA.
	TLS *ProviderDefaultsTLS `yaml:"tls,omitempty"`
}

// CanonicalProviderModel describes one model alias in one place: how requests
// reach it (backend_refs, api_format, pricing, reliability, ...) and what it
// is (its facts). The facts overlay the built-in catalog card named by
// catalog, or describe a custom model.
type CanonicalProviderModel struct {
	Name            string                `yaml:"name"`
	Catalog         string                `yaml:"catalog,omitempty"`
	Deployment      string                `yaml:"deployment,omitempty"`
	Reasoning       *CanonicalReasoning   `yaml:"reasoning,omitempty"`
	ProviderModelID string                `yaml:"provider_model_id,omitempty"`
	BackendRefs     []CanonicalBackendRef `yaml:"backend_refs,omitempty"`
	Pricing         ModelPricing          `yaml:"pricing,omitempty"`
	Reliability     ProviderReliability   `yaml:"reliability,omitempty"`
	// APIFormat is the API the model's backends serve, by API name:
	// chat_completions (the default), responses, messages, images or
	// systemone. openai and anthropic are accepted as aliases of
	// chat_completions and messages.
	APIFormat string `yaml:"api_format,omitempty"`
	// Schema is the version of that API the backends speak, one of the API's
	// schema IDs; omitted, the API's default (systemone/0.2 for systemone).
	Schema           string            `yaml:"schema,omitempty"`
	ExternalModelIDs map[string]string `yaml:"external_model_ids,omitempty"`
	// UnsupportedCapabilities lists the request capabilities the model's
	// backends reject: sampling_min_p or sampling_logit_bias, both of which a
	// vLLM backend that runs speculative decoding rejects. The Router sends
	// this model requests without those fields and reports each in
	// x-vsr-warnings; other models still receive them.
	UnsupportedCapabilities []string `yaml:"unsupported_capabilities,omitempty"`
	ModelFacts              `yaml:",inline"`
}

// ModelFactFields are the keys of the facts a providers.models[] entry states.
func ModelFactFields() []string {
	facts := reflect.TypeOf(ModelFacts{})
	fields := make([]string, 0, facts.NumField())
	for i := range facts.NumField() {
		name, _, _ := strings.Cut(facts.Field(i).Tag.Get("yaml"), ",")
		fields = append(fields, name)
	}
	return fields
}

// CanonicalReasoning either references one built-in family or defines the
// request projection for a private/custom model inline. Family is mutually
// exclusive with the inline fields.
type CanonicalReasoning struct {
	Family              string            `yaml:"family,omitempty"`
	Type                string            `yaml:"type,omitempty"`
	Parameter           string            `yaml:"parameter,omitempty"`
	ActivationParameter string            `yaml:"activation_parameter,omitempty"`
	EffortFlags         map[string]string `yaml:"effort_flags,omitempty"`
	Levels              []string          `yaml:"levels,omitempty"`
	Default             string            `yaml:"default,omitempty"`
	Modes               []string          `yaml:"modes,omitempty"`
	DefaultMode         string            `yaml:"default_mode,omitempty"`
	Disabled            string            `yaml:"disabled,omitempty"`
}

// ProviderReliability controls data-plane load balancing, timeouts, retries
// and endpoint health for one provider model. Every data plane honors it the
// same way; an empty field keeps the default.
type ProviderReliability struct {
	LBPolicy            string `yaml:"lb_policy,omitempty"`
	RetryCount          int    `yaml:"retry_count,omitempty"`
	RetryOn             string `yaml:"retry_on,omitempty"`
	Consecutive5xx      int    `yaml:"consecutive_5xx,omitempty"`
	BaseEjectionTime    string `yaml:"base_ejection_time,omitempty"`
	MaxEjectionPercent  int    `yaml:"max_ejection_percent,omitempty"`
	HealthCheckPath     string `yaml:"health_check_path,omitempty"`
	HealthCheckInterval string `yaml:"health_check_interval,omitempty"`
	HealthCheckTimeout  string `yaml:"health_check_timeout,omitempty"`

	// ConnectTimeout bounds opening a connection, TLS included (default 10s).
	ConnectTimeout string `yaml:"connect_timeout,omitempty"`
	// TotalTimeout bounds the whole upstream call, retries and streamed body
	// included; it replaces the listener timeout for this model. 0s disables it.
	TotalTimeout string `yaml:"total_timeout,omitempty"`
	// IdleTimeout bounds each wait for more of the response; it replaces the
	// listener timeout for this model. 0s disables it.
	IdleTimeout string `yaml:"idle_timeout,omitempty"`
	// PerTryTimeout bounds each attempt until its response starts.
	PerTryTimeout string `yaml:"per_try_timeout,omitempty"`
	// FirstByteTimeout bounds the wait for the first response body byte, so a
	// stalled stream can still be retried. Only the native gateway honors it.
	FirstByteTimeout string `yaml:"first_byte_timeout,omitempty"`
	// RetriableStatusCodes are retried when retry_on includes
	// retriable-status-codes.
	RetriableStatusCodes []int `yaml:"retriable_status_codes,omitempty"`
	// RetryBackOffBase and RetryBackOffMax shape the exponential back-off with
	// full jitter between retries (defaults 25ms and ten times the base).
	RetryBackOffBase string `yaml:"retry_back_off_base,omitempty"`
	RetryBackOffMax  string `yaml:"retry_back_off_max,omitempty"`
	// RetryAfterMax makes a retry wait the response's Retry-After seconds, up
	// to this bound; without it Retry-After is ignored.
	RetryAfterMax string `yaml:"retry_after_max,omitempty"`
	// RetryBudgetPercent and RetryBudgetMinConcurrency replace the fixed limit
	// of three concurrent retries with a share of the requests in flight
	// (defaults 20% and 3 once either is set).
	RetryBudgetPercent        float64 `yaml:"retry_budget_percent,omitempty"`
	RetryBudgetMinConcurrency int     `yaml:"retry_budget_min_concurrency,omitempty"`
}

// IsZero reports whether no reliability field is set.
func (r ProviderReliability) IsZero() bool {
	return reflect.DeepEqual(r, ProviderReliability{})
}

// CanonicalBackendRef defines one physical backend target for a provider model.
type CanonicalBackendRef struct {
	Name       string `yaml:"name,omitempty"`
	Endpoint   string `yaml:"endpoint,omitempty"`
	Protocol   string `yaml:"protocol,omitempty"`
	Weight     int    `yaml:"weight,omitempty"`
	Provider   string `yaml:"provider,omitempty"`
	APIVersion string `yaml:"api_version,omitempty"`
	ChatPath   string `yaml:"chat_path,omitempty"`
	APIKey     string `yaml:"api_key,omitempty"`
	// Auth is the backend's credential, as in a deployment's connection
	// block: the credential mode (delegated sends the requesting user's
	// credential instead of a key), the header (the provider's own when
	// unset), the prefix ("" sends the bare key, presence-aware) and the
	// key's environment variable.
	Auth *DeploymentAuth `yaml:"auth,omitempty"`
	// Headers are non-secret headers sent with every request.
	Headers map[string]string `yaml:"headers,omitempty"`
	// TLS verifies an https backend as a deployment's tls block does, for
	// Envoy and for the Router's own calls alike: ca_file replaces the system
	// roots, server_name the endpoint's host, and cert_file with key_file
	// present a client certificate. Verification is never skipped.
	TLS *DeploymentTLS `yaml:"tls,omitempty"`
}

func canonicalProviderDefaults(providers CanonicalProviders) CanonicalProviderDefaults {
	return providers.Defaults
}
