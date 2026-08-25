package config

// CanonicalProviders keeps physical deployment bindings separate from the
// connection-free semantic Model cards under routing.modelCards.
type CanonicalProviders struct {
	Defaults CanonicalProviderDefaults `yaml:"defaults,omitempty"`
	Models   []CanonicalProviderModel  `yaml:"models,omitempty"`
}

// CanonicalProviderDefaults contains provider-wide routing defaults. Model
// connections and per-Model runtime policy belong to CanonicalProviderModel.
type CanonicalProviderDefaults struct {
	DefaultModel           string                           `yaml:"default_model,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
}

// CanonicalProviderModel is the public v0.3 physical Model record. It joins
// exactly one routing.modelCards member by Name and never carries generated
// resource, backend, revision, or Provider-catalog identity.
type CanonicalProviderModel struct {
	Name             string                `yaml:"name"`
	ReasoningFamily  string                `yaml:"reasoning_family,omitempty"`
	ProviderModelID  string                `yaml:"provider_model_id,omitempty"`
	BackendRefs      []CanonicalBackendRef `yaml:"backend_refs,omitempty"`
	Control          ModelControl          `yaml:"control,omitempty"`
	Pricing          ModelRuntimePricing   `yaml:"pricing,omitempty"`
	APIFormat        string                `yaml:"api_format,omitempty"`
	ExternalModelIDs map[string]string     `yaml:"external_model_ids,omitempty"`
}

// ModelControl owns per-Model invocation and physical-backend resilience. Its
// substructures keep related policy readable without exposing the flat
// transport representation used by the runtime.
type ModelControl struct {
	Retry   *ModelRetry   `yaml:"retry,omitempty" json:"retry,omitempty"`
	Timeout *ModelTimeout `yaml:"timeout,omitempty" json:"timeout,omitempty"`
}

// ModelRetry.Count is the number of additional attempts after the initial
// call. On is a set of closed, Router-owned retry evidence classes.
type ModelRetry struct {
	Count int      `yaml:"count,omitempty" json:"count,omitempty"`
	On    []string `yaml:"on,omitempty" json:"on,omitempty"`
}

type ModelTimeout struct {
	Request string `yaml:"request,omitempty" json:"request,omitempty"`
	Stream  string `yaml:"stream,omitempty" json:"stream,omitempty"`
}

// CanonicalBackendRef is one public physical backend reference. Provider
// Integrations compile it into the provider-neutral immutable runtime shape.
type CanonicalBackendRef struct {
	Name         string            `yaml:"name,omitempty"`
	Endpoint     string            `yaml:"endpoint,omitempty"`
	Protocol     string            `yaml:"protocol,omitempty"`
	Weight       int               `yaml:"weight,omitempty"`
	Type         string            `yaml:"type,omitempty"`
	BaseURL      string            `yaml:"base_url,omitempty"`
	Provider     string            `yaml:"provider,omitempty"`
	AuthHeader   string            `yaml:"auth_header,omitempty"`
	AuthPrefix   string            `yaml:"auth_prefix,omitempty"`
	ExtraHeaders map[string]string `yaml:"extra_headers,omitempty"`
	APIVersion   string            `yaml:"api_version,omitempty"`
	ChatPath     string            `yaml:"chat_path,omitempty"`
	Credential   string            `yaml:"credential,omitempty"`
	APIKey       string            `yaml:"api_key,omitempty" json:"-"`
	APIKeyEnv    string            `yaml:"api_key_env,omitempty"`
}

func canonicalProviderDefaults(providers CanonicalProviders) CanonicalProviderDefaults {
	return providers.Defaults
}
