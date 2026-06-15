package config

// CanonicalProviders holds deployment bindings and provider defaults.
type CanonicalProviders struct {
	Defaults CanonicalProviderDefaults `yaml:"defaults,omitempty"`
	Models   []CanonicalProviderModel  `yaml:"models,omitempty"`
}

// CanonicalProviderDefaults groups provider-wide defaults separately from
// per-model access bindings.
type CanonicalProviderDefaults struct {
	DefaultModel           string                           `yaml:"default_model,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
}

// CanonicalProviderModel binds a logical routing model to concrete access
// details without mixing those access details into provider-wide defaults.
type CanonicalProviderModel struct {
	Name             string                `yaml:"name"`
	ReasoningFamily  string                `yaml:"reasoning_family,omitempty"`
	ProviderModelID  string                `yaml:"provider_model_id,omitempty"`
	BackendRefs      []CanonicalBackendRef `yaml:"backend_refs,omitempty"`
	Pricing          ModelPricing          `yaml:"pricing,omitempty"`
	APIFormat        string                `yaml:"api_format,omitempty"`
	ExternalModelIDs map[string]string     `yaml:"external_model_ids,omitempty"`
}

// CanonicalBackendRef defines one physical backend target for a provider model.
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
	APIKey       string            `yaml:"api_key,omitempty"`
	APIKeyEnv    string            `yaml:"api_key_env,omitempty"`
}

func canonicalProviderDefaults(providers CanonicalProviders) CanonicalProviderDefaults {
	return providers.Defaults
}

func canonicalBackendRefs(model CanonicalProviderModel) []CanonicalBackendRef {
	return model.BackendRefs
}
