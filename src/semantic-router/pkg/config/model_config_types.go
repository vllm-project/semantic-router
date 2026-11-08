package config

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// PromptCompressionConfig controls NLP-based prompt compression before signal extraction.
type PromptCompressionConfig struct {
	Enabled        bool     `yaml:"enabled"`
	Profile        string   `yaml:"profile,omitempty"`
	MaxTokens      int      `yaml:"max_tokens"`
	MinLength      int      `yaml:"min_length,omitempty"`
	SkipSignals    []string `yaml:"skip_signals,omitempty"`
	TextRankWeight float64  `yaml:"textrank_weight,omitempty"`
	PositionWeight float64  `yaml:"position_weight,omitempty"`
	TFIDFWeight    float64  `yaml:"tfidf_weight,omitempty"`
	NoveltyWeight  float64  `yaml:"novelty_weight,omitempty"`
	PositionDepth  float64  `yaml:"position_depth,omitempty"`
	PreserveFirstN int      `yaml:"preserve_first_n,omitempty"`
	PreserveLastN  int      `yaml:"preserve_last_n,omitempty"`
}

func (pc PromptCompressionConfig) SkipSignalsSet() map[string]bool {
	signals := pc.SkipSignals
	if len(signals) == 0 {
		signals = []string{SignalTypeJailbreak, SignalTypePII}
	}
	m := make(map[string]bool, len(signals))
	for _, s := range signals {
		m[s] = true
	}
	return m
}

type ExternalModelConfig struct {
	Name             string                        `yaml:"name,omitempty"`
	Provider         string                        `yaml:"llm_provider"`
	ModelRole        string                        `yaml:"model_role"`
	ModelEndpoint    ClassifierVLLMEndpoint        `yaml:"llm_endpoint,omitempty"`
	ModelName        string                        `yaml:"llm_model_name,omitempty"`
	TimeoutSeconds   float64                       `yaml:"llm_timeout_seconds,omitempty"`
	AccessKey        string                        `yaml:"access_key,omitempty" json:"-"`
	MaxTokens        int                           `yaml:"max_tokens,omitempty"`
	Temperature      float64                       `yaml:"temperature,omitempty"`
	Reasoning        *ExternalModelReasoningConfig `yaml:"reasoning,omitempty"`
	MaxRequestBytes  int64                         `yaml:"max_request_bytes,omitempty"`
	MaxResponseBytes int64                         `yaml:"max_response_bytes,omitempty"`
}

// ExternalModelReasoningConfig controls reasoning for requests made to a
// vLLM-backed external classifier. Family references the shared reasoning-family
// catalog; the external model contract intentionally does not support inline
// family definitions.
type ExternalModelReasoningConfig struct {
	Family          string `yaml:"family" jsonschema:"required"`
	UseReasoning    *bool  `yaml:"use_reasoning" jsonschema:"required"`
	ReasoningEffort string `yaml:"reasoning_effort,omitempty"`
}

// AdmissionConfig bounds concurrent inference for one Router Model
// deployment. Absent config means no gate and preserves current behavior.
type AdmissionConfig struct {
	MaxConcurrency int    `yaml:"max_concurrency"`
	MaxQueue       int    `yaml:"max_queue,omitempty"`
	QueueTimeoutMs int    `yaml:"queue_timeout_ms,omitempty"`
	OnOverflow     string `yaml:"on_overflow,omitempty"`
}

type ClassifierVLLMEndpoint struct {
	Address         string `yaml:"address"`
	Port            int    `yaml:"port"`
	Protocol        string `yaml:"protocol,omitempty"`
	Name            string `yaml:"name,omitempty"`
	UseChatTemplate bool   `yaml:"use_chat_template,omitempty"`
	PromptTemplate  string `yaml:"prompt_template,omitempty"`
}

type VLLMEndpoint struct {
	Name                string `yaml:"name"`
	Address             string `yaml:"address"`
	Port                int    `yaml:"port"`
	Weight              int    `yaml:"weight,omitempty"`
	Type                string `yaml:"type,omitempty"`
	APIKey              string `yaml:"api_key,omitempty" json:"-"`
	ProviderProfileName string `yaml:"provider_profile,omitempty"`
	Model               string `yaml:"model,omitempty"`
	Protocol            string `yaml:"protocol,omitempty"`
}

type ProviderProfile struct {
	Type               string                          `yaml:"type"`
	Protocol           string                          `yaml:"protocol,omitempty"`
	ReasoningTransport modelcatalog.ReasoningTransport `yaml:"reasoning_transport,omitempty"`
	// ReasoningModes and ReasoningEfforts are catalog-materialized provider API
	// constraints. They are intentionally absent from the public config surface.
	ReasoningModes   []string `yaml:"-"`
	ReasoningEfforts []string `yaml:"-"`
	BaseURL          string   `yaml:"base_url,omitempty"`
	AuthHeader       string   `yaml:"auth_header,omitempty"`
	AuthPrefix       string   `yaml:"auth_prefix,omitempty"`
	// AuthPrefixSet distinguishes an omitted override from an explicit empty
	// prefix after canonical config has been materialized.
	AuthPrefixSet bool              `yaml:"-"`
	ExtraHeaders  map[string]string `yaml:"extra_headers,omitempty"`
	APIVersion    string            `yaml:"api_version,omitempty"`
	ChatPath      string            `yaml:"chat_path,omitempty"`
	// TrustedGateway selects the request's trusted gateway credential as this
	// backend's credential; static keys and per-user provider keys are not
	// consulted for it.
	TrustedGateway bool `yaml:"trusted_gateway,omitempty"`
}

type ModelPricing struct {
	Currency          string   `yaml:"currency,omitempty"`
	PromptPer1M       float64  `yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M   float64  `yaml:"completion_per_1m,omitempty"`
	CachedInputPer1M  float64  `yaml:"cached_input_per_1m,omitempty"`
	CacheWritePer1M   *float64 `yaml:"cache_write_per_1m,omitempty"`
	CacheWrite1hPer1M *float64 `yaml:"cache_write_1h_per_1m,omitempty"`
}

type ModelParams struct {
	Deployment         string              `yaml:"deployment,omitempty"`
	PreferredEndpoints []string            `yaml:"preferred_endpoints,omitempty"`
	Pricing            ModelPricing        `yaml:"pricing,omitempty"`
	Reliability        ProviderReliability `yaml:"reliability,omitempty"`
	ReasoningFamily    string              `yaml:"reasoning_family,omitempty"`
	// AuthoredModel preserves the typed user declaration across materialization.
	// Effective catalog defaults must not leak into exported user YAML, and an
	// api_key_env reference must not be replaced by its expanded secret value.
	AuthoredModel        *CanonicalProviderModel                        `yaml:"-" json:"-"`
	LoRAs                []LoRAAdapter                                  `yaml:"loras,omitempty"`
	AccessKey            string                                         `yaml:"access_key,omitempty" json:"-"`
	AccessKeys           map[string]string                              `yaml:"-" json:"-"`
	Catalog              string                                         `yaml:"catalog,omitempty"`
	ParamSize            string                                         `yaml:"param_size,omitempty"`
	ContextWindowSize    int                                            `yaml:"context_window_size,omitempty"`
	MaxOutputTokens      int                                            `yaml:"max_output_tokens,omitempty"`
	APIFormat            string                                         `yaml:"api_format,omitempty"`
	Description          string                                         `yaml:"description,omitempty"`
	Capabilities         []string                                       `yaml:"capabilities,omitempty"`
	Tags                 []string                                       `yaml:"tags,omitempty"`
	IndexResults         map[string]modelcatalog.IndexResult            `yaml:"-" json:"-"`
	IndexResultsByEffort map[string]map[string]modelcatalog.IndexResult `yaml:"-" json:"-"`
	QualityIndex         string                                         `yaml:"-" json:"-"`
	ExternalModelIDs     map[string]string                              `yaml:"external_model_ids,omitempty"`
	Modality             string                                         `yaml:"modality,omitempty"`
}

// EvidenceScore resolves a versioned static model index. Missing, failed, and
// not-applicable results return ok=false and are never coerced to zero.
func (params ModelParams) EvidenceScore(index string) (float64, bool) {
	result, ok := params.EvidenceResult(index)
	if !ok {
		return 0, false
	}
	return *result.Score, true
}

// EvidenceResult resolves one available preferred-effort index result and
// returns a defensive copy so callers can inspect coverage safely.
func (params ModelParams) EvidenceResult(index string) (modelcatalog.IndexResult, bool) {
	if index == "" {
		index = params.QualityIndex
	}
	result, ok := params.IndexResults[index]
	if !ok || result.Status != "available" || result.Score == nil {
		return modelcatalog.IndexResult{}, false
	}
	return cloneCatalogIndexResult(result), true
}

// EvidenceScoreAt resolves evidence for the exact configured reasoning effort
// when one is present on the candidate. An empty effort uses the model's
// catalog-preferred result. Scores are never borrowed across efforts.
func (params ModelParams) EvidenceScoreAt(index, reasoningEffort string) (float64, bool) {
	result, ok := params.EvidenceResultAt(index, reasoningEffort)
	if !ok {
		return 0, false
	}
	return *result.Score, true
}

// EvidenceResultAt resolves an available index result for the exact configured
// reasoning effort. Results are never borrowed from another effort.
func (params ModelParams) EvidenceResultAt(index, reasoningEffort string) (modelcatalog.IndexResult, bool) {
	if reasoningEffort == "" {
		return params.EvidenceResult(index)
	}
	if index == "" {
		index = params.QualityIndex
	}
	results, ok := params.IndexResultsByEffort[reasoningEffort]
	if !ok {
		return modelcatalog.IndexResult{}, false
	}
	result, ok := results[index]
	if !ok || result.Status != "available" || result.Score == nil {
		return modelcatalog.IndexResult{}, false
	}
	return cloneCatalogIndexResult(result), true
}

type LoRAAdapter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type ReasoningFamilyConfig struct {
	Type                string            `yaml:"type"`
	Parameter           string            `yaml:"parameter"`
	ActivationParameter string            `yaml:"activation_parameter,omitempty"`
	EffortFlags         map[string]string `yaml:"effort_flags,omitempty"`
	Levels              []string          `yaml:"levels,omitempty"`
	Default             string            `yaml:"default,omitempty"`
	Modes               []string          `yaml:"modes,omitempty"`
	DefaultMode         string            `yaml:"default_mode,omitempty"`
	Disabled            string            `yaml:"disabled,omitempty"`
}

const (
	PIITypeAge             = "AGE"
	PIITypeCreditCard      = "CREDIT_CARD"
	PIITypeDateTime        = "DATE_TIME"
	PIITypeDomainName      = "DOMAIN_NAME"
	PIITypeEmailAddress    = "EMAIL_ADDRESS"
	PIITypeGPE             = "GPE"
	PIITypeIBANCode        = "IBAN_CODE"
	PIITypeIPAddress       = "IP_ADDRESS"
	PIITypeNoPII           = "NO_PII"
	PIITypeNRP             = "NRP"
	PIITypeOrganization    = "ORGANIZATION"
	PIITypePerson          = "PERSON"
	PIITypePhoneNumber     = "PHONE_NUMBER"
	PIITypeStreetAddress   = "STREET_ADDRESS"
	PIITypeUSDriverLicense = "US_DRIVER_LICENSE"
	PIITypeUSSSN           = "US_SSN"
	PIITypeZipCode         = "ZIP_CODE"
)

// FindExternalModelByRole searches for an external model configuration by its role.
func (cfg *RouterConfig) FindExternalModelByRole(role string) *ExternalModelConfig {
	for i := range cfg.ExternalModels {
		if cfg.ExternalModels[i].ModelRole == role {
			return &cfg.ExternalModels[i]
		}
	}
	return nil
}

func (cfg *RouterConfig) FindExternalModelByName(name string) *ExternalModelConfig {
	for i := range cfg.ExternalModels {
		if cfg.ExternalModels[i].Name == name {
			return &cfg.ExternalModels[i]
		}
	}
	return nil
}
