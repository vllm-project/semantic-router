// Package catalog owns the typed built-in model-support registry and compiles
// operator bindings into one immutable effective view.
package catalog

type CatalogHeader struct {
	CatalogVersion           string   `json:"catalog_version"`
	Channel                  string   `json:"channel"`
	DefaultModel             string   `json:"default_model"`
	EnabledModels            []string `json:"enabled_models"`
	DefaultIntelligenceIndex string   `json:"default_intelligence_index"`
}

type ProtocolOperation struct {
	ID     string `json:"id"`
	Method string `json:"method"`
	Path   string `json:"path"`
}

type ProtocolDefinition struct {
	ID              string              `json:"id"`
	DisplayName     string              `json:"display_name"`
	WireFormat      string              `json:"wire_format"`
	DefaultBasePath string              `json:"default_base_path"`
	Operations      []ProtocolOperation `json:"operations"`
	Capabilities    []string            `json:"capabilities"`
}

type ProviderAuth struct {
	Strategy       string `json:"strategy"`
	Header         string `json:"header"`
	Prefix         string `json:"prefix"`
	InjectedHeader string `json:"injected_header,omitempty"`
}

type ProviderPresentation struct {
	Logo       string `json:"logo" yaml:"logo"`
	Monogram   string `json:"monogram" yaml:"monogram"`
	Monochrome bool   `json:"monochrome" yaml:"monochrome"`
}

type Conformance struct {
	Status     string `json:"status"`
	VerifiedAt string `json:"verified_at,omitempty"`
}

// CatalogModelBinding is a provider-owned mapping from one canonical model
// card to the provider-native identifier exposed by that API or runtime.
type CatalogModelBinding struct {
	Catalog            string                     `json:"catalog"`
	ID                 string                     `json:"id"`
	Protocols          []string                   `json:"protocols"`
	ReasoningTransport ReasoningTransport         `json:"reasoning_transport,omitempty"`
	Pricing            Pricing                    `json:"pricing,omitempty"`
	Restrictions       map[string]any             `json:"restrictions,omitempty"`
	Lifecycle          string                     `json:"lifecycle,omitempty"`
	Verification       CatalogBindingVerification `json:"verification"`
}

type ReasoningTransport string

const (
	ReasoningTransportChatTemplate     ReasoningTransport = "chat_template_kwargs"
	ReasoningTransportTopLevelEffort   ReasoningTransport = "top_level_effort"
	ReasoningTransportTopLevelBoolean  ReasoningTransport = "top_level_boolean"
	ReasoningTransportReasoningObject  ReasoningTransport = "reasoning_object"
	ReasoningTransportThinkingObject   ReasoningTransport = "thinking_object"
	ReasoningTransportDeepSeekThinking ReasoningTransport = "deepseek_thinking"
)

type ProviderDefinition struct {
	ID                  string                `json:"id"`
	DisplayName         string                `json:"display_name"`
	Description         string                `json:"description"`
	Category            string                `json:"category"`
	SupportTier         string                `json:"support_tier"`
	DefaultBaseURL      string                `json:"default_base_url,omitempty"`
	Protocols           []string              `json:"protocols"`
	DefaultProtocol     string                `json:"default_protocol"`
	SupportedOperations []string              `json:"supported_operations"`
	PathOverrides       map[string]string     `json:"path_overrides,omitempty"`
	DefaultHeaders      map[string]string     `json:"default_headers,omitempty"`
	ReasoningTransport  ReasoningTransport    `json:"reasoning_transport,omitempty"`
	APIVersionQuery     bool                  `json:"api_version_query,omitempty"`
	Auth                ProviderAuth          `json:"auth"`
	Presentation        ProviderPresentation  `json:"presentation"`
	Conformance         Conformance           `json:"conformance"`
	Models              []CatalogModelBinding `json:"models,omitempty"`
}

// CredentialsRef keeps secrets out of the catalog and lets configuration name
// either an environment variable or an already-expanded value. APIKey is
// runtime-only and is omitted from JSON projections.
type CredentialsRef struct {
	APIKey    string `json:"-" yaml:"api_key,omitempty"`
	APIKeyEnv string `json:"api_key_env,omitempty" yaml:"api_key_env,omitempty"`
}

// Endpoint describes one interchangeable network target in a provider
// instance. URL includes the transport scheme and optional path prefix.
type Endpoint struct {
	Name   string `json:"name,omitempty" yaml:"name,omitempty"`
	URL    string `json:"url" yaml:"url"`
	Weight int    `json:"weight,omitempty" yaml:"weight,omitempty"`
}

// ProviderInstance is an operator-owned connection to a catalog provider.
// Its local Name is intentionally independent from the provider Catalog ID.
type ProviderInstance struct {
	Name        string            `json:"name" yaml:"name"`
	Catalog     string            `json:"catalog" yaml:"catalog"`
	BaseURL     string            `json:"base_url,omitempty" yaml:"base_url,omitempty"`
	Endpoints   []Endpoint        `json:"endpoints,omitempty" yaml:"endpoints,omitempty"`
	Credentials CredentialsRef    `json:"credentials,omitempty" yaml:"credentials,omitempty"`
	Headers     map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
	APIVersion  string            `json:"api_version,omitempty" yaml:"api_version,omitempty"`
	AuthHeader  string            `json:"auth_header,omitempty" yaml:"auth_header,omitempty"`
	AuthPrefix  string            `json:"auth_prefix,omitempty" yaml:"auth_prefix,omitempty"`
	ChatPath    string            `json:"chat_path,omitempty" yaml:"chat_path,omitempty"`
}

// ModelProviderBinding binds a request-facing model alias to one provider
// instance. Multiple bindings are explicit fallbacks, not an implicit join.
type ModelProviderBinding struct {
	Name             string            `json:"name" yaml:"name"`
	ModelID          string            `json:"model_id,omitempty" yaml:"model_id,omitempty"`
	Protocol         string            `json:"protocol,omitempty" yaml:"protocol,omitempty"`
	Pricing          Pricing           `json:"pricing,omitempty" yaml:"pricing,omitempty"`
	Reliability      Reliability       `json:"reliability,omitempty" yaml:"reliability,omitempty"`
	ExternalModelIDs map[string]string `json:"external_model_ids,omitempty" yaml:"external_model_ids,omitempty"`
}

// ModelAlias is the small user-facing routing identity. Catalog points to a
// canonical ModelCard; Name is the alias accepted in requests and decisions.
type ModelAlias struct {
	Name      string                 `json:"name" yaml:"name"`
	Catalog   string                 `json:"catalog" yaml:"catalog"`
	Providers []ModelProviderBinding `json:"providers" yaml:"providers"`
	// BindingDefaults preserves provider-model metadata even when an alias has
	// no live backend binding (for example an offline pricing catalog entry).
	// It is an internal materialization input, not a user-facing config block.
	BindingDefaults ModelProviderBinding `json:"-" yaml:"-"`
}

type ReasoningFamilyDefinition struct {
	ID        string   `json:"id" yaml:"name"`
	Type      string   `json:"type" yaml:"type"`
	Parameter string   `json:"parameter" yaml:"parameter"`
	Levels    []string `json:"levels" yaml:"levels,omitempty"`
	Default   string   `json:"default" yaml:"default,omitempty"`
	// Disabled is the provider/model-native level that explicitly turns
	// reasoning off. It is empty for always-reasoning families.
	Disabled string `json:"disabled,omitempty" yaml:"disabled,omitempty"`
}

type Modalities struct {
	Input  []string `json:"input" yaml:"input,omitempty"`
	Output []string `json:"output" yaml:"output,omitempty"`
}

type ModelLimits struct {
	ContextWindowSize int `json:"context_window_size,omitempty" yaml:"context_window_size,omitempty"`
	MaxOutputTokens   int `json:"max_output_tokens,omitempty" yaml:"max_output_tokens,omitempty"`
}

type LoRAAdapter struct {
	Name        string `json:"name" yaml:"name"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

type ModelVerification struct {
	Authority   string `json:"authority" yaml:"authority"`
	Status      string `json:"status" yaml:"status"`
	VerifiedAt  string `json:"verified_at" yaml:"verified_at,omitempty"`
	Source      string `json:"source,omitempty" yaml:"source,omitempty"`
	AssetSHA256 string `json:"asset_sha256,omitempty" yaml:"asset_sha256,omitempty"`
}

type ModelDistribution struct {
	Type    string `json:"type" yaml:"type"`
	Source  string `json:"source" yaml:"source"`
	License string `json:"license,omitempty" yaml:"license,omitempty"`
}

type ModelRole struct {
	Name              string   `json:"name"`
	Required          bool     `json:"required"`
	MinimumCandidates int      `json:"minimum_candidates"`
	Traits            []string `json:"traits"`
	RecommendedPool   []string `json:"recommended_pool"`
}

type ModelCard struct {
	ID              string               `json:"id"`
	DisplayName     string               `json:"display_name"`
	Description     string               `json:"description"`
	Kind            string               `json:"kind"`
	Publisher       string               `json:"publisher"`
	Presentation    ProviderPresentation `json:"presentation"`
	Distribution    ModelDistribution    `json:"distribution"`
	Family          string               `json:"family"`
	ParameterSize   string               `json:"parameter_size,omitempty"`
	Revision        string               `json:"revision,omitempty"`
	ReleasedAt      string               `json:"released_at,omitempty"`
	KnowledgeCutoff string               `json:"knowledge_cutoff,omitempty"`
	Lifecycle       string               `json:"lifecycle"`
	Limits          ModelLimits          `json:"limits,omitempty"`
	Capabilities    []string             `json:"capabilities"`
	Modalities      Modalities           `json:"modalities"`
	ReasoningFamily string               `json:"reasoning_family,omitempty"`
	Tags            []string             `json:"tags,omitempty"`
	Generation      int                  `json:"generation,omitempty"`
	PolicyVersion   string               `json:"policy_version,omitempty"`
	Asset           string               `json:"asset,omitempty"`
	Entrypoint      string               `json:"entrypoint,omitempty"`
	Recipe          string               `json:"recipe,omitempty"`
	Traits          []string             `json:"traits,omitempty"`
	Roles           []ModelRole          `json:"roles,omitempty"`
	Verification    ModelVerification    `json:"verification"`
}

// ModelCardOverlay is a presence-aware handwritten card or built-in override.
// Pointer fields distinguish an intentional zero/empty override from omission.
type ModelCardOverlay struct {
	Name string `json:"name" yaml:"name"`
	// BuiltIn is an internal presence-aware binding hint. A nil value keeps the
	// catalog package's direct-call behavior; config adapters set it explicitly
	// so a custom alias that happens to equal a built-in ID stays custom.
	BuiltIn           *bool                      `json:"-" yaml:"-"`
	DisplayName       *string                    `json:"display_name,omitempty" yaml:"display_name,omitempty"`
	Description       *string                    `json:"description,omitempty" yaml:"description,omitempty"`
	Publisher         *string                    `json:"publisher,omitempty" yaml:"publisher,omitempty"`
	Presentation      *ProviderPresentation      `json:"presentation,omitempty" yaml:"presentation,omitempty"`
	Distribution      *ModelDistribution         `json:"distribution,omitempty" yaml:"distribution,omitempty"`
	Family            *string                    `json:"family,omitempty" yaml:"family,omitempty"`
	ParameterSize     *string                    `json:"parameter_size,omitempty" yaml:"parameter_size,omitempty"`
	Revision          *string                    `json:"revision,omitempty" yaml:"revision,omitempty"`
	ReleasedAt        *string                    `json:"released_at,omitempty" yaml:"released_at,omitempty"`
	KnowledgeCutoff   *string                    `json:"knowledge_cutoff,omitempty" yaml:"knowledge_cutoff,omitempty"`
	Lifecycle         *string                    `json:"lifecycle,omitempty" yaml:"lifecycle,omitempty"`
	ContextWindowSize *int                       `json:"context_window_size,omitempty" yaml:"context_window_size,omitempty"`
	MaxOutputTokens   *int                       `json:"max_output_tokens,omitempty" yaml:"max_output_tokens,omitempty"`
	Capabilities      *[]string                  `json:"capabilities,omitempty" yaml:"capabilities,omitempty"`
	Modalities        *Modalities                `json:"modalities,omitempty" yaml:"modalities,omitempty"`
	ReasoningFamily   *string                    `json:"reasoning_family,omitempty" yaml:"reasoning_family,omitempty"`
	Reasoning         *ReasoningFamilyDefinition `json:"reasoning,omitempty" yaml:"reasoning,omitempty"`
	Tags              *[]string                  `json:"tags,omitempty" yaml:"tags,omitempty"`
	LoRAs             *[]LoRAAdapter             `json:"loras,omitempty" yaml:"loras,omitempty"`
	Evaluations       []UserEvaluation           `json:"evaluations,omitempty" yaml:"evaluations,omitempty"`
	Verification      *ModelVerification         `json:"verification,omitempty" yaml:"verification,omitempty"`
	// RuntimeModality preserves the existing router-specific ar/diffusion/omni
	// classification while canonical model facts use input/output modalities.
	RuntimeModality *string `json:"-" yaml:"-"`
}

// UserEvaluation is the intentionally small operator-facing measurement
// surface. Repository benchmark definitions own metric semantics; provenance
// and verification are assigned internally when this is materialized.
type UserEvaluation struct {
	Benchmark        string             `json:"benchmark" yaml:"benchmark"`
	BenchmarkProfile string             `json:"benchmark_profile,omitempty" yaml:"benchmark_profile,omitempty"`
	ReasoningEffort  string             `json:"reasoning_effort,omitempty" yaml:"reasoning_effort,omitempty"`
	Metrics          map[string]float64 `json:"metrics" yaml:"metrics"`
	Source           string             `json:"source,omitempty" yaml:"source,omitempty"`
	MeasuredAt       string             `json:"measured_at,omitempty" yaml:"measured_at,omitempty"`
	Metadata         map[string]any     `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// EvaluationConfig is the optional operator extension surface.
type EvaluationConfig struct {
	Benchmarks []BenchmarkDefinition `json:"benchmarks,omitempty" yaml:"benchmarks,omitempty"`
	Records    []EvaluationRecord    `json:"records,omitempty" yaml:"records,omitempty"`
	Indices    []IndexDefinition     `json:"indices,omitempty" yaml:"indices,omitempty"`
}

// Defaults selects model and evidence behavior without exposing catalog build
// version or digest in ordinary configuration.
type Defaults struct {
	Model           string `json:"model,omitempty" yaml:"model,omitempty"`
	ReasoningEffort string `json:"reasoning_effort,omitempty" yaml:"reasoning_effort,omitempty"`
	// QualityIndex is release-owned runtime metadata. It is deliberately not a
	// user-facing YAML setting.
	QualityIndex string `json:"quality_index,omitempty" yaml:"-"`
}

type Pricing struct {
	Currency         string   `json:"currency,omitempty" yaml:"currency,omitempty"`
	PromptPer1M      float64  `json:"prompt_per_1m,omitempty" yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M  float64  `json:"completion_per_1m,omitempty" yaml:"completion_per_1m,omitempty"`
	CachedInputPer1M float64  `json:"cached_input_per_1m,omitempty" yaml:"cached_input_per_1m,omitempty"`
	CacheWritePer1M  *float64 `json:"cache_write_per_1m,omitempty" yaml:"cache_write_per_1m,omitempty"`
}

type Reliability struct {
	LBPolicy            string `json:"lb_policy,omitempty" yaml:"lb_policy,omitempty"`
	RetryCount          int    `json:"retry_count,omitempty" yaml:"retry_count,omitempty"`
	RetryOn             string `json:"retry_on,omitempty" yaml:"retry_on,omitempty"`
	Consecutive5xx      int    `json:"consecutive_5xx,omitempty" yaml:"consecutive_5xx,omitempty"`
	BaseEjectionTime    string `json:"base_ejection_time,omitempty" yaml:"base_ejection_time,omitempty"`
	MaxEjectionPercent  int    `json:"max_ejection_percent,omitempty" yaml:"max_ejection_percent,omitempty"`
	HealthCheckPath     string `json:"health_check_path,omitempty" yaml:"health_check_path,omitempty"`
	HealthCheckInterval string `json:"health_check_interval,omitempty" yaml:"health_check_interval,omitempty"`
	HealthCheckTimeout  string `json:"health_check_timeout,omitempty" yaml:"health_check_timeout,omitempty"`
}

type CatalogBindingVerification struct {
	Status     string `json:"status" yaml:"status,omitempty"`
	VerifiedAt string `json:"verified_at,omitempty" yaml:"verified_at,omitempty"`
	Source     string `json:"source,omitempty" yaml:"source,omitempty"`
}

type BenchmarkProfile struct {
	ID          string `json:"id" yaml:"id"`
	DisplayName string `json:"display_name" yaml:"display_name"`
	Description string `json:"description" yaml:"description"`
}

type BenchmarkMetric struct {
	ID        string     `json:"id" yaml:"id"`
	Unit      string     `json:"unit" yaml:"unit"`
	Direction string     `json:"direction" yaml:"direction"`
	Range     [2]float64 `json:"range" yaml:"range"`
}

type BenchmarkDefinition struct {
	ID             string             `json:"id" yaml:"id"`
	DisplayName    string             `json:"display_name" yaml:"display_name"`
	Domain         string             `json:"domain" yaml:"domain"`
	Source         string             `json:"source,omitempty" yaml:"source,omitempty"`
	DefaultProfile string             `json:"default_profile" yaml:"default_profile"`
	Profiles       []BenchmarkProfile `json:"profiles" yaml:"profiles"`
	Metrics        []BenchmarkMetric  `json:"metrics" yaml:"metrics"`
}

type EvaluationEvidence struct {
	Provenance      string `json:"provenance" yaml:"provenance"`
	Verification    string `json:"verification" yaml:"verification"`
	Source          string `json:"source,omitempty" yaml:"source,omitempty"`
	Artifact        string `json:"artifact,omitempty" yaml:"artifact,omitempty"`
	Redistributable bool   `json:"redistributable" yaml:"redistributable"`
}

// EvaluationSubject records any material property of the measured subject.
// Keys are intentionally open so benchmark-specific harness, prompt, tool,
// runtime, and model-variant metadata survives every generated projection.
type EvaluationSubject map[string]any

type EvaluationRecord struct {
	ID               string             `json:"id" yaml:"id"`
	Model            string             `json:"model" yaml:"model"`
	Benchmark        string             `json:"benchmark" yaml:"benchmark"`
	BenchmarkProfile string             `json:"benchmark_profile" yaml:"benchmark_profile"`
	ReasoningEffort  string             `json:"reasoning_effort" yaml:"reasoning_effort"`
	Subject          EvaluationSubject  `json:"subject,omitempty" yaml:"subject,omitempty"`
	Metrics          map[string]float64 `json:"metrics" yaml:"metrics"`
	Status           string             `json:"status" yaml:"status"`
	MeasuredAt       string             `json:"measured_at,omitempty" yaml:"measured_at,omitempty"`
	Evidence         EvaluationEvidence `json:"evidence" yaml:"evidence"`
}

type EvaluationCoverage struct {
	Model            string   `json:"model"`
	ReasoningEffort  string   `json:"reasoning_effort"`
	Benchmark        string   `json:"benchmark"`
	BenchmarkProfile string   `json:"benchmark_profile"`
	Metric           string   `json:"metric"`
	Status           string   `json:"status"`
	Value            *float64 `json:"value,omitempty"`
	Evaluation       string   `json:"evaluation,omitempty"`
}

type NormalizationPoint struct {
	Input  float64 `json:"input" yaml:"input"`
	Output float64 `json:"output" yaml:"output"`
}

type Normalization struct {
	Type   string               `json:"type" yaml:"type"`
	Min    *float64             `json:"min,omitempty" yaml:"min,omitempty"`
	Max    *float64             `json:"max,omitempty" yaml:"max,omitempty"`
	K      *float64             `json:"k,omitempty" yaml:"k,omitempty"`
	X0     *float64             `json:"x0,omitempty" yaml:"x0,omitempty"`
	Points []NormalizationPoint `json:"points,omitempty" yaml:"points,omitempty"`
	Values map[string]float64   `json:"values,omitempty" yaml:"values,omitempty"`
}

type IndexComponent struct {
	Benchmark        string        `json:"benchmark,omitempty" yaml:"benchmark,omitempty"`
	Metric           string        `json:"metric,omitempty" yaml:"metric,omitempty"`
	BenchmarkProfile string        `json:"benchmark_profile,omitempty" yaml:"benchmark_profile,omitempty"`
	Index            string        `json:"index,omitempty" yaml:"index,omitempty"`
	Weight           float64       `json:"weight" yaml:"weight"`
	Normalization    Normalization `json:"normalization,omitempty" yaml:"normalization,omitempty"`
}

type MissingPolicy struct {
	Policy  string  `json:"policy" yaml:"policy"`
	Minimum float64 `json:"minimum,omitempty" yaml:"minimum,omitempty"`
}

type IndexDefinition struct {
	ID          string             `json:"id" yaml:"id"`
	DisplayName string             `json:"display_name" yaml:"display_name"`
	Description string             `json:"description,omitempty" yaml:"description,omitempty"`
	Methodology string             `json:"methodology,omitempty" yaml:"methodology,omitempty"`
	Aggregation string             `json:"aggregation" yaml:"aggregation"`
	Scale       [2]float64         `json:"scale" yaml:"scale"`
	Missing     MissingPolicy      `json:"missing" yaml:"missing"`
	Domains     map[string]float64 `json:"domains,omitempty" yaml:"domains,omitempty"`
	Components  []IndexComponent   `json:"components" yaml:"components"`
}

type IndexComponentResult struct {
	Benchmark        string   `json:"benchmark,omitempty"`
	Metric           string   `json:"metric,omitempty"`
	BenchmarkProfile string   `json:"benchmark_profile,omitempty"`
	Index            string   `json:"index,omitempty"`
	Evaluation       string   `json:"evaluation,omitempty"`
	Weight           float64  `json:"weight"`
	Status           string   `json:"status"`
	Value            *float64 `json:"value,omitempty"`
	Normalized       *float64 `json:"normalized,omitempty"`
}

type IndexResult struct {
	Model           string                 `json:"model"`
	ReasoningEffort string                 `json:"reasoning_effort"`
	Index           string                 `json:"index"`
	Status          string                 `json:"status"`
	Score           *float64               `json:"score"`
	Coverage        float64                `json:"coverage"`
	Components      []IndexComponentResult `json:"components"`
	Domains         map[string]float64     `json:"domains,omitempty"`
	Provenance      []string               `json:"provenance"`
}

// FieldProvenance records whether an effective field came from the repository
// catalog or an operator override.
type FieldProvenance map[string]string

type EffectiveModelCard struct {
	Card            ModelCard        `json:"card"`
	LoRAs           []LoRAAdapter    `json:"loras,omitempty"`
	Evaluations     []UserEvaluation `json:"evaluations,omitempty"`
	Provenance      FieldProvenance  `json:"provenance"`
	RuntimeModality string           `json:"runtime_modality,omitempty"`
}

type EffectiveProvider struct {
	Instance   ProviderInstance   `json:"instance"`
	Definition ProviderDefinition `json:"definition"`
}

type EffectiveModel struct {
	Alias           string                            `json:"alias"`
	Catalog         string                            `json:"catalog"`
	Card            EffectiveModelCard                `json:"card"`
	Providers       []EffectiveModelProvider          `json:"providers"`
	Indices         map[string]IndexResult            `json:"indices,omitempty"`
	IndicesByEffort map[string]map[string]IndexResult `json:"indices_by_effort,omitempty"`
	BindingDefaults ModelProviderBinding              `json:"-"`
}

type EffectiveModelProvider struct {
	Binding        ModelProviderBinding `json:"binding"`
	Provider       EffectiveProvider    `json:"provider"`
	CatalogBinding *CatalogModelBinding `json:"catalog_binding,omitempty"`
}

// CompileInput contains only user-owned extensions and bindings. Repository
// catalog metadata is supplied separately through Registry.
type CompileInput struct {
	Defaults          Defaults                    `json:"defaults,omitempty" yaml:"defaults,omitempty"`
	Providers         []ProviderInstance          `json:"providers,omitempty" yaml:"providers,omitempty"`
	Models            []ModelAlias                `json:"models,omitempty" yaml:"models,omitempty"`
	ModelCards        []ModelCardOverlay          `json:"model_cards,omitempty" yaml:"model_cards,omitempty"`
	ReasoningFamilies []ReasoningFamilyDefinition `json:"reasoning_families,omitempty" yaml:"reasoning_families,omitempty"`
	Evaluations       EvaluationConfig            `json:"evaluations,omitempty" yaml:"evaluations,omitempty"`
}

type snapshot struct {
	SchemaVersion      string                      `json:"schema_version"`
	Catalogs           []CatalogHeader             `json:"catalogs"`
	Protocols          []ProtocolDefinition        `json:"protocols"`
	Providers          []ProviderDefinition        `json:"providers"`
	ReasoningFamilies  []ReasoningFamilyDefinition `json:"reasoning_families"`
	Models             []ModelCard                 `json:"models"`
	Benchmarks         []BenchmarkDefinition       `json:"benchmarks"`
	Evaluations        []EvaluationRecord          `json:"evaluations"`
	EvaluationCoverage []EvaluationCoverage        `json:"evaluation_coverage"`
	Indices            []IndexDefinition           `json:"indices"`
	IndexResults       []IndexResult               `json:"index_results"`
}
