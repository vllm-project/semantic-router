package config

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// Classifier represents the configuration for text classification.
type Classifier struct {
	CategoryModel    `yaml:"category_model"`
	MCPCategoryModel `yaml:"mcp_category_model,omitempty"`
	PIIModel         `yaml:"pii_model"`
	PreferenceModel  PreferenceModelConfig `yaml:"preference_model,omitempty"`
}

type BertModel struct {
	ModelID   string  `yaml:"model_id"`
	Threshold float32 `yaml:"threshold"`
	UseCPU    bool    `yaml:"use_cpu"`
}

type CategoryModel struct {
	// MaxSequenceLength bounds a complete text with Window, otherwise one
	// inference. Implicit local Vela zero/nil resolves to a 32K document scan
	// with 512-token forwards during owned preparation.
	MaxSequenceLength int `yaml:"max_sequence_length,omitempty"`
	// Enabled turns category classification on or off explicitly. Nil keeps the
	// historical behaviour of running whenever a model is configured.
	Enabled   *bool   `yaml:"enabled,omitempty"`
	ModelID   string  `yaml:"model_id"`
	Threshold float32 `yaml:"threshold"`
	UseCPU    bool    `yaml:"use_cpu"`
	// Backend attaches a named remote classifier. Its absence preserves local
	// category inference exactly as before.
	Backend             *RemoteClassifierBackend `yaml:"backend,omitempty"`
	CategoryMappingPath string                   `yaml:"category_mapping_path"`
	FallbackCategory    string                   `yaml:"fallback_category,omitempty"`
}

type PIIModel struct {
	Window *SequenceHeadWindowConfig `yaml:"window,omitempty"`
	// MaxSequenceLength bounds a complete text with Window, otherwise one
	// inference. Implicit local Vela zero/nil resolves to a 32K document scan
	// with 512-token forwards during owned preparation.
	MaxSequenceLength int `yaml:"max_sequence_length,omitempty"`
	// Enabled turns PII classification on or off explicitly. Nil keeps the
	// historical behaviour of running whenever a model is configured.
	Enabled        *bool   `yaml:"enabled,omitempty"`
	ModelID        string  `yaml:"model_id"`
	Threshold      float32 `yaml:"threshold"`
	UseCPU         bool    `yaml:"use_cpu"`
	PIIMappingPath string  `yaml:"pii_mapping_path"`
	// Backend attaches a named remote token classifier speaking token_spans.v1.
	// Its absence preserves local PII inference exactly as before.
	Backend *RemoteClassifierBackend `yaml:"backend,omitempty"`

	// ClassifierOnErrorConfig contributes OnError (allow|block). With block, a
	// PII rule whose content could not be fully classified (backend error, or a
	// provider that declared truncated_at) matches as classification_error
	// instead of reading as clean.
	ClassifierOnErrorConfig `yaml:",inline"`
	// UnscannedConfig contributes OnUnscanned (block|allow): content the model
	// did not read in full matches as unscanned unless it is allow.
	UnscannedConfig `yaml:",inline"`
}

type EmbeddingModels struct {
	Qwen3ModelPath      string                  `yaml:"qwen3_model_path"`
	MmBertModelPath     string                  `yaml:"mmbert_model_path"`
	MultiModalModelPath string                  `yaml:"multimodal_model_path,omitempty"`
	UseCPU              bool                    `yaml:"use_cpu"`
	EmbeddingConfig     HNSWConfig              `yaml:"embedding_config,omitempty"`
	Endpoint            EmbeddingEndpointConfig `yaml:"endpoint,omitempty"`
}

func (e EmbeddingModels) MinSimilarityThreshold() float32 {
	return e.EmbeddingConfig.WithDefaults().MinScoreThreshold
}

// HNSWConfig contains settings for optimizing embedding-backed classification.
type HNSWConfig struct {
	// FullContext sends complete routing text to mmBERT up to its model capacity.
	// False retains bounded representative sampling for routing latency.
	FullContext       bool   `yaml:"full_context,omitempty"`
	Backend           string `yaml:"backend,omitempty"`
	ModelType         string `yaml:"model_type,omitempty"`
	PreloadEmbeddings bool   `yaml:"preload_embeddings"`
	TargetDimension   int    `yaml:"target_dimension,omitempty"`
	TargetLayer       int    `yaml:"target_layer,omitempty"`
	// EnableSoftMatching allows below-threshold matches when no rule meets its
	// threshold. This ranked fallback is opt-in; routing predicates default to
	// the threshold declared by each rule.
	EnableSoftMatching *bool `yaml:"enable_soft_matching,omitempty"`
	// TopK limits emitted embedding matches only when positive. The default 0
	// preserves all accepted predicates for projections and decision priority.
	TopK              *int                   `yaml:"top_k,omitempty"`
	MinScoreThreshold float32                `yaml:"min_score_threshold,omitempty"`
	PrototypeScoring  PrototypeScoringConfig `yaml:"prototype_scoring,omitempty"`
}

func (c HNSWConfig) WithDefaults() HNSWConfig {
	result := c
	if result.Backend == "" {
		result.Backend = EmbeddingBackendModelRuntime
	}
	if result.ModelType == "" {
		if normalizeEmbeddingBackend(result.Backend) == EmbeddingBackendOpenAICompatible {
			result.ModelType = EmbeddingModelTypeRemote
		} else {
			result.ModelType = EmbeddingModelTypeQwen3
		}
	}
	// Zero selects the prepared model's native dimension. The provider validates
	// explicit dimensions; a shared default must not resize another model.
	if result.EnableSoftMatching == nil {
		defaultEnabled := false
		result.EnableSoftMatching = &defaultEnabled
	}
	if result.TopK == nil || *result.TopK < 0 {
		defaultTopK := 0
		result.TopK = &defaultTopK
	}
	if result.MinScoreThreshold <= 0 {
		result.MinScoreThreshold = 0.5
	}
	result.PrototypeScoring = result.PrototypeScoring.WithDefaults()
	return result
}

type MCPCategoryModel struct {
	Enabled          bool              `yaml:"enabled"`
	TransportType    string            `yaml:"transport_type"`
	Command          string            `yaml:"command,omitempty"`
	Args             []string          `yaml:"args,omitempty"`
	Env              map[string]string `yaml:"env,omitempty"`
	URL              string            `yaml:"url,omitempty"`
	ToolName         string            `yaml:"tool_name,omitempty"`
	Threshold        float32           `yaml:"threshold"`
	TimeoutSeconds   int               `yaml:"timeout_seconds,omitempty"`
	MaxResponseBytes int64             `yaml:"max_response_bytes,omitempty"`
}

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

type PromptGuardConfig struct {
	// MaxSequenceLength bounds a complete text with Window, otherwise one
	// inference. Implicit local Vela zero/nil resolves to a 32K document scan
	// with 512-token forwards during owned preparation.
	MaxSequenceLength    int                       `yaml:"max_sequence_length,omitempty"`
	Backend              *RemoteClassifierBackend  `yaml:"backend,omitempty"`
	Window               *SequenceHeadWindowConfig `yaml:"window,omitempty"`
	Enabled              bool                      `yaml:"enabled"`
	ModelID              string                    `yaml:"model_id"`
	Threshold            float32                   `yaml:"threshold"`
	UseCPU               bool                      `yaml:"use_cpu"`
	JailbreakMappingPath string                    `yaml:"jailbreak_mapping_path"`
	PositiveLabels       []string                  `yaml:"positive_labels,omitempty"`

	// ClassifierOnErrorConfig contributes OnError (allow|block), shared with
	// every other pluggable classifier backend instead of being redeclared
	// per struct.
	ClassifierOnErrorConfig `yaml:",inline"`
	// UnscannedConfig contributes OnUnscanned (block|allow): content the guard
	// did not read in full matches as unscanned unless it is allow.
	UnscannedConfig `yaml:",inline"`
}

type FeedbackDetectorConfig struct {
	// MaxSequenceLength bounds a complete text with Window, otherwise one
	// inference. Implicit local Vela zero/nil resolves to a 32K document scan
	// with 512-token forwards during owned preparation.
	MaxSequenceLength   int     `yaml:"max_sequence_length,omitempty"`
	Enabled             bool    `yaml:"enabled"`
	ModelID             string  `yaml:"model_id"`
	Threshold           float32 `yaml:"threshold"`
	UseCPU              bool    `yaml:"use_cpu"`
	FeedbackMappingPath string  `yaml:"feedback_mapping_path"`
}

type PreferenceModelConfig struct {
	UseContrastive   *bool                  `yaml:"use_contrastive,omitempty"`
	EmbeddingModel   string                 `yaml:"embedding_model,omitempty"`
	PrototypeScoring PrototypeScoringConfig `yaml:"prototype_scoring,omitempty"`
}

func (c PreferenceModelConfig) WithDefaults() PreferenceModelConfig {
	result := c
	result.PrototypeScoring = result.PrototypeScoring.WithDefaults()
	return result
}

func (c PreferenceModelConfig) ContrastiveEnabled() bool {
	return c.UseContrastive != nil && *c.UseContrastive
}

type ComplexityModelConfig struct {
	PrototypeScoring PrototypeScoringConfig `yaml:"prototype_scoring,omitempty"`
	// Backend attaches a named remote scorer. Its absence preserves local
	// prototype scoring exactly as before. It sits here, beside
	// prototype_scoring, rather than on a rule: routing.signals is replaced
	// wholesale per recipe, so a backend declared there would disappear under
	// any recipe that did not repeat it.
	Backend *RemoteClassifierBackend `yaml:"backend,omitempty"`
}

func (c ComplexityModelConfig) WithDefaults() ComplexityModelConfig {
	result := c
	result.PrototypeScoring = result.PrototypeScoring.WithDefaults()
	return result
}

type ExternalModelConfig struct {
	Name             string                        `yaml:"name,omitempty"`
	Provider         string                        `yaml:"llm_provider"`
	ModelRole        string                        `yaml:"model_role"`
	ModelEndpoint    ClassifierVLLMEndpoint        `yaml:"llm_endpoint,omitempty"`
	ModelName        string                        `yaml:"llm_model_name,omitempty"`
	TimeoutSeconds   int                           `yaml:"llm_timeout_seconds,omitempty"`
	ParserType       string                        `yaml:"parser_type,omitempty"`
	Threshold        float32                       `yaml:"threshold,omitempty"`
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

type ToolFilteringWeights struct {
	Embed    *float32 `json:"embed,omitempty" yaml:"embed,omitempty"`
	Lexical  *float32 `json:"lexical,omitempty" yaml:"lexical,omitempty"`
	Tag      *float32 `json:"tag,omitempty" yaml:"tag,omitempty"`
	Name     *float32 `json:"name,omitempty" yaml:"name,omitempty"`
	Category *float32 `json:"category,omitempty" yaml:"category,omitempty"`
}

type AdvancedToolFilteringConfig struct {
	Enabled                     bool                              `json:"enabled" yaml:"enabled"`
	RetrievalStrategy           string                            `json:"retrieval_strategy,omitempty" yaml:"retrieval_strategy,omitempty"`
	CandidatePoolSize           *int                              `json:"candidate_pool_size,omitempty" yaml:"candidate_pool_size,omitempty"`
	MinLexicalOverlap           *int                              `json:"min_lexical_overlap,omitempty" yaml:"min_lexical_overlap,omitempty"`
	MinCombinedScore            *float32                          `json:"min_combined_score,omitempty" yaml:"min_combined_score,omitempty"`
	Weights                     ToolFilteringWeights              `json:"weights,omitempty" yaml:"weights,omitempty"`
	UseCategoryFilter           *bool                             `json:"use_category_filter,omitempty" yaml:"use_category_filter,omitempty"`
	CategoryConfidenceThreshold *float32                          `json:"category_confidence_threshold,omitempty" yaml:"category_confidence_threshold,omitempty"`
	AllowTools                  []string                          `json:"allow_tools,omitempty" yaml:"allow_tools,omitempty"`
	BlockTools                  []string                          `json:"block_tools,omitempty" yaml:"block_tools,omitempty"`
	HybridHistory               *HybridHistoryToolRetrievalConfig `json:"hybrid_history,omitempty" yaml:"hybrid_history,omitempty"`
}

// HybridHistoryToolRetrievalConfig tunes hybrid_history retrieval (semantic + short history + priors + repetition).
type HybridHistoryToolRetrievalConfig struct {
	HistoryHorizon             *int     `json:"history_horizon,omitempty" yaml:"history_horizon,omitempty"`
	MinHistorySteps            *int     `json:"min_history_steps,omitempty" yaml:"min_history_steps,omitempty"`
	HistoryConfidenceThreshold *float32 `json:"history_confidence_threshold,omitempty" yaml:"history_confidence_threshold,omitempty"`
	WeightSemantic             *float32 `json:"weight_semantic,omitempty" yaml:"weight_semantic,omitempty"`
	WeightHistoryTransition    *float32 `json:"weight_history_transition,omitempty" yaml:"weight_history_transition,omitempty"`
	WeightDecisionPrior        *float32 `json:"weight_decision_prior,omitempty" yaml:"weight_decision_prior,omitempty"`
	RepetitionPenaltyStrength  *float32 `json:"repetition_penalty_strength,omitempty" yaml:"repetition_penalty_strength,omitempty"`
}

type ToolsConfig struct {
	Enabled             bool                         `yaml:"enabled"`
	TopK                int                          `yaml:"top_k"`
	SimilarityThreshold *float32                     `yaml:"similarity_threshold,omitempty"`
	ToolsDBPath         string                       `yaml:"tools_db_path"`
	FallbackToEmpty     bool                         `yaml:"fallback_to_empty"`
	AdvancedFiltering   *AdvancedToolFilteringConfig `yaml:"advanced_filtering,omitempty"`
}

type HallucinationMitigationConfig struct {
	Enabled            bool                     `yaml:"enabled"`
	FactCheckModel     FactCheckModelConfig     `yaml:"fact_check_model"`
	HallucinationModel HallucinationModelConfig `yaml:"hallucination_model"`
}

type FactCheckModelConfig struct {
	// MaxSequenceLength bounds a complete text with Window, otherwise one
	// inference. Implicit local Vela zero/nil resolves to a 32K document scan
	// with 512-token forwards during owned preparation.
	MaxSequenceLength int     `yaml:"max_sequence_length,omitempty"`
	ModelID           string  `yaml:"model_id"`
	Threshold         float32 `yaml:"threshold"`
	UseCPU            bool    `yaml:"use_cpu"`
}

type HallucinationModelConfig struct {
	Backend string `yaml:"backend,omitempty"`
	// Endpoint is the remote detector's base URL, which the endpoint detector
	// derives from its binding's external model; it is not configurable.
	Endpoint           string  `yaml:"-"`
	IncludeExplanation bool    `yaml:"include_explanation,omitempty"`
	ModelID            string  `yaml:"model_id"`
	Threshold          float32 `yaml:"threshold"`
	UseCPU             bool    `yaml:"use_cpu"`
	MinSpanLength      int     `yaml:"min_span_length,omitempty"`
	MinSpanConfidence  float32 `yaml:"min_span_confidence,omitempty"`
	ContextWindowSize  int     `yaml:"context_window_size,omitempty"`
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
}

type ModelPricing struct {
	Currency         string   `yaml:"currency,omitempty"`
	PromptPer1M      float64  `yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M  float64  `yaml:"completion_per_1m,omitempty"`
	CachedInputPer1M float64  `yaml:"cached_input_per_1m,omitempty"`
	CacheWritePer1M  *float64 `yaml:"cache_write_per_1m,omitempty"`
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

type PIIPolicy struct {
	AllowByDefault bool     `yaml:"allow_by_default"`
	PIITypes       []string `yaml:"pii_types_allowed,omitempty"`
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

// moduleActive resolves an explicit enabled flag. Nil means the module was not
// configured either way, so the caller's configuration checks decide.
func moduleActive(enabled *bool) bool { return enabled == nil || *enabled }

// Active reports whether category classification was explicitly disabled.
func (m CategoryModel) Active() bool { return moduleActive(m.Enabled) }

// Active reports whether PII classification was explicitly disabled.
func (m PIIModel) Active() bool { return moduleActive(m.Enabled) }
