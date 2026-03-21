package config

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
	ModelID             string  `yaml:"model_id"`
	Threshold           float32 `yaml:"threshold"`
	UseCPU              bool    `yaml:"use_cpu"`
	UseModernBERT       bool    `yaml:"use_modernbert"`
	UseMmBERT32K        bool    `yaml:"use_mmbert_32k"`
	CategoryMappingPath string  `yaml:"category_mapping_path"`
	FallbackCategory    string  `yaml:"fallback_category,omitempty"`
}

type PIIModel struct {
	ModelID        string  `yaml:"model_id"`
	Threshold      float32 `yaml:"threshold"`
	UseCPU         bool    `yaml:"use_cpu"`
	UseMmBERT32K   bool    `yaml:"use_mmbert_32k"`
	PIIMappingPath string  `yaml:"pii_mapping_path"`
}

type EmbeddingModels struct {
	Qwen3ModelPath      string     `yaml:"qwen3_model_path"`
	GemmaModelPath      string     `yaml:"gemma_model_path"`
	MmBertModelPath     string     `yaml:"mmbert_model_path"`
	MultiModalModelPath string     `yaml:"multimodal_model_path,omitempty"`
	BertModelPath       string     `yaml:"bert_model_path"`
	UseCPU              bool       `yaml:"use_cpu"`
	EmbeddingConfig     HNSWConfig `yaml:"embedding_config,omitempty"`
}

func (e EmbeddingModels) MinSimilarityThreshold() float32 {
	return e.EmbeddingConfig.WithDefaults().MinScoreThreshold
}

// HNSWConfig contains settings for optimizing embedding-backed classification.
type HNSWConfig struct {
	ModelType          string  `yaml:"model_type,omitempty"`
	PreloadEmbeddings  bool    `yaml:"preload_embeddings"`
	TargetDimension    int     `yaml:"target_dimension,omitempty"`
	TargetLayer        int     `yaml:"target_layer,omitempty"`
	EnableSoftMatching *bool   `yaml:"enable_soft_matching,omitempty"`
	TopK               *int    `yaml:"top_k,omitempty"`
	MinScoreThreshold  float32 `yaml:"min_score_threshold,omitempty"`
}

func (c HNSWConfig) WithDefaults() HNSWConfig {
	result := c
	if result.ModelType == "" {
		result.ModelType = "qwen3"
	}
	if result.TargetDimension <= 0 {
		result.TargetDimension = 768
	}
	if result.EnableSoftMatching == nil {
		defaultEnabled := true
		result.EnableSoftMatching = &defaultEnabled
	}
	if result.TopK == nil {
		defaultTopK := 1
		result.TopK = &defaultTopK
	} else if *result.TopK < 0 {
		defaultTopK := 1
		result.TopK = &defaultTopK
	}
	if result.MinScoreThreshold <= 0 {
		result.MinScoreThreshold = 0.5
	}
	return result
}

type MCPCategoryModel struct {
	Enabled        bool              `yaml:"enabled"`
	TransportType  string            `yaml:"transport_type"`
	Command        string            `yaml:"command,omitempty"`
	Args           []string          `yaml:"args,omitempty"`
	Env            map[string]string `yaml:"env,omitempty"`
	URL            string            `yaml:"url,omitempty"`
	ToolName       string            `yaml:"tool_name,omitempty"`
	Threshold      float32           `yaml:"threshold"`
	TimeoutSeconds int               `yaml:"timeout_seconds,omitempty"`
}

// PromptCompressionConfig controls NLP-based prompt compression before signal extraction.
type PromptCompressionConfig struct {
	Enabled        bool     `yaml:"enabled"`
	MaxTokens      int      `yaml:"max_tokens"`
	MinLength      int      `yaml:"min_length,omitempty"`
	SkipSignals    []string `yaml:"skip_signals,omitempty"`
	TextRankWeight float64  `yaml:"textrank_weight,omitempty"`
	PositionWeight float64  `yaml:"position_weight,omitempty"`
	TFIDFWeight    float64  `yaml:"tfidf_weight,omitempty"`
	PositionDepth  float64  `yaml:"position_depth,omitempty"`
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
	Enabled              bool    `yaml:"enabled"`
	ModelID              string  `yaml:"model_id"`
	Threshold            float32 `yaml:"threshold"`
	UseCPU               bool    `yaml:"use_cpu"`
	UseModernBERT        bool    `yaml:"use_modernbert"`
	UseMmBERT32K         bool    `yaml:"use_mmbert_32k"`
	JailbreakMappingPath string  `yaml:"jailbreak_mapping_path"`
	UseVLLM              bool    `yaml:"use_vllm,omitempty"`
}

type FeedbackDetectorConfig struct {
	Enabled             bool    `yaml:"enabled"`
	ModelID             string  `yaml:"model_id"`
	Threshold           float32 `yaml:"threshold"`
	UseCPU              bool    `yaml:"use_cpu"`
	UseModernBERT       bool    `yaml:"use_modernbert"`
	UseMmBERT32K        bool    `yaml:"use_mmbert_32k"`
	FeedbackMappingPath string  `yaml:"feedback_mapping_path"`
}

type PreferenceModelConfig struct {
	UseContrastive *bool  `yaml:"use_contrastive,omitempty"`
	EmbeddingModel string `yaml:"embedding_model,omitempty"`
}

func (c PreferenceModelConfig) WithDefaults() PreferenceModelConfig {
	result := c
	if result.UseContrastive == nil {
		defaultEnabled := true
		result.UseContrastive = &defaultEnabled
	}
	return result
}

func (c PreferenceModelConfig) ContrastiveEnabled() bool {
	return *c.WithDefaults().UseContrastive
}

type ExternalModelConfig struct {
	Provider       string                 `yaml:"llm_provider"`
	ModelRole      string                 `yaml:"model_role"`
	ModelEndpoint  ClassifierVLLMEndpoint `yaml:"llm_endpoint,omitempty"`
	ModelName      string                 `yaml:"llm_model_name,omitempty"`
	TimeoutSeconds int                    `yaml:"llm_timeout_seconds,omitempty"`
	ParserType     string                 `yaml:"parser_type,omitempty"`
	Threshold      float32                `yaml:"threshold,omitempty"`
	AccessKey      string                 `yaml:"access_key,omitempty"`
	MaxTokens      int                    `yaml:"max_tokens,omitempty"`
	Temperature    float64                `yaml:"temperature,omitempty"`
}

type ToolFilteringWeights struct {
	Embed    *float32 `yaml:"embed,omitempty"`
	Lexical  *float32 `yaml:"lexical,omitempty"`
	Tag      *float32 `yaml:"tag,omitempty"`
	Name     *float32 `yaml:"name,omitempty"`
	Category *float32 `yaml:"category,omitempty"`
}

type AdvancedToolFilteringConfig struct {
	Enabled                     bool                 `yaml:"enabled"`
	CandidatePoolSize           *int                 `yaml:"candidate_pool_size,omitempty"`
	MinLexicalOverlap           *int                 `yaml:"min_lexical_overlap,omitempty"`
	MinCombinedScore            *float32             `yaml:"min_combined_score,omitempty"`
	Weights                     ToolFilteringWeights `yaml:"weights,omitempty"`
	UseCategoryFilter           *bool                `yaml:"use_category_filter,omitempty"`
	CategoryConfidenceThreshold *float32             `yaml:"category_confidence_threshold,omitempty"`
	AllowTools                  []string             `yaml:"allow_tools,omitempty"`
	BlockTools                  []string             `yaml:"block_tools,omitempty"`
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
	Enabled                 bool                     `yaml:"enabled"`
	FactCheckModel          FactCheckModelConfig     `yaml:"fact_check_model"`
	HallucinationModel      HallucinationModelConfig `yaml:"hallucination_model"`
	NLIModel                NLIModelConfig           `yaml:"nli_model"`
	OnHallucinationDetected string                   `yaml:"on_hallucination_detected,omitempty"`
}

type FactCheckModelConfig struct {
	ModelID      string  `yaml:"model_id"`
	Threshold    float32 `yaml:"threshold"`
	UseCPU       bool    `yaml:"use_cpu"`
	UseMmBERT32K bool    `yaml:"use_mmbert_32k"`
}

type HallucinationModelConfig struct {
	ModelID                string  `yaml:"model_id"`
	Threshold              float32 `yaml:"threshold"`
	UseCPU                 bool    `yaml:"use_cpu"`
	MinSpanLength          int     `yaml:"min_span_length,omitempty"`
	MinSpanConfidence      float32 `yaml:"min_span_confidence,omitempty"`
	ContextWindowSize      int     `yaml:"context_window_size,omitempty"`
	EnableNLIFiltering     bool    `yaml:"enable_nli_filtering,omitempty"`
	NLIEntailmentThreshold float32 `yaml:"nli_entailment_threshold,omitempty"`
}

type NLIModelConfig struct {
	ModelID   string  `yaml:"model_id"`
	Threshold float32 `yaml:"threshold"`
	UseCPU    bool    `yaml:"use_cpu"`
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
	APIKey              string `yaml:"api_key,omitempty"`
	ProviderProfileName string `yaml:"provider_profile,omitempty"`
	Model               string `yaml:"model,omitempty"`
	Protocol            string `yaml:"protocol,omitempty"`
}

type ProviderProfile struct {
	Type         string            `yaml:"type"`
	BaseURL      string            `yaml:"base_url,omitempty"`
	AuthHeader   string            `yaml:"auth_header,omitempty"`
	AuthPrefix   string            `yaml:"auth_prefix,omitempty"`
	ExtraHeaders map[string]string `yaml:"extra_headers,omitempty"`
	APIVersion   string            `yaml:"api_version,omitempty"`
	ChatPath     string            `yaml:"chat_path,omitempty"`
}

type ModelPricing struct {
	Currency        string  `yaml:"currency,omitempty"`
	PromptPer1M     float64 `yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M float64 `yaml:"completion_per_1m,omitempty"`
}

type ModelParams struct {
	PreferredEndpoints []string          `yaml:"preferred_endpoints,omitempty"`
	Pricing            ModelPricing      `yaml:"pricing,omitempty"`
	ReasoningFamily    string            `yaml:"reasoning_family,omitempty"`
	LoRAs              []LoRAAdapter     `yaml:"loras,omitempty"`
	AccessKey          string            `yaml:"access_key,omitempty"`
	ParamSize          string            `yaml:"param_size,omitempty"`
	ContextWindowSize  int               `yaml:"context_window_size,omitempty"`
	APIFormat          string            `yaml:"api_format,omitempty"`
	Description        string            `yaml:"description,omitempty"`
	Capabilities       []string          `yaml:"capabilities,omitempty"`
	Tags               []string          `yaml:"tags,omitempty"`
	QualityScore       float64           `yaml:"quality_score,omitempty"`
	ExternalModelIDs   map[string]string `yaml:"external_model_ids,omitempty"`
	Modality           string            `yaml:"modality,omitempty"`
	ImageGenBackend    string            `yaml:"image_gen_backend,omitempty"`
}

type LoRAAdapter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type ReasoningFamilyConfig struct {
	Type      string `yaml:"type"`
	Parameter string `yaml:"parameter"`
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
