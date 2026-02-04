package config

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ConfigSource defines where to load dynamic configuration from
type ConfigSource string

const (
	// ConfigSourceFile loads configuration from file (default)
	ConfigSourceFile ConfigSource = "file"
	// ConfigSourceKubernetes loads configuration from Kubernetes CRDs
	ConfigSourceKubernetes ConfigSource = "kubernetes"
)

// Model role constants for external models
const (
	ModelRoleGuardrail      = "guardrail"
	ModelRoleClassification = "classification"
	ModelRoleScoring        = "scoring"
	ModelRolePreference     = "preference" // For route preference matching via external LLM
)

// Signal type constants for rule conditions
const (
	SignalTypeKeyword      = "keyword"
	SignalTypeEmbedding    = "embedding"
	SignalTypeDomain       = "domain"
	SignalTypeFactCheck    = "fact_check"
	SignalTypeUserFeedback = "user_feedback"
	SignalTypePreference   = "preference"
	SignalTypeLanguage     = "language"
	SignalTypeLatency      = "latency"
	SignalTypeContext      = "context"
	SignalTypeComplexity   = "complexity"
)

// API format constants for model backends
const (
	// APIFormatOpenAI is the default OpenAI-compatible API format (used by vLLM, etc.)
	APIFormatOpenAI = "openai"
	// APIFormatAnthropic is the Anthropic Messages API format (used by Claude models)
	APIFormatAnthropic = "anthropic"
)

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// ConfigSource specifies where to load dynamic configuration from (file or kubernetes)
	// +optional
	// +kubebuilder:default=file
	ConfigSource ConfigSource `yaml:"config_source,omitempty"`

	// MoMRegistry maps local model paths to HuggingFace repository IDs
	// Example: "models/mom-embedding-light": "sentence-transformers/all-MiniLM-L12-v2"
	MoMRegistry map[string]string `yaml:"mom_registry,omitempty"`

	/*
		Static: Global Configuration
		Timing: Should be handled when starting the router.
	*/
	// Inline models configuration
	InlineModels `yaml:",inline"`
	/*
		Static: Global Configuration
		Timing: Should be handled when starting the router.
	*/
	// External models configuration
	ExternalModels []ExternalModelConfig `yaml:"external_models,omitempty"`

	// Semantic cache configuration
	SemanticCache `yaml:"semantic_cache"`
	// Response API configuration for stateful conversations
	ResponseAPI ResponseAPIConfig `yaml:"response_api"`
	// Router Replay configuration for recording routing decisions
	RouterReplay RouterReplayConfig `yaml:"router_replay"`
	// Looper configuration for multi-model execution strategies
	Looper LooperConfig `yaml:"looper,omitempty"`
	// LLMObservability for LLM tracing, metrics, and logging
	LLMObservability `yaml:",inline"`
	// API server configuration
	APIServer `yaml:",inline"`
	// Router-specific options
	RouterOptions `yaml:",inline"`
	/*
		Dynamic: User Facing Configurations
		Timing: Should be dynamically handled when running router.
	*/
	// Intelligent routing configuration
	IntelligentRouting `yaml:",inline"`
	// Backend models configuration
	BackendModels `yaml:",inline"`
	// ToolSelection for automatic tool selection
	ToolSelection `yaml:",inline"`
}

// ToolSelection represents the configuration for automatic tool selection
type ToolSelection struct {
	// Tools configuration for automatic tool selection
	Tools ToolsConfig `yaml:"tools"`
}

// API server configuration
type APIServer struct {
	// API configuration for classification endpoints
	API APIConfig `yaml:"api"`
}

// LLMObservability represents the configuration for LLM observability
type LLMObservability struct {
	// Observability configuration for tracing, metrics, and logging
	Observability ObservabilityConfig `yaml:"observability"`
}

type RouterOptions struct {
	// Auto model name for automatic model selection (default: "MoM")
	// This is the model name that clients should use to trigger automatic model selection
	// For backward compatibility, "auto" is also accepted and treated as an alias
	AutoModelName string `yaml:"auto_model_name,omitempty"`

	// Include configured models in /v1/models list endpoint (default: false)
	// When false, only the auto model name is returned
	// When true, all models configured in model_config are also included
	IncludeConfigModelsInList bool `yaml:"include_config_models_in_list,omitempty"`

	// Gateway route cache clearing
	ClearRouteCache bool `yaml:"clear_route_cache"`
}

// InlineModels represents the configuration for models that are built into the binary
type InlineModels struct {
	// Embedding models configuration (Phase 4: Long-context embedding support)
	EmbeddingModels `yaml:"embedding_models"`

	// BERT model configuration for Candle BERT similarity comparison
	BertModel `yaml:"bert_model"`

	// Classifier configuration for text classification
	Classifier `yaml:"classifier"`

	// Prompt guard configuration
	PromptGuard PromptGuardConfig `yaml:"prompt_guard"`

	// Hallucination mitigation configuration
	HallucinationMitigation HallucinationMitigationConfig `yaml:"hallucination_mitigation"`

	// Feedback detector configuration for user satisfaction detection
	FeedbackDetector FeedbackDetectorConfig `yaml:"feedback_detector"`
}

// IntelligentRouting represents the configuration for intelligent routing
type IntelligentRouting struct {
	// Signals extraction rules from user queries
	Signals `yaml:",inline"`

	// Decisions for routing logic (combines rules with AND/OR operators)
	Decisions []Decision `yaml:"decisions,omitempty"`

	// Strategy for selecting decision when multiple decisions match
	// "priority" - select decision with highest priority
	// "confidence" - select decision with highest confidence score
	Strategy string `yaml:"strategy,omitempty"`

	// ModelSelection configures the algorithm used for model selection
	// Supported methods: "static", "elo", "router_dc", "automix", "hybrid"
	ModelSelection ModelSelectionConfig `yaml:"model_selection,omitempty"`

	// Reasoning mode configuration
	ReasoningConfig `yaml:",inline"`
}

// ModelSelectionConfig represents configuration for advanced model selection algorithms
// Reference papers:
//   - Elo: RouteLLM (arXiv:2406.18665) - Weighted Elo using Bradley-Terry model
//   - RouterDC: Query-Based Router by Dual Contrastive Learning (arXiv:2409.19886)
//   - AutoMix: Automatically Mixing Language Models (arXiv:2310.12963)
//   - Hybrid: Cost-Efficient Quality-Aware Query Routing (arXiv:2404.14618)
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm to use
	// Options: "static", "elo", "router_dc", "automix", "hybrid"
	// Default: "static" (uses static scores from configuration)
	Method string `yaml:"method,omitempty"`

	// Enabled indicates if model selection is enabled
	Enabled bool `yaml:"enabled,omitempty"`

	// Elo configuration for Elo rating-based selection
	Elo EloSelectionConfig `yaml:"elo,omitempty"`

	// RouterDC configuration for dual-contrastive learning selection
	RouterDC RouterDCSelectionConfig `yaml:"router_dc,omitempty"`

	// AutoMix configuration for POMDP-based cascaded routing
	AutoMix AutoMixSelectionConfig `yaml:"automix,omitempty"`

	// Hybrid configuration for combined selection methods
	Hybrid HybridSelectionConfig `yaml:"hybrid,omitempty"`

	// ML configuration for ML-based selection (KNN, KMeans, SVM)
	ML MLSelectionConfig `yaml:"ml,omitempty"`
}

// MLSelectionConfig holds configuration for all ML-based selectors
type MLSelectionConfig struct {
	// ModelsPath is the base path for pretrained model files
	ModelsPath string `yaml:"models_path,omitempty"`

	// EmbeddingDim is the embedding dimension (default: 1024 for Qwen3)
	EmbeddingDim int `yaml:"embedding_dim,omitempty"`

	// KNN configuration
	KNN MLKNNConfig `yaml:"knn,omitempty"`

	// KMeans configuration
	KMeans MLKMeansConfig `yaml:"kmeans,omitempty"`

	// SVM configuration
	SVM MLSVMConfig `yaml:"svm,omitempty"`
}

// MLKNNConfig holds KNN-specific configuration
type MLKNNConfig struct {
	K              int    `yaml:"k,omitempty"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

// MLKMeansConfig holds KMeans-specific configuration
type MLKMeansConfig struct {
	NumClusters      int     `yaml:"num_clusters,omitempty"`
	EfficiencyWeight float64 `yaml:"efficiency_weight,omitempty"`
	PretrainedPath   string  `yaml:"pretrained_path,omitempty"`
}

// MLSVMConfig holds SVM-specific configuration
type MLSVMConfig struct {
	Kernel         string  `yaml:"kernel,omitempty"`
	Gamma          float64 `yaml:"gamma,omitempty"`
	PretrainedPath string  `yaml:"pretrained_path,omitempty"`
}

// EloSelectionConfig configures Elo rating-based model selection
type EloSelectionConfig struct {
	// InitialRating is the starting Elo rating for new models (default: 1500)
	InitialRating float64 `yaml:"initial_rating,omitempty"`

	// KFactor controls rating volatility (default: 32)
	KFactor float64 `yaml:"k_factor,omitempty"`

	// CategoryWeighted enables per-category Elo ratings (default: true)
	CategoryWeighted bool `yaml:"category_weighted,omitempty"`

	// DecayFactor applies time decay to old comparisons (0-1, default: 0)
	DecayFactor float64 `yaml:"decay_factor,omitempty"`

	// MinComparisons before rating is considered stable (default: 5)
	MinComparisons int `yaml:"min_comparisons,omitempty"`

	// CostScalingFactor scales cost consideration (0 = ignore cost)
	CostScalingFactor float64 `yaml:"cost_scaling_factor,omitempty"`

	// StoragePath is the file path for persisting Elo ratings (optional)
	// If set, ratings are loaded on startup and saved after each feedback update
	StoragePath string `yaml:"storage_path,omitempty"`

	// AutoSaveInterval is how often to auto-save ratings (e.g., "5m", "30s")
	// Only used when StoragePath is set. Default: "1m"
	AutoSaveInterval string `yaml:"auto_save_interval,omitempty"`
}

// RouterDCSelectionConfig configures dual-contrastive learning selection
type RouterDCSelectionConfig struct {
	// Temperature for softmax scaling (default: 0.07)
	Temperature float64 `yaml:"temperature,omitempty"`

	// DimensionSize for embeddings (default: 768)
	DimensionSize int `yaml:"dimension_size,omitempty"`

	// MinSimilarity threshold for valid matches (default: 0.3)
	MinSimilarity float64 `yaml:"min_similarity,omitempty"`

	// UseQueryContrastive enables query-side contrastive learning
	UseQueryContrastive bool `yaml:"use_query_contrastive,omitempty"`

	// UseModelContrastive enables model-side contrastive learning
	UseModelContrastive bool `yaml:"use_model_contrastive,omitempty"`

	// RequireDescriptions enforces that all models have descriptions
	// When true, validation will fail if any model lacks a description
	RequireDescriptions bool `yaml:"require_descriptions,omitempty"`

	// UseCapabilities enables using structured capability tags for matching
	// When true, capabilities are included in the embedding text
	UseCapabilities bool `yaml:"use_capabilities,omitempty"`
}

// AutoMixSelectionConfig configures POMDP-based cascaded routing
type AutoMixSelectionConfig struct {
	// VerificationThreshold for self-verification (default: 0.7)
	VerificationThreshold float64 `yaml:"verification_threshold,omitempty"`

	// MaxEscalations limits escalation count (default: 2)
	MaxEscalations int `yaml:"max_escalations,omitempty"`

	// CostAwareRouting enables cost-quality tradeoff (default: true)
	CostAwareRouting bool `yaml:"cost_aware_routing,omitempty"`

	// CostQualityTradeoff balance (0 = quality, 1 = cost, default: 0.3)
	CostQualityTradeoff float64 `yaml:"cost_quality_tradeoff,omitempty"`

	// DiscountFactor for POMDP value iteration (default: 0.95)
	DiscountFactor float64 `yaml:"discount_factor,omitempty"`

	// UseLogprobVerification uses logprobs for confidence (default: true)
	UseLogprobVerification bool `yaml:"use_logprob_verification,omitempty"`
}

// HybridSelectionConfig configures combined selection methods
type HybridSelectionConfig struct {
	// EloWeight for Elo rating contribution (0-1, default: 0.3)
	EloWeight float64 `yaml:"elo_weight,omitempty"`

	// RouterDCWeight for embedding similarity (0-1, default: 0.3)
	RouterDCWeight float64 `yaml:"router_dc_weight,omitempty"`

	// AutoMixWeight for POMDP value (0-1, default: 0.2)
	AutoMixWeight float64 `yaml:"automix_weight,omitempty"`

	// CostWeight for cost consideration (0-1, default: 0.2)
	CostWeight float64 `yaml:"cost_weight,omitempty"`

	// QualityGapThreshold triggers escalation (default: 0.1)
	QualityGapThreshold float64 `yaml:"quality_gap_threshold,omitempty"`

	// NormalizeScores before combination (default: true)
	NormalizeScores bool `yaml:"normalize_scores,omitempty"`
}

type Signals struct {
	// Keyword-based classification rules
	KeywordRules []KeywordRule `yaml:"keyword_rules,omitempty"`

	// Embedding-based classification rules
	EmbeddingRules []EmbeddingRule `yaml:"embedding_rules,omitempty"`

	// Categories for domain classification (only metadata, used by domain rules)
	Categories []Category `yaml:"categories"`

	// FactCheck rules for fact-check signal classification
	// When matched, outputs "needs_fact_check" or "no_fact_check_needed" signal
	FactCheckRules []FactCheckRule `yaml:"fact_check_rules,omitempty"`

	// UserFeedback rules for user feedback signal classification
	// When matched, outputs one of: "need_clarification", "satisfied", "want_different", "wrong_answer"
	UserFeedbackRules []UserFeedbackRule `yaml:"user_feedback_rules,omitempty"`

	// Preference rules for route preference matching via external LLM
	// When matched, outputs the preference name (route name) that best matches the conversation
	PreferenceRules []PreferenceRule `yaml:"preference_rules,omitempty"`

	// Language rules for multi-language detection signal classification
	// When matched, outputs the detected language code (e.g., "en", "es", "zh", "fr")
	LanguageRules []LanguageRule `yaml:"language_rules,omitempty"`

	// Latency rules for latency-based signal classification
	// When matched, outputs the latency rule name if available models meet TPOT requirements
	LatencyRules []LatencyRule `yaml:"latency_rules,omitempty"`

	// Context rules for token count-based classification
	// When matched, outputs the rule name (e.g., "low_token_count", "high_token_count")
	ContextRules []ContextRule `yaml:"context_rules,omitempty"`

	// Complexity rules for complexity-based classification using embedding similarity
	// When matched, outputs the rule name with difficulty level (e.g., "code_complexity:hard", "math_complexity:easy")
	ComplexityRules []ComplexityRule `yaml:"complexity_rules,omitempty"`
}

// BackendModels represents the configuration for backend models
type BackendModels struct {
	// Model parameters configuration
	ModelConfig map[string]ModelParams `yaml:"model_config"`

	// Default LLM model to use if no match is found
	DefaultModel string `yaml:"default_model"`

	// vLLM endpoints configuration for multiple backend support
	VLLMEndpoints []VLLMEndpoint `yaml:"vllm_endpoints"`
}

type ReasoningConfig struct {
	// Default reasoning effort level (low, medium, high) when not specified per category
	DefaultReasoningEffort string `yaml:"default_reasoning_effort,omitempty"`

	// Reasoning family configurations to define how different model families handle reasoning syntax
	ReasoningFamilies map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
}

// Classifier represents the configuration for text classification
type Classifier struct {
	// In-tree category classifier
	CategoryModel `yaml:"category_model"`
	// Out-of-tree category classifier using MCP
	MCPCategoryModel `yaml:"mcp_category_model,omitempty"`
	// PII detection model
	PIIModel `yaml:"pii_model"`
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
	UseMmBERT32K        bool    `yaml:"use_mmbert_32k"` // Use mmBERT-32K (YaRN, 32K context) instead of ModernBERT
	CategoryMappingPath string  `yaml:"category_mapping_path"`
	// FallbackCategory is returned when classification confidence is below threshold.
	// Default is "other" if not specified.
	FallbackCategory string `yaml:"fallback_category,omitempty"`
}

type PIIModel struct {
	ModelID        string  `yaml:"model_id"`
	Threshold      float32 `yaml:"threshold"`
	UseCPU         bool    `yaml:"use_cpu"`
	UseMmBERT32K   bool    `yaml:"use_mmbert_32k"` // Use mmBERT-32K (YaRN, 32K context) for PII detection
	PIIMappingPath string  `yaml:"pii_mapping_path"`
}

type EmbeddingModels struct {
	// Path to Qwen3-Embedding-0.6B model directory
	Qwen3ModelPath string `yaml:"qwen3_model_path"`
	// Path to EmbeddingGemma-300M model directory
	GemmaModelPath string `yaml:"gemma_model_path"`
	// Path to mmBERT 2D Matryoshka embedding model directory
	// Supports layer early exit (3/6/11/22 layers) and dimension reduction (64-768)
	MmBertModelPath string `yaml:"mmbert_model_path"`
	// Use CPU for inference (default: true, auto-detect GPU if available)
	UseCPU bool `yaml:"use_cpu"`

	// HNSW configuration for embedding-based classification
	// These settings control the preloading and HNSW indexing for embedding-based classification
	HNSWConfig HNSWConfig `yaml:"hnsw_config,omitempty"`
}

// HNSWConfig contains settings for optimizing the embedding classifier
// Note: Despite the name, HNSW indexing is no longer used for embedding classification.
// The classifier always uses brute-force search to ensure complete results for all candidates.
// This struct is kept for backward compatibility and may be renamed in a future version.
type HNSWConfig struct {
	// ModelType specifies which embedding model to use (default: "qwen3")
	// Options: "qwen3" (high quality, 32K context), "gemma" (fast, 8K context), or "mmbert" (multilingual, 2D Matryoshka)
	// This model will be used for both preloading and runtime embedding generation
	ModelType string `yaml:"model_type,omitempty"`

	// PreloadEmbeddings enables precomputing candidate embeddings at startup (default: true)
	// When enabled, candidate embeddings are computed once during initialization
	// rather than on every request, significantly improving runtime performance
	PreloadEmbeddings bool `yaml:"preload_embeddings"`

	// TargetDimension is the embedding dimension to use (default: 768)
	// Supports Matryoshka dimensions: 768, 512, 256, 128, 64
	TargetDimension int `yaml:"target_dimension,omitempty"`

	// TargetLayer is the layer for mmBERT early exit (default: 0 = full model)
	// Only used when ModelType is "mmbert"
	// Options: 3 (fastest, ~7x speedup), 6 (~3.6x), 11 (~2x), 22 (full model)
	TargetLayer int `yaml:"target_layer,omitempty"`

	// EnableSoftMatching enables soft matching mode (default: true)
	// When enabled, if no rule meets its threshold, returns the rule with highest score
	// (as long as it exceeds MinScoreThreshold)
	// Use pointer to distinguish between "not set" (nil) and explicitly disabled (false)
	EnableSoftMatching *bool `yaml:"enable_soft_matching,omitempty"`

	// MinScoreThreshold is the minimum score required for soft matching (default: 0.5)
	// Only used when EnableSoftMatching is true
	// If the highest score is below this threshold, no rule will be matched
	MinScoreThreshold float32 `yaml:"min_score_threshold,omitempty"`
}

// WithDefaults returns a copy of the config with default values applied
func (c HNSWConfig) WithDefaults() HNSWConfig {
	result := c
	// ModelType defaults to "qwen3" for high quality embeddings
	if result.ModelType == "" {
		result.ModelType = "qwen3"
	}
	if result.TargetDimension <= 0 {
		result.TargetDimension = 768
	}
	// EnableSoftMatching: nil means not set, use default true
	// false means explicitly disabled (valid value)
	if result.EnableSoftMatching == nil {
		defaultEnabled := true
		result.EnableSoftMatching = &defaultEnabled
	}
	// MinScoreThreshold defaults to 0.5 for soft matching
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
	ToolName       string            `yaml:"tool_name,omitempty"` // Optional: will auto-discover if not specified
	Threshold      float32           `yaml:"threshold"`
	TimeoutSeconds int               `yaml:"timeout_seconds,omitempty"`
}

// LooperConfig defines the configuration for multi-model execution looper
type LooperConfig struct {
	// Endpoint is the OpenAI-compatible API endpoint to call for model execution
	// Example: "http://localhost:8080/v1/chat/completions"
	Endpoint string `yaml:"endpoint"`

	// Timeout is the maximum duration for each model call (default: 30s)
	TimeoutSeconds int `yaml:"timeout_seconds,omitempty"`

	// RetryCount is the number of retries for failed model calls (default: 0)
	RetryCount int `yaml:"retry_count,omitempty"`

	// Headers are additional headers to include in requests to the endpoint
	Headers map[string]string `yaml:"headers,omitempty"`
}

// IsEnabled returns true if the looper endpoint is configured
func (l *LooperConfig) IsEnabled() bool {
	return l.Endpoint != ""
}

// GetTimeout returns the configured timeout or default (30 seconds)
func (l *LooperConfig) GetTimeout() int {
	if l.TimeoutSeconds <= 0 {
		return 30
	}
	return l.TimeoutSeconds
}

// RedisConfig defines the complete configuration structure for Redis cache backend.
type RedisConfig struct {
	Connection struct {
		Host     string `json:"host" yaml:"host"`
		Port     int    `json:"port" yaml:"port"`
		Database int    `json:"database" yaml:"database"`
		Password string `json:"password" yaml:"password"`
		Timeout  int    `json:"timeout" yaml:"timeout"`
		TLS      struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			CertFile string `json:"cert_file" yaml:"cert_file"`
			KeyFile  string `json:"key_file" yaml:"key_file"`
			CAFile   string `json:"ca_file" yaml:"ca_file"`
		} `json:"tls" yaml:"tls"`
	} `json:"connection" yaml:"connection"`
	Index struct {
		Name        string `json:"name" yaml:"name"`
		Prefix      string `json:"prefix" yaml:"prefix"`
		VectorField struct {
			Name       string `json:"name" yaml:"name"`
			Dimension  int    `json:"dimension" yaml:"dimension"`
			MetricType string `json:"metric_type" yaml:"metric_type"` // L2, IP, COSINE
		} `json:"vector_field" yaml:"vector_field"`
		IndexType string `json:"index_type" yaml:"index_type"` // HNSW or FLAT
		Params    struct {
			M              int `json:"M" yaml:"M"`
			EfConstruction int `json:"efConstruction" yaml:"efConstruction"`
		} `json:"params" yaml:"params"`
	} `json:"index" yaml:"index"`
	Search struct {
		TopK int `json:"topk" yaml:"topk"`
	} `json:"search" yaml:"search"`
	Development struct {
		DropIndexOnStartup bool `json:"drop_index_on_startup" yaml:"drop_index_on_startup"`
		AutoCreateIndex    bool `json:"auto_create_index" yaml:"auto_create_index"`
		VerboseErrors      bool `json:"verbose_errors" yaml:"verbose_errors"`
	} `json:"development" yaml:"development"`
	Logging struct {
		Level          string `json:"level" yaml:"level"`
		EnableQueryLog bool   `json:"enable_query_log" yaml:"enable_query_log"`
		EnableMetrics  bool   `json:"enable_metrics" yaml:"enable_metrics"`
	} `json:"logging" yaml:"logging"`
}

// MilvusConfig defines the complete configuration structure for Milvus cache backend.
// Fields use both json/yaml tags because sigs.k8s.io/yaml converts YAML→JSON before decoding,
// so json tags ensure snake_case keys map correctly without switching parsers.
type MilvusConfig struct {
	Connection struct {
		Host     string `json:"host" yaml:"host"`
		Port     int    `json:"port" yaml:"port"`
		Database string `json:"database" yaml:"database"`
		Timeout  int    `json:"timeout" yaml:"timeout"`
		Auth     struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			Username string `json:"username" yaml:"username"`
			Password string `json:"password" yaml:"password"`
		} `json:"auth" yaml:"auth"`
		TLS struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			CertFile string `json:"cert_file" yaml:"cert_file"`
			KeyFile  string `json:"key_file" yaml:"key_file"`
			CAFile   string `json:"ca_file" yaml:"ca_file"`
		} `json:"tls" yaml:"tls"`
	} `json:"connection" yaml:"connection"`
	Collection struct {
		Name        string `json:"name" yaml:"name"`
		Description string `json:"description" yaml:"description"`
		VectorField struct {
			Name       string `json:"name" yaml:"name"`
			Dimension  int    `json:"dimension" yaml:"dimension"`
			MetricType string `json:"metric_type" yaml:"metric_type"`
		} `json:"vector_field" yaml:"vector_field"`
		Index struct {
			Type   string `json:"type" yaml:"type"`
			Params struct {
				M              int `json:"M" yaml:"M"`
				EfConstruction int `json:"efConstruction" yaml:"efConstruction"`
			} `json:"params" yaml:"params"`
		} `json:"index" yaml:"index"`
	} `json:"collection" yaml:"collection"`
	Search struct {
		Params struct {
			Ef int `json:"ef" yaml:"ef"`
		} `json:"params" yaml:"params"`
		TopK             int    `json:"topk" yaml:"topk"`
		ConsistencyLevel string `json:"consistency_level" yaml:"consistency_level"`
	} `json:"search" yaml:"search"`
	Performance struct {
		ConnectionPool struct {
			MaxConnections     int `json:"max_connections" yaml:"max_connections"`
			MaxIdleConnections int `json:"max_idle_connections" yaml:"max_idle_connections"`
			AcquireTimeout     int `json:"acquire_timeout" yaml:"acquire_timeout"`
		} `json:"connection_pool" yaml:"connection_pool"`
		Batch struct {
			InsertBatchSize int `json:"insert_batch_size" yaml:"insert_batch_size"`
			Timeout         int `json:"timeout" yaml:"timeout"`
		} `json:"batch" yaml:"batch"`
	} `json:"performance" yaml:"performance"`
	DataManagement struct {
		TTL struct {
			Enabled         bool   `json:"enabled" yaml:"enabled"`
			TimestampField  string `json:"timestamp_field" yaml:"timestamp_field"`
			CleanupInterval int    `json:"cleanup_interval" yaml:"cleanup_interval"`
		} `json:"ttl" yaml:"ttl"`
		Compaction struct {
			Enabled  bool `json:"enabled" yaml:"enabled"`
			Interval int  `json:"interval" yaml:"interval"`
		} `json:"compaction" yaml:"compaction"`
	} `json:"data_management" yaml:"data_management"`
	Logging struct {
		Level          string `json:"level" yaml:"level"`
		EnableQueryLog bool   `json:"enable_query_log" yaml:"enable_query_log"`
		EnableMetrics  bool   `json:"enable_metrics" yaml:"enable_metrics"`
	} `json:"logging" yaml:"logging"`
	Development struct {
		DropCollectionOnStartup bool `json:"drop_collection_on_startup" yaml:"drop_collection_on_startup"`
		AutoCreateCollection    bool `json:"auto_create_collection" yaml:"auto_create_collection"`
		VerboseErrors           bool `json:"verbose_errors" yaml:"verbose_errors"`
	} `json:"development" yaml:"development"`
}

type SemanticCache struct {
	// Type of cache backend to use
	BackendType string `yaml:"backend_type,omitempty"`

	// Enable semantic caching
	Enabled bool `yaml:"enabled"`

	// Similarity threshold for cache hits (0.0-1.0)
	// If not specified, will use the BertModel.Threshold
	SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`

	// Maximum number of cache entries to keep (applies to in-memory cache)
	MaxEntries int `yaml:"max_entries,omitempty"`

	// Time-to-live for cache entries in seconds (0 means no expiration)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`

	// Eviction policy for in-memory cache ("fifo", "lru", "lfu")
	EvictionPolicy string `yaml:"eviction_policy,omitempty"`

	// Redis configuration
	Redis *RedisConfig `yaml:"redis,omitempty"`

	// Milvus configuration
	Milvus *MilvusConfig `yaml:"milvus,omitempty"`

	// BackendConfigPath is a path to the backend-specific configuration file (Deprecated)
	BackendConfigPath string `yaml:"backend_config_path,omitempty"`

	// Embedding model to use for semantic similarity ("bert", "qwen3", "gemma")
	// - "bert": Fast, 384-dim, good for short texts (default)
	// - "qwen3": High quality, 1024-dim, supports 32K context
	// - "gemma": Balanced, 768-dim, supports 8K context
	// Default: "bert"
	EmbeddingModel string `yaml:"embedding_model,omitempty"`
}

// ResponseAPIConfig configures the Response API for stateful conversations.
// The Response API provides OpenAI-compatible /v1/responses endpoints
// that support conversation chaining via previous_response_id.
// Requests are translated to Chat Completions format and routed through Envoy.
type ResponseAPIConfig struct {
	// Enable Response API endpoints
	Enabled bool `yaml:"enabled"`

	// Storage backend type: "memory", "milvus", "redis"
	// Default: "memory"
	StoreBackend string `yaml:"store_backend,omitempty"`

	// Time-to-live for stored responses in seconds (0 = 30 days default)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`

	// Maximum number of responses to store (for memory backend)
	MaxResponses int `yaml:"max_responses,omitempty"`

	// Path to backend-specific configuration (for milvus)
	BackendConfigPath string `yaml:"backend_config_path,omitempty"`

	// Milvus configuration (when store_backend is "milvus")
	Milvus ResponseAPIMilvusConfig `yaml:"milvus,omitempty"`

	// Redis configuration (when store_backend is "redis")
	Redis ResponseAPIRedisConfig `yaml:"redis,omitempty"`
}

// ResponseAPIMilvusConfig configures Milvus storage for Response API.
type ResponseAPIMilvusConfig struct {
	// Milvus server address (e.g., "localhost:19530")
	Address string `yaml:"address"`

	// Database name
	Database string `yaml:"database,omitempty"`

	// Collection name for storing responses
	Collection string `yaml:"collection,omitempty"`
}

// ResponseAPIRedisConfig configures Redis storage for Response API.
// Supports both inline configuration and external config file.
type ResponseAPIRedisConfig struct {
	// Basic connection (inline)
	Address  string `yaml:"address,omitempty" json:"address,omitempty"`
	Password string `yaml:"password,omitempty" json:"password,omitempty"`
	DB       int    `yaml:"db" json:"db"`

	// Key management
	// Default: "sr:" (base prefix for keys like sr:response:xxx, sr:conversation:xxx)
	KeyPrefix string `yaml:"key_prefix,omitempty" json:"key_prefix,omitempty"`

	// Cluster support
	ClusterMode      bool     `yaml:"cluster_mode,omitempty" json:"cluster_mode,omitempty"`
	ClusterAddresses []string `yaml:"cluster_addresses,omitempty" json:"cluster_addresses,omitempty"`

	// Connection pooling
	PoolSize     int `yaml:"pool_size,omitempty" json:"pool_size,omitempty"`
	MinIdleConns int `yaml:"min_idle_conns,omitempty" json:"min_idle_conns,omitempty"`
	MaxRetries   int `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`

	// Timeouts (seconds)
	DialTimeout  int `yaml:"dial_timeout,omitempty" json:"dial_timeout,omitempty"`
	ReadTimeout  int `yaml:"read_timeout,omitempty" json:"read_timeout,omitempty"`
	WriteTimeout int `yaml:"write_timeout,omitempty" json:"write_timeout,omitempty"`

	// TLS
	TLSEnabled  bool   `yaml:"tls_enabled,omitempty" json:"tls_enabled,omitempty"`
	TLSCertPath string `yaml:"tls_cert_path,omitempty" json:"tls_cert_path,omitempty"`
	TLSKeyPath  string `yaml:"tls_key_path,omitempty" json:"tls_key_path,omitempty"`
	TLSCAPath   string `yaml:"tls_ca_path,omitempty" json:"tls_ca_path,omitempty"`

	// Optional external config file
	ConfigPath string `yaml:"config_path,omitempty" json:"config_path,omitempty"`
}

// KeywordRule defines a rule for keyword-based classification.
type KeywordRule struct {
	Name          string   `yaml:"name"` // Name is also used as category
	Operator      string   `yaml:"operator"`
	Keywords      []string `yaml:"keywords"`
	CaseSensitive bool     `yaml:"case_sensitive"`
}

// Aggregation method used in keyword embedding rule
type AggregationMethod string

const (
	AggregationMethodMean AggregationMethod = "mean"
	AggregationMethodMax  AggregationMethod = "max"
	AggregationMethodAny  AggregationMethod = "any"
)

// EmbeddingRule defines a rule for keyword embedding based similarity match rule.
type EmbeddingRule struct {
	Name                      string            `yaml:"name"` // Name is also used as category
	SimilarityThreshold       float32           `yaml:"threshold"`
	Candidates                []string          `yaml:"candidates"` // Renamed from Keywords
	AggregationMethodConfiged AggregationMethod `yaml:"aggregation_method"`
}

// APIConfig represents configuration for API endpoints
type APIConfig struct {
	// Batch classification configuration (zero-config auto-discovery)
	BatchClassification struct {
		// Metrics configuration for batch classification monitoring
		Metrics BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
	} `yaml:"batch_classification"`
}

// ObservabilityConfig represents configuration for observability features
type ObservabilityConfig struct {
	// Tracing configuration for distributed tracing
	Tracing TracingConfig `yaml:"tracing"`

	// Metrics configuration for enhanced metrics collection
	Metrics MetricsConfig `yaml:"metrics"`
}

// MetricsConfig represents configuration for metrics collection
type MetricsConfig struct {
	// Enabled controls whether the Prometheus metrics endpoint is served
	// When omitted, defaults to true
	Enabled *bool `yaml:"enabled,omitempty"`

	// Enable windowed metrics collection for load balancing
	WindowedMetrics WindowedMetricsConfig `yaml:"windowed_metrics"`
}

// WindowedMetricsConfig represents configuration for time-windowed metrics
type WindowedMetricsConfig struct {
	// Enable windowed metrics collection
	Enabled bool `yaml:"enabled"`

	// Time windows to track (in duration format, e.g., "1m", "5m", "15m", "1h", "24h")
	// Default: ["1m", "5m", "15m", "1h", "24h"]
	TimeWindows []string `yaml:"time_windows,omitempty"`

	// Update interval for windowed metrics computation (e.g., "10s", "30s")
	// Default: "10s"
	UpdateInterval string `yaml:"update_interval,omitempty"`

	// Enable model-level metrics tracking
	ModelMetrics bool `yaml:"model_metrics"`

	// Enable queue depth estimation
	QueueDepthEstimation bool `yaml:"queue_depth_estimation"`

	// Maximum number of models to track (to prevent cardinality explosion)
	// Default: 100
	MaxModels int `yaml:"max_models,omitempty"`
}

// TracingConfig represents configuration for distributed tracing
type TracingConfig struct {
	// Enable distributed tracing
	Enabled bool `yaml:"enabled"`

	// Provider type (opentelemetry, openinference, openllmetry)
	Provider string `yaml:"provider,omitempty"`

	// Exporter configuration
	Exporter TracingExporterConfig `yaml:"exporter"`

	// Sampling configuration
	Sampling TracingSamplingConfig `yaml:"sampling"`

	// Resource attributes
	Resource TracingResourceConfig `yaml:"resource"`
}

// TracingExporterConfig represents exporter configuration
type TracingExporterConfig struct {
	// Exporter type (otlp, jaeger, zipkin, stdout)
	Type string `yaml:"type"`

	// Endpoint for the exporter (e.g., localhost:4317 for OTLP)
	Endpoint string `yaml:"endpoint,omitempty"`

	// Use insecure connection (no TLS)
	Insecure bool `yaml:"insecure,omitempty"`
}

// TracingSamplingConfig represents sampling configuration
type TracingSamplingConfig struct {
	// Sampling type (always_on, always_off, probabilistic)
	Type string `yaml:"type"`

	// Sampling rate for probabilistic sampling (0.0 to 1.0)
	Rate float64 `yaml:"rate,omitempty"`
}

// TracingResourceConfig represents resource attributes
type TracingResourceConfig struct {
	// Service name
	ServiceName string `yaml:"service_name"`

	// Service version
	ServiceVersion string `yaml:"service_version,omitempty"`

	// Deployment environment
	DeploymentEnvironment string `yaml:"deployment_environment,omitempty"`
}

// BatchClassificationMetricsConfig represents configuration for batch classification metrics
type BatchClassificationMetricsConfig struct {
	// Sample rate for metrics collection (0.0-1.0, 1.0 means collect all metrics)
	SampleRate float64 `yaml:"sample_rate,omitempty"`

	// Batch size range labels for metrics (optional - uses sensible defaults if not specified)
	// Default ranges: "1", "2-5", "6-10", "11-20", "21-50", "50+"
	BatchSizeRanges []BatchSizeRangeConfig `yaml:"batch_size_ranges,omitempty"`

	// Histogram buckets for metrics (directly configured)
	DurationBuckets []float64 `yaml:"duration_buckets,omitempty"`
	SizeBuckets     []float64 `yaml:"size_buckets,omitempty"`

	// Enable detailed metrics collection
	Enabled bool `yaml:"enabled,omitempty"`

	// Enable detailed goroutine tracking (may impact performance)
	DetailedGoroutineTracking bool `yaml:"detailed_goroutine_tracking,omitempty"`

	// Enable high-resolution timing (nanosecond precision)
	HighResolutionTiming bool `yaml:"high_resolution_timing,omitempty"`
}

// BatchSizeRangeConfig defines a batch size range with its boundaries and label
type BatchSizeRangeConfig struct {
	Min   int    `yaml:"min"`
	Max   int    `yaml:"max"` // -1 means no upper limit
	Label string `yaml:"label"`
}

// PromptGuardConfig represents configuration for the prompt guard jailbreak detection
type PromptGuardConfig struct {
	// Enable prompt guard jailbreak detection
	Enabled bool `yaml:"enabled"`

	// Model ID for the jailbreak classification model (Candle model path)
	// Ignored when use_vllm is true
	ModelID string `yaml:"model_id"`

	// Threshold for jailbreak detection (0.0-1.0)
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference (Candle CPU flag)
	// Ignored when use_vllm is true
	UseCPU bool `yaml:"use_cpu"`

	// Use ModernBERT for jailbreak detection (Candle ModernBERT flag)
	// Ignored when use_vllm is true
	UseModernBERT bool `yaml:"use_modernbert"`

	// Use mmBERT-32K for jailbreak detection (32K context, YaRN RoPE, multilingual)
	// Takes precedence over UseModernBERT when both are true
	UseMmBERT32K bool `yaml:"use_mmbert_32k"`

	// Path to the jailbreak type mapping file
	JailbreakMappingPath string `yaml:"jailbreak_mapping_path"`

	// Use vLLM REST API instead of Candle for guardrail/safety checks
	// When true, vLLM configuration must be provided in external_models with model_role="guardrail"
	// When false (default), uses Candle-based classification with ModelID, UseCPU, and UseModernBERT/UseMmBERT32K
	UseVLLM bool `yaml:"use_vllm,omitempty"`
}

// FeedbackDetectorConfig represents configuration for user feedback detection
type FeedbackDetectorConfig struct {
	// Enable user feedback detection
	Enabled bool `yaml:"enabled"`

	// Model ID for the feedback classification model (Candle model path)
	// Default: "models/feedback-detector"
	ModelID string `yaml:"model_id"`

	// Threshold for feedback detection (0.0-1.0)
	// Default: 0.5
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference (Candle CPU flag)
	UseCPU bool `yaml:"use_cpu"`

	// Use ModernBERT for feedback detection (Candle ModernBERT flag)
	UseModernBERT bool `yaml:"use_modernbert"`

	// Use mmBERT-32K for feedback detection (32K context, YaRN RoPE, multilingual)
	// Takes precedence over UseModernBERT when both are true
	UseMmBERT32K bool `yaml:"use_mmbert_32k"`

	// Path to the feedback type mapping file
	FeedbackMappingPath string `yaml:"feedback_mapping_path"`
}

// ExternalModelConfig represents configuration for external LLM-based models
type ExternalModelConfig struct {
	// Provider (e.g., "vllm")
	Provider string `yaml:"llm_provider"`
	// Classifier type (e.g., "guardrail", "classification", "scoring")
	ModelRole string `yaml:"model_role"`
	// Dedicated LLM endpoint configuration for PromptGuard
	// This is separate from vllm_endpoints (which are for backend inference)
	ModelEndpoint ClassifierVLLMEndpoint `yaml:"llm_endpoint,omitempty"`
	// Model name on LLM server (e.g., "Qwen/Qwen3Guard-Gen-0.6B")
	ModelName string `yaml:"llm_model_name,omitempty"`
	// Timeout for LLM API calls in seconds
	// Default: 30 seconds if not specified
	TimeoutSeconds int `yaml:"llm_timeout_seconds,omitempty"`
	// Response parser type (optional, auto-detected from model name if not set)
	// Options: "qwen3guard", "json", "simple", "auto"
	// "auto" tries multiple parsers (OR logic)
	ParserType string `yaml:"parser_type,omitempty"`
	// Threshold for classification (0.0-1.0)
	// Used for guardrail models to determine detection threshold
	Threshold float32 `yaml:"threshold,omitempty"`
	// Optional access key for Authorization header
	// If provided, will be sent as "Authorization: Bearer <access_key>"
	AccessKey string `yaml:"access_key,omitempty"`
}

// ToolFilteringWeights defines per-signal weights for advanced tool filtering.
// All fields are optional and only used when advanced filtering is enabled.
type ToolFilteringWeights struct {
	Embed    *float32 `yaml:"embed,omitempty"`
	Lexical  *float32 `yaml:"lexical,omitempty"`
	Tag      *float32 `yaml:"tag,omitempty"`
	Name     *float32 `yaml:"name,omitempty"`
	Category *float32 `yaml:"category,omitempty"`
}

// AdvancedToolFilteringConfig represents opt-in advanced tool filtering settings.
type AdvancedToolFilteringConfig struct {
	// Enable advanced tool filtering.
	Enabled bool `yaml:"enabled"`

	// Candidate pool size before secondary filtering.
	CandidatePoolSize *int `yaml:"candidate_pool_size,omitempty"`

	// Minimum lexical overlap for keyword filtering.
	MinLexicalOverlap *int `yaml:"min_lexical_overlap,omitempty"`

	// Minimum combined score threshold (0.0-1.0).
	MinCombinedScore *float32 `yaml:"min_combined_score,omitempty"`

	// Weights for combined scoring.
	Weights ToolFilteringWeights `yaml:"weights,omitempty"`

	// Enable category-based filtering.
	UseCategoryFilter *bool `yaml:"use_category_filter,omitempty"`

	// Minimum confidence required for category filtering (0.0-1.0).
	CategoryConfidenceThreshold *float32 `yaml:"category_confidence_threshold,omitempty"`

	// Explicit allow/block lists for tool names.
	AllowTools []string `yaml:"allow_tools,omitempty"`
	BlockTools []string `yaml:"block_tools,omitempty"`
}

// ToolsConfig represents configuration for automatic tool selection
type ToolsConfig struct {
	// Enable automatic tool selection
	Enabled bool `yaml:"enabled"`

	// Number of top tools to select based on similarity (top-k)
	TopK int `yaml:"top_k"`

	// Similarity threshold for tool selection (0.0-1.0)
	// If not specified, will use the BertModel.Threshold
	SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`

	// Path to the tools database file (JSON format)
	ToolsDBPath string `yaml:"tools_db_path"`

	// Fallback behavior: if true, return empty tools on failure; if false, return error
	FallbackToEmpty bool `yaml:"fallback_to_empty"`

	// Advanced tool filtering (opt-in).
	AdvancedFiltering *AdvancedToolFilteringConfig `yaml:"advanced_filtering,omitempty"`
}

// HallucinationMitigationConfig represents configuration for hallucination mitigation
// This feature classifies prompts to determine if they need fact-checking, and when tools
// are used (for RAG), verifies that the LLM response is grounded in the provided context.
type HallucinationMitigationConfig struct {
	// Enable hallucination mitigation
	Enabled bool `yaml:"enabled"`

	// Fact-check classifier configuration
	FactCheckModel FactCheckModelConfig `yaml:"fact_check_model"`

	// Hallucination detection model configuration
	HallucinationModel HallucinationModelConfig `yaml:"hallucination_model"`

	// NLI model configuration for enhanced hallucination detection with explanations
	NLIModel NLIModelConfig `yaml:"nli_model"`

	// Action when hallucination detected: "warn"
	// "warn" - log warning and add response header with hallucination info
	// Default: "warn"
	OnHallucinationDetected string `yaml:"on_hallucination_detected,omitempty"`
}

// FactCheckModelConfig represents configuration for the fact-check classifier
// This classifier determines whether a user prompt requires external fact verification
type FactCheckModelConfig struct {
	// Path to the fact-check classifier model
	ModelID string `yaml:"model_id"`

	// Confidence threshold for classifying as FACT_CHECK_NEEDED (0.0-1.0)
	// Default: 0.7
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference
	UseCPU bool `yaml:"use_cpu"`

	// Use mmBERT-32K for fact-check classification (32K context, YaRN RoPE, multilingual)
	UseMmBERT32K bool `yaml:"use_mmbert_32k"`
}

// HallucinationModelConfig represents configuration for hallucination detection model
// The model uses NLI to detect if LLM responses contain claims not supported by context
type HallucinationModelConfig struct {
	// Path to the hallucination detection model
	ModelID string `yaml:"model_id"`

	// Confidence threshold for hallucination detection (0.0-1.0)
	// Lower values are more sensitive to potential hallucinations
	// Default: 0.5
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference
	UseCPU bool `yaml:"use_cpu"`

	// Minimum span length (in tokens) to consider for hallucination detection
	// Helps reduce false positives from single-token mismatches
	// Default: 1
	MinSpanLength int `yaml:"min_span_length,omitempty"`

	// MinSpanConfidence is the minimum average confidence (0.0–1.0)
	// required for a span to be considered non-hallucinated.
	// Spans with average confidence below this threshold are flagged
	// as potential hallucinations.
	// Default: 0.0 (disable span confidence filtering)
	MinSpanConfidence float32 `yaml:"min_span_confidence,omitempty"`

	// Context window size for span extraction (in tokens)
	// Provides additional context around detected spans for better accuracy
	// Default: 50
	ContextWindowSize int `yaml:"context_window_size,omitempty"`

	// EnableNLIFiltering enables NLI-based false positive filtering.
	// When enabled, an NLI model verifies whether detected hallucination
	// spans are actually unsupported by the surrounding context,
	// reducing false positives.
	// Default: false
	EnableNLIFiltering bool `yaml:"enable_nli_filtering,omitempty"`

	// NLIEntailmentThreshold is the confidence threshold (0.0-1.0)
	// for NLI entailment when filtering hallucination spans.
	// Spans with NLI entailment confidence above this threshold
	// are considered supported by context and not hallucinations.
	// Default: 0.75
	NLIEntailmentThreshold float32 `yaml:"nli_entailment_threshold,omitempty"`
}

// NLIModelConfig represents configuration for the NLI (Natural Language Inference) model
// Used for enhanced hallucination detection with explanations
// Recommended model: tasksource/ModernBERT-base-nli
type NLIModelConfig struct {
	// Path to the NLI model
	ModelID string `yaml:"model_id"`

	// Confidence threshold for NLI classification (0.0-1.0)
	// Default: 0.7
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference
	UseCPU bool `yaml:"use_cpu"`
}

// ClassifierVLLMEndpoint represents a vLLM endpoint configuration for classifiers
// This is separate from VLLMEndpoint (which is for backend inference)
type ClassifierVLLMEndpoint struct {
	// Address of the vLLM endpoint (IP address)
	Address string `yaml:"address"`

	// Port of the vLLM endpoint
	Port int `yaml:"port"`

	// Optional name identifier for the endpoint (for logging and debugging)
	Name string `yaml:"name,omitempty"`

	// Use chat template format for models requiring chat format (e.g., Qwen3Guard)
	UseChatTemplate bool `yaml:"use_chat_template,omitempty"`

	// Custom prompt template (supports %s placeholder for the prompt)
	// If empty, uses default formatting
	PromptTemplate string `yaml:"prompt_template,omitempty"`
}

// VLLMEndpoint represents a vLLM backend endpoint configuration
type VLLMEndpoint struct {
	// Name identifier for the endpoint
	Name string `yaml:"name"`

	// Address of the vLLM endpoint
	Address string `yaml:"address"`

	// Port of the vLLM endpoint
	Port int `yaml:"port"`

	// Load balancing weight for this endpoint
	Weight int `yaml:"weight,omitempty"`

	// Type of endpoint API: "vllm" (default), "openai", "ollama", "huggingface", "openrouter"
	// This determines how requests are formatted and which API format to use
	// +optional
	// +kubebuilder:default=vllm
	Type string `yaml:"type,omitempty"`

	// API key for authenticated endpoints (HuggingFace, OpenRouter, etc.)
	// Can also be set via environment variables: HF_API_KEY, OPENROUTER_API_KEY
	// +optional
	APIKey string `yaml:"api_key,omitempty"`
}

// ModelPricing represents configuration for model-specific parameters
type ModelPricing struct {
	// ISO currency code for the pricing (e.g., "USD"). Defaults to "USD" when omitted.
	Currency string `yaml:"currency,omitempty"`

	// Price per 1M tokens (unit: <currency>/1_000_000 tokens)
	PromptPer1M     float64 `yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M float64 `yaml:"completion_per_1m,omitempty"`
}

type ModelParams struct {
	// Preferred endpoints for this model (optional)
	PreferredEndpoints []string `yaml:"preferred_endpoints,omitempty"`

	// Optional pricing used for cost computation
	Pricing ModelPricing `yaml:"pricing,omitempty"`

	// Reasoning family for this model (e.g., "deepseek", "qwen3", "gpt-oss")
	// If empty, the model doesn't support reasoning mode
	ReasoningFamily string `yaml:"reasoning_family,omitempty"`

	// LoRA adapters available for this model
	// These must be registered with vLLM using --lora-modules flag
	LoRAs []LoRAAdapter `yaml:"loras,omitempty"`

	// Access key for authentication with the model endpoint
	// When set, router will add "Authorization: Bearer {access_key}" header to requests
	AccessKey string `yaml:"access_key,omitempty"`

	// ParamSize represents the model parameter size (e.g., "10b", "5b", "100m")
	// Used by confidence algorithm to determine model order.
	// Larger parameter count typically means more capable but slower/costlier model.
	ParamSize string `yaml:"param_size,omitempty"`

	// APIFormat specifies the API format for this model: "openai" (default) or "anthropic"
	// When set to "anthropic", the router will translate OpenAI-format requests to Anthropic
	// Messages API format and convert responses back to OpenAI format
	APIFormat string `yaml:"api_format,omitempty"`

	// Description provides a natural language description of the model's capabilities
	// Used by RouterDC to compute model embeddings for query-model matching
	// Example: "Fast, efficient model for simple queries and basic code generation"
	Description string `yaml:"description,omitempty"`

	// Capabilities is a list of structured capability tags for the model
	// Used by RouterDC and hybrid selection methods for capability matching
	// Example: ["chat", "code", "reasoning", "math", "creative"]
	Capabilities []string `yaml:"capabilities,omitempty"`

	// QualityScore is the estimated quality/capability score for the model (0.0-1.0)
	// Used by AutoMix and hybrid selection for quality-cost tradeoff calculations
	// Default: 0.8 if not specified
	// Example: 0.95 for a high-quality model, 0.6 for a fast but less capable model
	QualityScore float64 `yaml:"quality_score,omitempty"`

	// ExternalModelIDs maps endpoint types to their model identifiers
	// This allows mapping the internal model name to different external model IDs
	// Example: {"huggingface": "meta-llama/Llama-3.1-8B-Instruct", "ollama": "llama3.1:8b"}
	// +optional
	ExternalModelIDs map[string]string `yaml:"external_model_ids,omitempty"`
}

// LoRAAdapter represents a LoRA adapter configuration for a model
type LoRAAdapter struct {
	// Name of the LoRA adapter (must match the name registered with vLLM)
	Name string `yaml:"name"`
	// Description of what this LoRA adapter is optimized for
	Description string `yaml:"description,omitempty"`
}

// ReasoningFamilyConfig defines how a reasoning family handles reasoning mode
type ReasoningFamilyConfig struct {
	Type      string `yaml:"type"`      // "chat_template_kwargs" or "reasoning_effort"
	Parameter string `yaml:"parameter"` // "thinking", "enable_thinking", "reasoning_effort", etc.
}

// PIIPolicy represents the PII (Personally Identifiable Information) policy for a model
type PIIPolicy struct {
	// Allow all PII by default (true) or deny all by default (false)
	AllowByDefault bool `yaml:"allow_by_default"`

	// List of specific PII types to allow when AllowByDefault is false
	// This field explicitly lists the PII types that are allowed for this model
	PIITypes []string `yaml:"pii_types_allowed,omitempty"`
}

// PIIType constants for common PII types (matching pii_type_mapping.json)
const (
	PIITypeAge             = "AGE"               // Age information
	PIITypeCreditCard      = "CREDIT_CARD"       // Credit Card Number
	PIITypeDateTime        = "DATE_TIME"         // Date/Time information
	PIITypeDomainName      = "DOMAIN_NAME"       // Domain/Website names
	PIITypeEmailAddress    = "EMAIL_ADDRESS"     // Email Address
	PIITypeGPE             = "GPE"               // Geopolitical Entity
	PIITypeIBANCode        = "IBAN_CODE"         // International Bank Account Number
	PIITypeIPAddress       = "IP_ADDRESS"        // IP Address
	PIITypeNoPII           = "NO_PII"            // No PII detected
	PIITypeNRP             = "NRP"               // Nationality/Religious/Political group
	PIITypeOrganization    = "ORGANIZATION"      // Organization names
	PIITypePerson          = "PERSON"            // Person names
	PIITypePhoneNumber     = "PHONE_NUMBER"      // Phone Number
	PIITypeStreetAddress   = "STREET_ADDRESS"    // Physical Address
	PIITypeUSDriverLicense = "US_DRIVER_LICENSE" // US Driver's License Number
	PIITypeUSSSN           = "US_SSN"            // US Social Security Number
	PIITypeZipCode         = "ZIP_CODE"          // ZIP/Postal codes
)

// Category represents a category for routing queries
// Category represents a domain category (only metadata, used by domain rules)
type Category struct {
	// Metadata
	CategoryMetadata `yaml:",inline"`
	// ModelScores for the category
	ModelScores []ModelScore `yaml:"model_scores,omitempty"`
}

// ModelScore represents a model's score for a category
type ModelScore struct {
	Model        string  `yaml:"model"`
	Score        float64 `yaml:"score"`
	UseReasoning *bool   `yaml:"use_reasoning"`
}

// Decision represents a routing decision that combines multiple rules with AND/OR logic
type Decision struct {
	// Name is the unique identifier for this decision
	Name string `yaml:"name"`

	// Description provides information about what this decision handles
	Description string `yaml:"description,omitempty"`

	// Priority is used when strategy is "priority" - higher priority decisions are preferred
	Priority int `yaml:"priority,omitempty"`

	// Rules defines the combination of keyword/embedding/domain rules using AND/OR logic
	Rules RuleCombination `yaml:"rules"`

	// ModelSelectionAlgorithm configures how to select from multiple models in ModelRefs
	// If not specified, defaults to selecting the first model
	ModelSelectionAlgorithm *ModelSelectionConfig `yaml:"modelSelectionAlgorithm,omitempty"`

	// ModelRefs contains model references for this decision
	// When multiple models are specified, ModelSelectionAlgorithm determines which to use
	ModelRefs []ModelRef `yaml:"modelRefs,omitempty"`

	// Algorithm defines the multi-model execution strategy when multiple ModelRefs are configured.
	// When nil or not specified, only the first ModelRef is used.
	Algorithm *AlgorithmConfig `yaml:"algorithm,omitempty"`

	// Plugins contains policy configurations applied after rule matching
	Plugins []DecisionPlugin `yaml:"plugins,omitempty"`
}

// AlgorithmConfig defines how multiple models should be executed and aggregated
type AlgorithmConfig struct {
	// Type specifies the algorithm type:
	// Looper algorithms (multi-model execution):
	// - "confidence": Try smaller models first, escalate to larger models if confidence is low
	// - "ratings": Execute all models concurrently and return multiple choices for comparison
	// - "remom": Multi-round parallel reasoning with intelligent synthesis (Reasoning for Mixture of Models)
	// Selection algorithms (single model selection from candidates):
	// - "static": Use static scores from configuration (default)
	// - "elo": Use Elo rating system with Bradley-Terry model
	// - "router_dc": Use dual-contrastive learning for query-model matching
	// - "automix": Use POMDP-based cost-quality optimization
	// - "hybrid": Combine multiple selection methods with configurable weights
	Type string `yaml:"type"`

	// Looper algorithm configurations (for multi-model execution)
	Confidence *ConfidenceAlgorithmConfig `yaml:"confidence,omitempty"`
	Ratings    *RatingsAlgorithmConfig    `yaml:"ratings,omitempty"`
	ReMoM      *ReMoMAlgorithmConfig      `yaml:"remom,omitempty"`

	// Selection algorithm configurations (for single model selection)
	// These align with the global ModelSelectionConfig but can be overridden per-decision
	Elo      *EloSelectionConfig      `yaml:"elo,omitempty"`
	RouterDC *RouterDCSelectionConfig `yaml:"router_dc,omitempty"`
	AutoMix  *AutoMixSelectionConfig  `yaml:"automix,omitempty"`
	Hybrid   *HybridSelectionConfig   `yaml:"hybrid,omitempty"`

	// OnError defines behavior when algorithm fails: "skip" or "fail"
	// - "skip": Skip and use fallback (default)
	// - "fail": Return error immediately
	OnError string `yaml:"on_error,omitempty"`
}

// ConfidenceAlgorithmConfig configures the confidence algorithm
// This algorithm tries smaller models first and escalates to larger models if confidence is low
type ConfidenceAlgorithmConfig struct {
	// ConfidenceMethod specifies how to evaluate model confidence
	// - "avg_logprob": Use average logprob across all tokens (default)
	// - "margin": Use average margin between top-1 and top-2 logprobs (more accurate)
	// - "hybrid": Use weighted combination of both methods
	// - "self_verify": AutoMix self-verification - model evaluates its own answer (arXiv:2310.12963)
	ConfidenceMethod string `yaml:"confidence_method,omitempty"`

	// Threshold is the confidence threshold for escalation
	// For avg_logprob: logprobs are negative, higher (closer to 0) = more confident
	//   - Default: -1.0 (very permissive)
	//   - Typical range: -2.0 to -0.1
	// For margin: positive values, higher = more confident
	//   - Default: 0.5
	//   - Typical range: 0.1 to 2.0
	// For hybrid: normalized score between 0 and 1
	//   - Default: 0.5
	Threshold float64 `yaml:"threshold,omitempty"`

	// HybridWeights configures weights for hybrid method (only used when confidence_method="hybrid")
	// LogprobWeight + MarginWeight should equal 1.0
	HybridWeights *HybridWeightsConfig `yaml:"hybrid_weights,omitempty"`

	// OnError defines behavior when a model call fails: "skip" or "fail"
	// - "skip": Skip the failed model and try the next one (default)
	// - "fail": Return error immediately
	OnError string `yaml:"on_error,omitempty"`

	// EscalationOrder determines how models are ordered for cascaded execution
	// - "size": Order by param_size (smallest first) - default behavior
	// - "cost": Order by pricing (cheapest first) - AutoMix-style cost optimization
	// - "automix": Use POMDP-optimized ordering based on cost-quality tradeoff
	EscalationOrder string `yaml:"escalation_order,omitempty"`

	// CostQualityTradeoff controls the balance when escalation_order is "automix"
	// 0.0 = pure quality (ignore cost), 1.0 = pure cost (ignore quality)
	// Default: 0.3 (favor quality but consider cost)
	CostQualityTradeoff float64 `yaml:"cost_quality_tradeoff,omitempty"`
}

// HybridWeightsConfig configures weights for hybrid confidence method
type HybridWeightsConfig struct {
	LogprobWeight float64 `yaml:"logprob_weight,omitempty"` // Weight for avg_logprob (default: 0.5)
	MarginWeight  float64 `yaml:"margin_weight,omitempty"`  // Weight for margin (default: 0.5)
}

// RatingsAlgorithmConfig configures the ratings algorithm
// This algorithm executes all models concurrently and returns multiple choices for comparison
type RatingsAlgorithmConfig struct {
	// MaxConcurrent limits the number of concurrent model calls
	// Default: no limit (all models called concurrently)
	MaxConcurrent int `yaml:"max_concurrent,omitempty"`

	// OnError defines behavior when a model call fails: "skip" or "fail"
	// - "skip": Skip the failed model and return remaining results (default)
	// - "fail": Return error if any model fails
	OnError string `yaml:"on_error,omitempty"`
}

// ReMoMAlgorithmConfig configures the ReMoM (Reasoning for Mixture of Models) algorithm
// This algorithm performs multi-round parallel reasoning with intelligent synthesis
// Inspired by PaCoRe (arXiv:2601.05593) but extended to support mixture of models
type ReMoMAlgorithmConfig struct {
	// BreadthSchedule defines the number of parallel calls per round
	// The final round (K=1) is automatically appended
	// Examples:
	//   [4]      -> Low intensity: 2 rounds (4 → 1)
	//   [16]     -> Medium intensity: 2 rounds (16 → 1)
	//   [32, 4]  -> High intensity: 3 rounds (32 → 4 → 1)
	BreadthSchedule []int `yaml:"breadth_schedule"`

	// ModelDistribution specifies how to distribute calls among multiple models
	// - "weighted": Distribute proportionally based on model weights (default)
	// - "equal": Distribute evenly among all models
	// - "first_only": Use only the first model (PaCoRe-compatible mode)
	ModelDistribution string `yaml:"model_distribution,omitempty"`

	// Temperature for generation (default: 1.0)
	Temperature float64 `yaml:"temperature,omitempty"`

	// IncludeReasoning determines whether to include reasoning content in synthesis
	// When true, extracts vLLM reasoning fields and includes them in synthesis prompts
	// Default: false
	IncludeReasoning bool `yaml:"include_reasoning,omitempty"`

	// CompactionStrategy defines how to compact responses between rounds
	// - "full": Use complete responses (default)
	// - "last_n_tokens": Keep only the last N tokens
	CompactionStrategy string `yaml:"compaction_strategy,omitempty"`

	// CompactionTokens specifies how many tokens to keep when using "last_n_tokens" strategy
	// Default: 1000
	CompactionTokens int `yaml:"compaction_tokens,omitempty"`

	// SynthesisTemplate is a custom Go text/template for building synthesis prompts
	// Available variables: .OriginalContent, .ReferenceResponses
	// If empty, uses default template
	SynthesisTemplate string `yaml:"synthesis_template,omitempty"`

	// MaxConcurrent limits the number of concurrent model calls per round
	// Default: no limit (all calls in a round execute concurrently)
	MaxConcurrent int `yaml:"max_concurrent,omitempty"`

	// OnError defines behavior when a model call fails: "skip" or "fail"
	// - "skip": Skip the failed call and continue with remaining responses (default)
	// - "fail": Return error immediately
	OnError string `yaml:"on_error,omitempty"`

	// ShuffleSeed for reproducible shuffling of responses
	// Default: 42
	ShuffleSeed int `yaml:"shuffle_seed,omitempty"`

	// IncludeIntermediateResponses determines whether to include intermediate responses
	// in the response body for visualization in the dashboard
	// Default: true
	IncludeIntermediateResponses bool `yaml:"include_intermediate_responses,omitempty"`

	// MaxResponsesPerRound limits how many responses to save per round
	// Useful to avoid large response bodies
	// Default: no limit (save all responses)
	MaxResponsesPerRound int `yaml:"max_responses_per_round,omitempty"`
}

// MLModelSelectionConfig configures the ML-based model selection algorithm
// Supported types: knn, kmeans, svm
type MLModelSelectionConfig struct {
	// Type specifies the algorithm: "knn", "kmeans", "svm"
	Type string `yaml:"type"`

	// ModelsPath is the path to pre-trained model files (e.g., "trained_models/")
	// If specified, loads pre-trained models instead of creating empty selectors
	// The algorithm will look for {ModelsPath}/{type}_model.json
	ModelsPath string `yaml:"models_path,omitempty"`

	// K is the number of neighbors for KNN algorithm (default: 3)
	K int `yaml:"k,omitempty"`

	// NumClusters is the number of clusters for KMeans algorithm (default: equals number of models)
	NumClusters int `yaml:"num_clusters,omitempty"`

	// Kernel specifies the SVM kernel type: "linear", "rbf", "poly" (default: "rbf")
	Kernel string `yaml:"kernel,omitempty"`

	// Gamma is the RBF kernel parameter for SVM (default: 1.0)
	Gamma float64 `yaml:"gamma,omitempty"`

	// EfficiencyWeight controls the performance-efficiency tradeoff for KMeans (default: 0.3)
	// 0 = pure performance (quality), 1 = pure efficiency (latency)
	// Use pointer to distinguish "not set" (nil, uses default 0.3) from "explicitly 0"
	EfficiencyWeight *float64 `yaml:"efficiency_weight,omitempty"`

	// FeatureWeights allows custom weighting of features for selection
	FeatureWeights map[string]float64 `yaml:"feature_weights,omitempty"`
}

// ModelRef represents a reference to a model (without score field)
type ModelRef struct {
	Model string `yaml:"model"`
	// Optional LoRA adapter name - when specified, this LoRA adapter name will be used
	// as the final model name in requests instead of the base model name.
	LoRAName string `yaml:"lora_name,omitempty"`
	// Reasoning mode control on Model Level
	ModelReasoningControl `yaml:",inline"`
}

// DecisionPlugin represents a plugin configuration for a decision
type DecisionPlugin struct {
	// Type specifies the plugin type. Permitted values: "semantic-cache", "jailbreak", "pii", "system_prompt", "header_mutation", "hallucination", "router_replay".
	Type string `yaml:"type" json:"type"`

	// Configuration is the raw configuration for this plugin
	// The structure depends on the plugin type
	// When loaded from YAML, this will be a map[string]interface{}
	// When loaded from Kubernetes CRD, this will be []byte (from runtime.RawExtension)
	Configuration interface{} `yaml:"configuration,omitempty" json:"configuration,omitempty"`
}

// Plugin configuration structures for unmarshaling

// SemanticCachePluginConfig represents configuration for semantic-cache plugin
type SemanticCachePluginConfig struct {
	Enabled             bool     `json:"enabled" yaml:"enabled"`
	SimilarityThreshold *float32 `json:"similarity_threshold,omitempty" yaml:"similarity_threshold,omitempty"`
	TTLSeconds          *int     `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"` // Per-entry TTL (0 = do not cache, nil = use global default)
}

// JailbreakPluginConfig represents configuration for jailbreak plugin
type JailbreakPluginConfig struct {
	Enabled        bool     `json:"enabled" yaml:"enabled"`
	Threshold      *float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
	IncludeHistory bool     `json:"include_history,omitempty" yaml:"include_history,omitempty"` // Whether to include conversation history in detection (default: false)
}

// PIIPluginConfig represents configuration for pii plugin
type PIIPluginConfig struct {
	Enabled        bool     `json:"enabled" yaml:"enabled"`
	Threshold      *float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
	IncludeHistory bool     `json:"include_history,omitempty" yaml:"include_history,omitempty"` // Whether to include conversation history in detection (default: false)

	// PII Policy configuration
	// When Enabled is true, all PII types are blocked by default unless listed in PIITypesAllowed
	// When Enabled is false, PII detection is skipped entirely
	PIITypesAllowed []string `json:"pii_types_allowed,omitempty" yaml:"pii_types_allowed,omitempty"`
}

// SystemPromptPluginConfig represents configuration for system_prompt plugin
type SystemPromptPluginConfig struct {
	Enabled      *bool  `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	SystemPrompt string `json:"system_prompt,omitempty" yaml:"system_prompt,omitempty"`
	Mode         string `json:"mode,omitempty" yaml:"mode,omitempty"` // "replace" or "insert"
}

// HeaderMutationPluginConfig represents configuration for header_mutation plugin
type HeaderMutationPluginConfig struct {
	Add    []HeaderPair `json:"add,omitempty" yaml:"add,omitempty"`
	Update []HeaderPair `json:"update,omitempty" yaml:"update,omitempty"`
	Delete []string     `json:"delete,omitempty" yaml:"delete,omitempty"`
}

// HeaderPair represents a header name-value pair
type HeaderPair struct {
	Name  string `json:"name" yaml:"name"`
	Value string `json:"value" yaml:"value"`
}

// HallucinationPluginConfig represents configuration for hallucination detection plugin
type HallucinationPluginConfig struct {
	// Enable hallucination detection for this decision
	Enabled bool `json:"enabled" yaml:"enabled"`

	// UseNLI enables NLI (Natural Language Inference) model for detailed explanations
	// When enabled, each hallucinated span will include:
	// - NLI label (ENTAILMENT/NEUTRAL/CONTRADICTION)
	// - Confidence scores
	// - Severity level (0-4)
	// - Human-readable explanation
	UseNLI bool `json:"use_nli,omitempty" yaml:"use_nli,omitempty"`

	// HallucinationAction specifies the action when hallucination is detected
	// "header" - add warning headers to response (default)
	// "body" - prepend warning text to response content
	// "none" - no action, only log and metrics
	HallucinationAction string `json:"hallucination_action,omitempty" yaml:"hallucination_action,omitempty"`

	// UnverifiedFactualAction specifies the action when fact-check is needed but no tool context available
	// "header" - add warning headers to response (default)
	// "body" - prepend warning text to response content
	// "none" - no action, only log and metrics
	UnverifiedFactualAction string `json:"unverified_factual_action,omitempty" yaml:"unverified_factual_action,omitempty"`

	// IncludeHallucinationDetails includes detailed information in body warning
	// Only effective when HallucinationAction is "body"
	// When true, includes confidence score and hallucinated spans in the warning text
	IncludeHallucinationDetails bool `json:"include_hallucination_details,omitempty" yaml:"include_hallucination_details,omitempty"`
}

// RouterReplayPluginConfig represents configuration for router_replay plugin
// This is the per-decision plugin configuration (overrides global router_replay config)
type RouterReplayPluginConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`

	// MaxRecords controls the maximum number of replay records kept in memory.
	// Only applies when StoreBackend is "memory". Defaults to 200.
	MaxRecords int `json:"max_records,omitempty" yaml:"max_records,omitempty"`

	// CaptureRequestBody controls whether the original request body should be stored.
	// Defaults to false to avoid unintentionally persisting sensitive content.
	CaptureRequestBody bool `json:"capture_request_body,omitempty" yaml:"capture_request_body,omitempty"`

	// CaptureResponseBody controls whether the final response body should be stored.
	// Defaults to false. Enable when you want replay logs to include model output.
	CaptureResponseBody bool `json:"capture_response_body,omitempty" yaml:"capture_response_body,omitempty"`

	// MaxBodyBytes caps how many bytes of request/response body are recorded.
	// Defaults to 4096 bytes.
	MaxBodyBytes int `json:"max_body_bytes,omitempty" yaml:"max_body_bytes,omitempty"`
}

// RouterReplayConfig configures the router replay system at the system level.
// This provides storage backend configuration and system-level settings.
// Per-decision settings (max_records, capture settings) are configured via router_replay plugin.
type RouterReplayConfig struct {
	// StoreBackend specifies the storage backend to use.
	// Options: "memory", "redis", "postgres", "milvus". Defaults to "memory".
	StoreBackend string `json:"store_backend,omitempty" yaml:"store_backend,omitempty"`

	// TTLSeconds specifies how long records should be kept (in seconds).
	// Only applies to persistent backends (redis, postgres, milvus).
	// 0 means no expiration. Example: 2592000 for 30 days.
	TTLSeconds int `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`

	// AsyncWrites enables asynchronous writes to the storage backend.
	// Improves performance but may result in data loss if the process crashes.
	AsyncWrites bool `json:"async_writes,omitempty" yaml:"async_writes,omitempty"`

	// Redis configuration (required if StoreBackend is "redis")
	Redis *RouterReplayRedisConfig `json:"redis,omitempty" yaml:"redis,omitempty"`

	// Postgres configuration (required if StoreBackend is "postgres")
	Postgres *RouterReplayPostgresConfig `json:"postgres,omitempty" yaml:"postgres,omitempty"`

	// Milvus configuration (required if StoreBackend is "milvus")
	Milvus *RouterReplayMilvusConfig `json:"milvus,omitempty" yaml:"milvus,omitempty"`
}

// RouterReplayRedisConfig holds Redis-specific configuration for router replay.
type RouterReplayRedisConfig struct {
	Address       string `json:"address" yaml:"address"`
	DB            int    `json:"db,omitempty" yaml:"db,omitempty"`
	Password      string `json:"password,omitempty" yaml:"password,omitempty"`
	UseTLS        bool   `json:"use_tls,omitempty" yaml:"use_tls,omitempty"`
	TLSSkipVerify bool   `json:"tls_skip_verify,omitempty" yaml:"tls_skip_verify,omitempty"`
	MaxRetries    int    `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
	PoolSize      int    `json:"pool_size,omitempty" yaml:"pool_size,omitempty"`
	KeyPrefix     string `json:"key_prefix,omitempty" yaml:"key_prefix,omitempty"`
}

// RouterReplayPostgresConfig holds PostgreSQL-specific configuration for router replay.
type RouterReplayPostgresConfig struct {
	Host            string `json:"host" yaml:"host"`
	Port            int    `json:"port,omitempty" yaml:"port,omitempty"`
	Database        string `json:"database" yaml:"database"`
	User            string `json:"user" yaml:"user"`
	Password        string `json:"password,omitempty" yaml:"password,omitempty"`
	SSLMode         string `json:"ssl_mode,omitempty" yaml:"ssl_mode,omitempty"`
	MaxOpenConns    int    `json:"max_open_conns,omitempty" yaml:"max_open_conns,omitempty"`
	MaxIdleConns    int    `json:"max_idle_conns,omitempty" yaml:"max_idle_conns,omitempty"`
	ConnMaxLifetime int    `json:"conn_max_lifetime,omitempty" yaml:"conn_max_lifetime,omitempty"`
	TableName       string `json:"table_name,omitempty" yaml:"table_name,omitempty"`
}

// RouterReplayMilvusConfig holds Milvus-specific configuration for router replay.
type RouterReplayMilvusConfig struct {
	Address          string `json:"address" yaml:"address"`
	Username         string `json:"username,omitempty" yaml:"username,omitempty"`
	Password         string `json:"password,omitempty" yaml:"password,omitempty"`
	CollectionName   string `json:"collection_name,omitempty" yaml:"collection_name,omitempty"`
	ConsistencyLevel string `json:"consistency_level,omitempty" yaml:"consistency_level,omitempty"`
	ShardNum         int    `json:"shard_num,omitempty" yaml:"shard_num,omitempty"`
}

// Helper methods for Decision to access plugin configurations

// GetPluginConfig returns the configuration for a specific plugin type
// Returns nil if the plugin is not found
func (d *Decision) GetPluginConfig(pluginType string) interface{} {
	for _, plugin := range d.Plugins {
		if plugin.Type == pluginType {
			return plugin.Configuration
		}
	}
	return nil
}

// unmarshalPluginConfig unmarshals plugin configuration to a target struct
// Handles both map[string]interface{} (from YAML) and []byte (from Kubernetes RawExtension)
func unmarshalPluginConfig(config interface{}, target interface{}) error {
	if config == nil {
		return fmt.Errorf("plugin configuration is nil")
	}

	switch v := config.(type) {
	case map[string]interface{}:
		// From YAML file - convert via JSON
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}
		return json.Unmarshal(data, target)
	case map[interface{}]interface{}:
		// From YAML file with interface{} keys - convert to map[string]interface{} first
		converted := convertMapToStringKeys(v)
		data, err := json.Marshal(converted)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}
		return json.Unmarshal(data, target)
	case []byte:
		// From Kubernetes RawExtension - direct unmarshal
		return json.Unmarshal(v, target)
	default:
		return fmt.Errorf("unsupported configuration type: %T", config)
	}
}

// convertMapToStringKeys recursively converts map[interface{}]interface{} to map[string]interface{}
func convertMapToStringKeys(m map[interface{}]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range m {
		// Convert key to string
		key, ok := k.(string)
		if !ok {
			key = fmt.Sprintf("%v", k)
		}

		// Recursively convert nested maps
		switch val := v.(type) {
		case map[interface{}]interface{}:
			result[key] = convertMapToStringKeys(val)
		case []interface{}:
			result[key] = convertSliceValues(val)
		default:
			result[key] = v
		}
	}
	return result
}

// convertSliceValues recursively converts slice elements that are maps
func convertSliceValues(s []interface{}) []interface{} {
	result := make([]interface{}, len(s))
	for i, v := range s {
		switch val := v.(type) {
		case map[interface{}]interface{}:
			result[i] = convertMapToStringKeys(val)
		case []interface{}:
			result[i] = convertSliceValues(val)
		default:
			result[i] = v
		}
	}
	return result
}

// GetSemanticCacheConfig returns the semantic-cache plugin configuration
func (d *Decision) GetSemanticCacheConfig() *SemanticCachePluginConfig {
	config := d.GetPluginConfig("semantic-cache")
	if config == nil {
		return nil
	}

	result := &SemanticCachePluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal semantic-cache config: %v", err)
		return nil
	}
	return result
}

// GetJailbreakConfig returns the jailbreak plugin configuration
func (d *Decision) GetJailbreakConfig() *JailbreakPluginConfig {
	config := d.GetPluginConfig("jailbreak")
	if config == nil {
		return nil
	}

	result := &JailbreakPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal jailbreak config: %v", err)
		return nil
	}
	return result
}

// GetPIIConfig returns the pii plugin configuration
func (d *Decision) GetPIIConfig() *PIIPluginConfig {
	config := d.GetPluginConfig("pii")
	if config == nil {
		return nil
	}

	result := &PIIPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal pii config: %v", err)
		return nil
	}
	return result
}

// GetSystemPromptConfig returns the system_prompt plugin configuration
func (d *Decision) GetSystemPromptConfig() *SystemPromptPluginConfig {
	config := d.GetPluginConfig("system_prompt")
	if config == nil {
		return nil
	}

	result := &SystemPromptPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal system_prompt config: %v", err)
		return nil
	}
	return result
}

// GetHeaderMutationConfig returns the header_mutation plugin configuration
func (d *Decision) GetHeaderMutationConfig() *HeaderMutationPluginConfig {
	config := d.GetPluginConfig("header_mutation")
	if config == nil {
		return nil
	}

	result := &HeaderMutationPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal header_mutation config: %v", err)
		return nil
	}
	return result
}

// GetHallucinationConfig returns the hallucination plugin configuration
func (d *Decision) GetHallucinationConfig() *HallucinationPluginConfig {
	config := d.GetPluginConfig("hallucination")
	if config == nil {
		return nil
	}

	result := &HallucinationPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal hallucination config: %v", err)
		return nil
	}
	return result
}

// GetRouterReplayConfig returns the router_replay plugin configuration
func (d *Decision) GetRouterReplayConfig() *RouterReplayPluginConfig {
	config := d.GetPluginConfig("router_replay")
	if config == nil {
		return nil
	}

	result := &RouterReplayPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal router_replay config: %v", err)
		return nil
	}
	return result
}

// RuleCombination defines how to combine multiple rule conditions with AND/OR operators
type RuleCombination struct {
	// Operator specifies how to combine conditions: "AND" or "OR"
	Operator string `yaml:"operator"`

	// Conditions is the list of rule references to evaluate
	Conditions []RuleCondition `yaml:"conditions"`
}

// RuleCondition references a specific rule by type and name
type RuleCondition struct {
	// Type specifies the rule type: "keyword", "embedding", "domain", or "fact_check"
	Type string `yaml:"type"`

	// Name is the name of the rule to reference
	// For fact_check type, use "needs_fact_check" to match queries that need fact verification
	Name string `yaml:"name"`
}

// FactCheckRule defines a rule for fact-check signal classification
// Similar to KeywordRule and EmbeddingRule, but based on ML model classification
// The classifier determines if a query needs fact verification and outputs
// one of the predefined signals: "needs_fact_check" or "no_fact_check_needed"
// Threshold is read from hallucination_mitigation.fact_check_model.threshold
type FactCheckRule struct {
	// Name is the signal name that can be referenced in decision rules
	// e.g., "needs_fact_check" or "no_fact_check_needed"
	Name string `yaml:"name"`

	// Description provides human-readable explanation of when this signal is triggered
	Description string `yaml:"description,omitempty"`
}

// UserFeedbackRule defines a rule for user feedback signal classification
// Similar to FactCheckRule, but based on user satisfaction detection
// The classifier determines user feedback type from follow-up messages and outputs
// one of the predefined signals: "need_clarification", "satisfied", "want_different", "wrong_answer"
// Threshold is read from feedback_detector.threshold
type UserFeedbackRule struct {
	// Name is the signal name that can be referenced in decision rules
	// e.g., "need_clarification", "satisfied", "want_different", "wrong_answer"
	Name string `yaml:"name"`

	// Description provides human-readable explanation of when this signal is triggered
	Description string `yaml:"description,omitempty"`
}

// PreferenceRule defines a rule for route preference matching via external LLM
// The external LLM analyzes the conversation and route descriptions to determine
// the best matching route preference using prompt engineering
// Configuration is read from external_models with model_role="preference"
type PreferenceRule struct {
	// Name is the preference name (route name) that can be referenced in decision rules
	// e.g., "code_generation", "bug_fixing", "other"
	Name string `yaml:"name"`

	// Description provides human-readable explanation of what this route handles
	// This description is sent to the external LLM for route matching
	Description string `yaml:"description,omitempty"`
}

// LanguageRule defines a rule for multi-language detection signal classification
// The language classifier detects the query language and outputs language codes
// e.g., "en" (English), "es" (Spanish), "zh" (Chinese), "fr" (French)
type LanguageRule struct {
	// Name is the language code that can be referenced in decision rules
	// e.g., "en", "es", "zh", "fr", "de", "ja"
	Name string `yaml:"name"`

	// Description provides human-readable explanation of the language
	Description string `yaml:"description,omitempty"`
}

// LatencyRule defines a rule for latency-based signal classification
// The latency classifier evaluates if available models meet TPOT (Time Per Output Token) requirements
type LatencyRule struct {
	// Name is the latency rule name that can be referenced in decision rules
	Name string `yaml:"name"`

	// TPOTPercentile is the percentile bucket to use for TPOT (Time Per Output Token) evaluation (1-100)
	// Models with current TPOT <= the calculated TPOT percentile threshold will match this rule
	// Example: 10 means 10th percentile (top 10% fastest TPOT)
	// Example: 50 means median (top 50% fastest TPOT)
	// Works with any number of observations (1+), adapts to model's actual performance
	TPOTPercentile int `yaml:"tpot_percentile,omitempty"`

	// TTFTPercentile is the percentile bucket to use for TTFT (Time To First Token) evaluation (1-100)
	// Models with current TTFT <= the calculated TTFT percentile threshold will match this rule
	// Example: 10 means 10th percentile (top 10% fastest TTFT)
	// Example: 50 means median (top 50% fastest TTFT)
	// Works with any number of observations (1+), adapts to model's actual performance
	// At least one of TPOTPercentile or TTFTPercentile must be set
	TTFTPercentile int `yaml:"ttft_percentile,omitempty"`

	// Description provides human-readable explanation of the latency requirement
	Description string `yaml:"description,omitempty"`
}

// TokenCount represents a token count value with optional K/M suffixes
type TokenCount string

// Value parses the token count string into an integer
func (t TokenCount) Value() (int, error) {
	s := string(t)
	if s == "" {
		return 0, nil
	}
	s = strings.ToUpper(strings.TrimSpace(s))

	multiplier := 1.0
	if strings.HasSuffix(s, "K") {
		multiplier = 1000.0
		s = strings.TrimSuffix(s, "K")
	} else if strings.HasSuffix(s, "M") {
		multiplier = 1000000.0
		s = strings.TrimSuffix(s, "M")
	}

	val, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid token count format: %s", t)
	}

	return int(val * multiplier), nil
}

// ContextRule defines a rule for context-based (token count) classification
type ContextRule struct {
	Name        string     `yaml:"name"`
	MinTokens   TokenCount `yaml:"min_tokens"`
	MaxTokens   TokenCount `yaml:"max_tokens"`
	Description string     `yaml:"description,omitempty"`
}

// ComplexityCandidates defines hard and easy candidates for complexity classification
type ComplexityCandidates struct {
	Candidates []string `yaml:"candidates"`
}

// ComplexityRule defines a rule for complexity-based classification using embedding similarity
// The classifier computes max similarity to hard and easy candidates, then:
// - If (max_hard_sim - max_easy_sim) > threshold: outputs "rulename:hard"
// - If (max_hard_sim - max_easy_sim) < -threshold: outputs "rulename:easy"
// - Otherwise: outputs "rulename:medium"
//
// The Composer field allows filtering based on other signals (e.g., only apply code_complexity when domain is "computer_science")
// This is evaluated after all signals are computed in parallel, enabling signal dependencies.
type ComplexityRule struct {
	Name        string               `yaml:"name"`
	Threshold   float32              `yaml:"threshold"`
	Hard        ComplexityCandidates `yaml:"hard"`
	Easy        ComplexityCandidates `yaml:"easy"`
	Description string               `yaml:"description,omitempty"`
	Composer    *RuleCombination     `yaml:"composer,omitempty"` // Optional: filter based on other signals
}

// ModelReasoningControl represents reasoning mode control on model level
type ModelReasoningControl struct {
	UseReasoning         *bool  `yaml:"use_reasoning"`                   // Pointer to detect missing field
	ReasoningDescription string `yaml:"reasoning_description,omitempty"` // Model-specific reasoning description
	ReasoningEffort      string `yaml:"reasoning_effort,omitempty"`      // Model-specific reasoning effort level (low, medium, high)
}

// DomainAwarePolicies represents policies that can be configured on a per-category basis
type DomainAwarePolicies struct {
	// System prompt optimization
	SystemPromptPolicy `yaml:",inline"`
	// Semantic caching policy
	SemanticCachingPolicy `yaml:",inline"`
	// Jailbreak detection policy
	JailbreakPolicy `yaml:",inline"`
	// PII detection policy
	PIIDetectionPolicy `yaml:",inline"`
}

// CategoryMetadata represents metadata for a category
type CategoryMetadata struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	// MMLUCategories optionally maps this generic category to one or more MMLU-Pro categories
	// used by the classifier model. When provided, classifier outputs will be translated
	// from these MMLU categories to this generic category name.
	MMLUCategories []string `yaml:"mmlu_categories,omitempty"`
}

type SystemPromptPolicy struct {
	// SystemPrompt is an optional category-specific system prompt automatically injected into requests
	SystemPrompt string `yaml:"system_prompt,omitempty"`
	// SystemPromptEnabled controls whether the system prompt should be injected for this category
	// Defaults to true when SystemPrompt is not empty
	SystemPromptEnabled *bool `yaml:"system_prompt_enabled,omitempty"`
	// SystemPromptMode controls how the system prompt is injected: "replace" (default) or "insert"
	// "replace": Replace any existing system message with the category-specific prompt
	// "insert": Prepend the category-specific prompt to the existing system message content
	SystemPromptMode string `yaml:"system_prompt_mode,omitempty"`
}

// SemanticCachingPolicy represents category-specific caching policies
type SemanticCachingPolicy struct {
	// SemanticCacheEnabled controls whether semantic caching is enabled for this category
	// If nil, inherits from global SemanticCache.Enabled setting
	SemanticCacheEnabled *bool `yaml:"semantic_cache_enabled,omitempty"`
	// SemanticCacheSimilarityThreshold defines the minimum similarity score for cache hits (0.0-1.0)
	// If nil, uses the global threshold from SemanticCache.SimilarityThreshold or BertModel.Threshold
	SemanticCacheSimilarityThreshold *float32 `yaml:"semantic_cache_similarity_threshold,omitempty"`
}

// JailbreakPolicy represents category-specific jailbreak detection policies
type JailbreakPolicy struct {
	// JailbreakEnabled controls whether jailbreak detection is enabled for this category
	// If nil, inherits from global PromptGuard.Enabled setting
	JailbreakEnabled *bool `yaml:"jailbreak_enabled,omitempty"`
	// JailbreakThreshold defines the confidence threshold for jailbreak detection (0.0-1.0)
	// If nil, uses the global threshold from PromptGuard.Threshold
	JailbreakThreshold *float32 `yaml:"jailbreak_threshold,omitempty"`
}

// PIIDetectionPolicy represents category-specific PII detection policies
type PIIDetectionPolicy struct {
	// PIIEnabled controls whether PII detection is enabled for this category
	// If nil, inherits from global PII detection enabled setting (based on classifier.pii_model configuration)
	PIIEnabled *bool `yaml:"pii_enabled,omitempty"`
	// PIIThreshold defines the confidence threshold for PII detection (0.0-1.0)
	// If nil, uses the global threshold from Classifier.PIIModel.Threshold
	PIIThreshold *float32 `yaml:"pii_threshold,omitempty"`
}

// FindExternalModelByRole searches for an external model configuration by its role
// Returns nil if no matching model is found
func (cfg *RouterConfig) FindExternalModelByRole(role string) *ExternalModelConfig {
	for i := range cfg.ExternalModels {
		if cfg.ExternalModels[i].ModelRole == role {
			return &cfg.ExternalModels[i]
		}
	}
	return nil
}
