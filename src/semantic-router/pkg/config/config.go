package config

import (
	"encoding/json"
	"fmt"

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

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// ConfigSource specifies where to load dynamic configuration from (file or kubernetes)
	// +optional
	// +kubebuilder:default=file
	ConfigSource ConfigSource `yaml:"config_source,omitempty"`

	/*
		Static: Global Configuration
		Timing: Should be handled when starting the router.
	*/
	// Inline models configuration
	InlineModels `yaml:",inline"`
	// Semantic cache configuration
	SemanticCache `yaml:"semantic_cache"`
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
}

// IntelligentRouting represents the configuration for intelligent routing
type IntelligentRouting struct {
	// Keyword-based classification rules
	KeywordRules []KeywordRule `yaml:"keyword_rules,omitempty"`

	// Embedding-based classification rules
	EmbeddingRules []EmbeddingRule `yaml:"embedding_rules,omitempty"`

	// Categories for domain classification (only metadata, used by domain rules)
	Categories []Category `yaml:"categories"`

	// Decisions for routing logic (combines rules with AND/OR operators)
	Decisions []Decision `yaml:"decisions,omitempty"`

	// Strategy for selecting decision when multiple decisions match
	// "priority" - select decision with highest priority
	// "confidence" - select decision with highest confidence score
	Strategy string `yaml:"strategy,omitempty"`

	// Reasoning mode configuration
	ReasoningConfig `yaml:",inline"`
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
	CategoryMappingPath string  `yaml:"category_mapping_path"`
}

type PIIModel struct {
	ModelID        string  `yaml:"model_id"`
	Threshold      float32 `yaml:"threshold"`
	UseCPU         bool    `yaml:"use_cpu"`
	PIIMappingPath string  `yaml:"pii_mapping_path"`
}

type EmbeddingModels struct {
	// Path to Qwen3-Embedding-0.6B model directory
	Qwen3ModelPath string `yaml:"qwen3_model_path"`
	// Path to EmbeddingGemma-300M model directory
	GemmaModelPath string `yaml:"gemma_model_path"`
	// Use CPU for inference (default: true, auto-detect GPU if available)
	UseCPU bool `yaml:"use_cpu"`
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

	// Path to backend-specific configuration file
	BackendConfigPath string `yaml:"backend_config_path,omitempty"`

	// Embedding model to use for semantic similarity ("bert", "qwen3", "gemma")
	// - "bert": Fast, 384-dim, good for short texts (default)
	// - "qwen3": High quality, 1024-dim, supports 32K context
	// - "gemma": Balanced, 768-dim, supports 8K context
	// Default: "bert"
	EmbeddingModel string `yaml:"embedding_model,omitempty"`
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

	// Path to the jailbreak type mapping file
	JailbreakMappingPath string `yaml:"jailbreak_mapping_path"`

	// Use vLLM REST API instead of Candle for guardrail/safety checks
	// When true, ModelID, UseCPU, and UseModernBERT are ignored
	// When false (default), uses Candle-based classification
	UseVLLM bool `yaml:"use_vllm,omitempty"`

	// Dedicated vLLM endpoint configuration for PromptGuard
	// This is separate from vllm_endpoints (which are for backend inference)
	ClassifierVLLMEndpoint ClassifierVLLMEndpoint `yaml:"classifier_vllm_endpoint,omitempty"`

	// Model name on vLLM server (e.g., "Qwen/Qwen3Guard-Gen-0.6B")
	VLLMModelName string `yaml:"vllm_model_name,omitempty"`

	// Timeout for vLLM API calls in seconds
	// Default: 30 seconds if not specified
	VLLMTimeoutSeconds int `yaml:"vllm_timeout_seconds,omitempty"`

	// Response parser type (optional, auto-detected from model name if not set)
	// Options: "qwen3guard", "json", "simple", "auto"
	// "auto" tries multiple parsers (OR logic)
	ResponseParserType string `yaml:"response_parser_type,omitempty"`
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

	// ModelRefs contains model references for this decision (currently only supports one model)
	ModelRefs []ModelRef `yaml:"modelRefs,omitempty"`

	// Plugins contains policy configurations applied after rule matching
	Plugins []DecisionPlugin `yaml:"plugins,omitempty"`
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
	// Type specifies the plugin type: "semantic-cache", "jailbreak", "pii", "system_prompt"
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
}

// JailbreakPluginConfig represents configuration for jailbreak plugin
type JailbreakPluginConfig struct {
	Enabled   bool     `json:"enabled" yaml:"enabled"`
	Threshold *float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
}

// PIIPluginConfig represents configuration for pii plugin
type PIIPluginConfig struct {
	Enabled   bool     `json:"enabled" yaml:"enabled"`
	Threshold *float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`

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

// RuleCombination defines how to combine multiple rule conditions with AND/OR operators
type RuleCombination struct {
	// Operator specifies how to combine conditions: "AND" or "OR"
	Operator string `yaml:"operator"`

	// Conditions is the list of rule references to evaluate
	Conditions []RuleCondition `yaml:"conditions"`
}

// RuleCondition references a specific rule by type and name
type RuleCondition struct {
	// Type specifies the rule type: "keyword", "embedding", or "domain"
	Type string `yaml:"type"`

	// Name is the name of the rule to reference
	Name string `yaml:"name"`
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
