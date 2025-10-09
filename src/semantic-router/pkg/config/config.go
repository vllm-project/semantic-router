package config

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"gopkg.in/yaml.v3"
)

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// BERT model configuration for Candle BERT similarity comparison
	BertModel struct {
		ModelID   string  `yaml:"model_id"`
		Threshold float32 `yaml:"threshold"`
		UseCPU    bool    `yaml:"use_cpu"`
	} `yaml:"bert_model"`

	// Classifier configuration for text classification
	Classifier struct {
		CategoryModel struct {
			ModelID             string  `yaml:"model_id"`
			Threshold           float32 `yaml:"threshold"`
			UseCPU              bool    `yaml:"use_cpu"`
			UseModernBERT       bool    `yaml:"use_modernbert"`
			CategoryMappingPath string  `yaml:"category_mapping_path"`
		} `yaml:"category_model"`
		PIIModel struct {
			ModelID        string  `yaml:"model_id"`
			Threshold      float32 `yaml:"threshold"`
			UseCPU         bool    `yaml:"use_cpu"`
			PIIMappingPath string  `yaml:"pii_mapping_path"`
		} `yaml:"pii_model"`
	} `yaml:"classifier"`

	// Categories for routing queries
	Categories []Category `yaml:"categories"`

	// Rag Configurations for knowledge bases
	Rag RagConfig `yaml:"rag,omitempty"`

	// Default LLM model to use if no match is found
	DefaultModel string `yaml:"default_model"`

	// Default reasoning effort level (low, medium, high) when not specified per category
	DefaultReasoningEffort string `yaml:"default_reasoning_effort,omitempty"`

	// Reasoning family configurations to define how different model families handle reasoning syntax
	ReasoningFamilies map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`

	// Semantic cache configuration
	SemanticCache struct {
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
	} `yaml:"semantic_cache"`

	// Prompt guard configuration
	PromptGuard PromptGuardConfig `yaml:"prompt_guard"`

	// Model parameters configuration
	ModelConfig map[string]ModelParams `yaml:"model_config"`

	// Tools configuration for automatic tool selection
	Tools ToolsConfig `yaml:"tools"`

	// vLLM endpoints configuration for multiple backend support
	VLLMEndpoints []VLLMEndpoint `yaml:"vllm_endpoints"`

	// API configuration for classification endpoints
	API APIConfig `yaml:"api"`
}

// APIConfig represents configuration for API endpoints
type APIConfig struct {
	// Batch classification configuration (zero-config auto-discovery)
	BatchClassification struct {
		// Metrics configuration for batch classification monitoring
		Metrics BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
	} `yaml:"batch_classification"`
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

	// Model ID for the jailbreak classification model
	ModelID string `yaml:"model_id"`

	// Threshold for jailbreak detection (0.0-1.0)
	Threshold float32 `yaml:"threshold"`

	// Use CPU for inference
	UseCPU bool `yaml:"use_cpu"`

	// Use ModernBERT for jailbreak detection
	UseModernBERT bool `yaml:"use_modernbert"`

	// Path to the jailbreak type mapping file
	JailbreakMappingPath string `yaml:"jailbreak_mapping_path"`
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

// VLLMEndpoint represents a vLLM backend endpoint configuration
type VLLMEndpoint struct {
	// Name identifier for the endpoint
	Name string `yaml:"name"`

	// Address of the vLLM endpoint
	Address string `yaml:"address"`

	// Port of the vLLM endpoint
	Port int `yaml:"port"`

	// List of models served by this endpoint
	Models []string `yaml:"models"`

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
	// PII policy configuration for this model
	PIIPolicy PIIPolicy `yaml:"pii_policy,omitempty"`

	// Preferred endpoints for this model (optional)
	PreferredEndpoints []string `yaml:"preferred_endpoints,omitempty"`

	// Optional pricing used for cost computation
	Pricing ModelPricing `yaml:"pricing,omitempty"`

	// Reasoning family for this model (e.g., "deepseek", "qwen3", "gpt-oss")
	// If empty, the model doesn't support reasoning mode
	ReasoningFamily string `yaml:"reasoning_family,omitempty"`
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

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SemanticCache.SimilarityThreshold != nil {
		return *c.SemanticCache.SimilarityThreshold
	}
	return c.BertModel.Threshold
}

// ModelScore associates an LLM with its selection weight and reasoning flag within a category.
type ModelScore struct {
	Model        string  `yaml:"model"`
	Score        float64 `yaml:"score"`
	UseReasoning *bool   `yaml:"use_reasoning"` // Pointer to detect missing field
}

// Category represents a category for routing queries
type Category struct {
	Name                 string       `yaml:"name"`
	Description          string       `yaml:"description,omitempty"`
	ReasoningDescription string       `yaml:"reasoning_description,omitempty"`
	ReasoningEffort      string       `yaml:"reasoning_effort,omitempty"` // Configurable reasoning effort level (low, medium, high)
	RagStrategy			 string       `yaml:"rag_strategy,omitempty"` // Configurable rag strategy for a category (never, adaptive, always)
	RagDescription       string       `yaml:"rag_description,omitempty"`
	// TODO: Knowledge Domains?
	ModelScores          []ModelScore `yaml:"model_scores"`
	// MMLUCategories optionally maps this generic category to one or more MMLU-Pro categories
	// used by the classifier model. When provided, classifier outputs will be translated
	// from these MMLU categories to this generic category name.
	MMLUCategories []string `yaml:"mmlu_categories,omitempty"`
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

type RagConfig struct {
	Enabled bool `yaml:"enabled"`
	DefaultStrategy string `yaml:"rag_strategy,omitempty"` // Configurable rag strategy for a category (never, adaptive, always)
	KnowledgeBases []KnowledgeBase `yaml:"knowledge_bases,omitempty"`
	RetrievalParams RetrievalParams `yaml:"retrieval_params,omitempty"` // Should this be per knowledge base? (i.e in some cases we want to retrieve more chunks that others)
}

type KnowledgeBase struct {
	Name string `yaml:"name"`
	Type string `yaml:"type"` // VectorDB provider
	Endpoint string `yaml:"endpoint"`
	Collection string `yaml:"collection"`
	EmbeddingModel string `yaml:"embedding_model"`
	ApiKeyEnv string `yaml:"api_key_env,omitempty"`
	Index string `yaml:"index,omitempty"`
}

type RetrievalParams struct {
	TopK int `yaml:"top_k,omitempty"`
	SimilarityThreshold float32 `yaml:"similarity_threshold,omitempty"`
	MaxContextLength int `yaml:"max_context_length,omitempty"`
	RerankEnabled bool `yaml:"rerank_enabled,omitempty"`
}

// TODO: Reranking config?

// Legacy types - can be removed once migration is complete

// GetModelReasoningFamily returns the reasoning family configuration for a given model name
func (rc *RouterConfig) GetModelReasoningFamily(modelName string) *ReasoningFamilyConfig {
	if rc == nil || rc.ModelConfig == nil || rc.ReasoningFamilies == nil {
		return nil
	}

	// Look up the model in model_config
	modelParams, exists := rc.ModelConfig[modelName]
	if !exists || modelParams.ReasoningFamily == "" {
		return nil
	}

	// Look up the reasoning family configuration
	familyConfig, exists := rc.ReasoningFamilies[modelParams.ReasoningFamily]
	if !exists {
		return nil
	}

	return &familyConfig
}

// Legacy functions - can be removed once migration is complete

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
	configMu   sync.RWMutex
)

// LoadConfig loads the configuration from the specified YAML file once and caches it globally.
func LoadConfig(configPath string) (*RouterConfig, error) {
	configOnce.Do(func() {
		cfg, err := ParseConfigFile(configPath)
		if err != nil {
			configErr = err
			return
		}
		configMu.Lock()
		config = cfg
		configMu.Unlock()
	})
	if configErr != nil {
		return nil, configErr
	}
	configMu.RLock()
	defer configMu.RUnlock()
	return config, nil
}

// BoolPtr returns a pointer to a bool value (helper for tests and config)
func BoolPtr(b bool) *bool {
	return &b
}

// validateConfigStructure performs additional validation on the parsed config
func validateConfigStructure(cfg *RouterConfig) error {
	// Ensure all categories have at least one model with scores
	for _, category := range cfg.Categories {
		if len(category.ModelScores) == 0 {
			return fmt.Errorf("category '%s' has no model_scores defined - each category must have at least one model", category.Name)
		}

		// Validate each model score has the required fields
		for i, modelScore := range category.ModelScores {
			if modelScore.Model == "" {
				return fmt.Errorf("category '%s', model_scores[%d]: model name cannot be empty", category.Name, i)
			}
			if modelScore.Score <= 0 {
				return fmt.Errorf("category '%s', model '%s': score must be greater than 0, got %f", category.Name, modelScore.Model, modelScore.Score)
			}
			if modelScore.UseReasoning == nil {
				return fmt.Errorf("category '%s', model '%s': missing required field 'use_reasoning'", category.Name, modelScore.Model)
			}
		}
	}

	// Validate vLLM endpoints address formats
	if err := validateVLLMEndpoints(cfg.VLLMEndpoints); err != nil {
		return err
	}

	return nil
}

// ParseConfigFile parses the YAML config file without touching the global cache.
func ParseConfigFile(configPath string) (*RouterConfig, error) {
	// Resolve symlinks to handle Kubernetes ConfigMap mounts
	resolved, _ := filepath.EvalSymlinks(configPath)
	if resolved == "" {
		resolved = configPath
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := &RouterConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Validation after parsing
	if err := validateConfigStructure(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// ReplaceGlobalConfig replaces the globally cached config. It is safe for concurrent readers.
func ReplaceGlobalConfig(newCfg *RouterConfig) {
	configMu.Lock()
	defer configMu.Unlock()
	config = newCfg
	// Do not reset configOnce to avoid racing re-parses via LoadConfig; callers should use ParseConfigFile for fresher reads.
	configErr = nil
}

// GetConfig returns the current configuration
func GetConfig() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}

// GetCategoryDescriptions returns all category descriptions for similarity matching
func (c *RouterConfig) GetCategoryDescriptions() []string {
	var descriptions []string
	for _, category := range c.Categories {
		if category.Description != "" {
			descriptions = append(descriptions, category.Description)
		} else {
			// Use category name if no description is available
			descriptions = append(descriptions, category.Name)
		}
	}
	return descriptions
}

// GetModelForCategoryIndex returns the best LLM model name for the category at the given index
func (c *RouterConfig) GetModelForCategoryIndex(index int) string {
	if index < 0 || index >= len(c.Categories) {
		return c.DefaultModel
	}

	category := c.Categories[index]
	if len(category.ModelScores) > 0 {
		return category.ModelScores[0].Model
	}

	// Fall back to default model if category has no models
	return c.DefaultModel
}

// GetModelPricing returns pricing per 1M tokens and its currency for the given model.
// The currency indicates the unit of the returned rates (e.g., "USD").
func (c *RouterConfig) GetModelPricing(modelName string) (promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	if modelConfig, okc := c.ModelConfig[modelName]; okc {
		p := modelConfig.Pricing
		if p.PromptPer1M != 0 || p.CompletionPer1M != 0 {
			cur := p.Currency
			if cur == "" {
				cur = "USD"
			}
			return p.PromptPer1M, p.CompletionPer1M, cur, true
		}
	}
	return 0, 0, "", false
}

// GetModelPIIPolicy returns the PII policy for a given model
// If the model is not found in the config, returns a default policy that allows all PII
func (c *RouterConfig) GetModelPIIPolicy(modelName string) PIIPolicy {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.PIIPolicy
	}
	// Default policy allows all PII
	return PIIPolicy{
		AllowByDefault: true,
		PIITypes:       []string{},
	}
}

// IsModelAllowedForPIIType checks if a model is allowed to process a specific PII type
func (c *RouterConfig) IsModelAllowedForPIIType(modelName string, piiType string) bool {
	policy := c.GetModelPIIPolicy(modelName)

	// If allow_by_default is true, all PII types are allowed unless explicitly denied
	if policy.AllowByDefault {
		return true
	}

	// If allow_by_default is false, only explicitly allowed PII types are permitted
	return slices.Contains(policy.PIITypes, piiType)
}

// IsModelAllowedForPIITypes checks if a model is allowed to process any of the given PII types
func (c *RouterConfig) IsModelAllowedForPIITypes(modelName string, piiTypes []string) bool {
	for _, piiType := range piiTypes {
		if !c.IsModelAllowedForPIIType(modelName, piiType) {
			return false
		}
	}
	return true
}

// IsPIIClassifierEnabled checks if PII classification is enabled
func (c *RouterConfig) IsPIIClassifierEnabled() bool {
	return c.Classifier.PIIModel.ModelID != "" && c.Classifier.PIIModel.PIIMappingPath != ""
}

// IsCategoryClassifierEnabled checks if category classification is enabled
func (c *RouterConfig) IsCategoryClassifierEnabled() bool {
	return c.Classifier.CategoryModel.ModelID != "" && c.Classifier.CategoryModel.CategoryMappingPath != ""
}

// GetPromptGuardConfig returns the prompt guard configuration
func (c *RouterConfig) GetPromptGuardConfig() PromptGuardConfig {
	return c.PromptGuard
}

// IsPromptGuardEnabled checks if prompt guard jailbreak detection is enabled
func (c *RouterConfig) IsPromptGuardEnabled() bool {
	return c.PromptGuard.Enabled && c.PromptGuard.ModelID != "" && c.PromptGuard.JailbreakMappingPath != ""
}

// GetEndpointsForModel returns all endpoints that can serve the specified model
// If the model has preferred endpoints configured, returns only those endpoints that are available
// Otherwise, returns all endpoints that list the model in their Models array
func (c *RouterConfig) GetEndpointsForModel(modelName string) []VLLMEndpoint {
	var availableEndpoints []VLLMEndpoint

	// First, find all endpoints that can serve this model
	for _, endpoint := range c.VLLMEndpoints {
		if slices.Contains(endpoint.Models, modelName) {
			availableEndpoints = append(availableEndpoints, endpoint)
		}
	}

	// Check if model has preferred endpoints configured
	if modelConfig, ok := c.ModelConfig[modelName]; ok && len(modelConfig.PreferredEndpoints) > 0 {
		var preferredEndpoints []VLLMEndpoint
		for _, endpoint := range availableEndpoints {
			if slices.Contains(modelConfig.PreferredEndpoints, endpoint.Name) {
				preferredEndpoints = append(preferredEndpoints, endpoint)
			}
		}
		if len(preferredEndpoints) > 0 {
			return preferredEndpoints
		}
	}

	return availableEndpoints
}

// GetEndpointByName returns the endpoint with the specified name
func (c *RouterConfig) GetEndpointByName(name string) (*VLLMEndpoint, bool) {
	for _, endpoint := range c.VLLMEndpoints {
		if endpoint.Name == name {
			return &endpoint, true
		}
	}
	return nil, false
}

// GetAllModels returns a list of all models available across all endpoints
func (c *RouterConfig) GetAllModels() []string {
	modelSet := make(map[string]bool)
	var models []string

	for _, endpoint := range c.VLLMEndpoints {
		for _, model := range endpoint.Models {
			if !modelSet[model] {
				modelSet[model] = true
				models = append(models, model)
			}
		}
	}

	return models
}

// SelectBestEndpointForModel selects the best endpoint for a model based on weights and availability
// Returns the endpoint name and whether selection was successful
func (c *RouterConfig) SelectBestEndpointForModel(modelName string) (string, bool) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", false
	}

	// If only one endpoint, return it
	if len(endpoints) == 1 {
		return endpoints[0].Name, true
	}

	// Select endpoint with highest weight
	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	return bestEndpoint.Name, true
}

// SelectBestEndpointAddressForModel selects the best endpoint for a model and returns the address:port
// Returns the endpoint address:port string and whether selection was successful
func (c *RouterConfig) SelectBestEndpointAddressForModel(modelName string) (string, bool) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", false
	}

	// If only one endpoint, return it
	if len(endpoints) == 1 {
		return fmt.Sprintf("%s:%d", endpoints[0].Address, endpoints[0].Port), true
	}

	// Select endpoint with highest weight
	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	return fmt.Sprintf("%s:%d", bestEndpoint.Address, bestEndpoint.Port), true
}

// GetModelReasoningForCategory returns whether a specific model supports reasoning in a given category
func (c *RouterConfig) GetModelReasoningForCategory(categoryName string, modelName string) bool {
	for _, category := range c.Categories {
		if category.Name == categoryName {
			for _, modelScore := range category.ModelScores {
				if modelScore.Model == modelName {
					return modelScore.UseReasoning != nil && *modelScore.UseReasoning
				}
			}
		}
	}
	return false // Default to false if category or model not found
}

// GetBestModelForCategory returns the best scoring model for a given category
func (c *RouterConfig) GetBestModelForCategory(categoryName string) (string, bool) {
	for _, category := range c.Categories {
		if category.Name == categoryName {
			if len(category.ModelScores) > 0 {
				useReasoning := category.ModelScores[0].UseReasoning != nil && *category.ModelScores[0].UseReasoning
				return category.ModelScores[0].Model, useReasoning
			}
		}
	}
	return "", false // Return empty string and false if category not found or has no models
}

// ValidateEndpoints validates that all configured models have at least one endpoint
func (c *RouterConfig) ValidateEndpoints() error {
	// Get all models from categories
	allCategoryModels := make(map[string]bool)
	for _, category := range c.Categories {
		for _, modelScore := range category.ModelScores {
			allCategoryModels[modelScore.Model] = true
		}
	}

	// Add default model
	if c.DefaultModel != "" {
		allCategoryModels[c.DefaultModel] = true
	}

	// Check that each model has at least one endpoint
	for model := range allCategoryModels {
		endpoints := c.GetEndpointsForModel(model)
		if len(endpoints) == 0 {
			return fmt.Errorf("model '%s' has no available endpoints", model)
		}
	}

	return nil
}

// IsSystemPromptEnabled returns whether system prompt injection is enabled for a category
func (c *Category) IsSystemPromptEnabled() bool {
	// If SystemPromptEnabled is explicitly set, use that value
	if c.SystemPromptEnabled != nil {
		return *c.SystemPromptEnabled
	}
	// Default to true if SystemPrompt is not empty
	return c.SystemPrompt != ""
}

// GetSystemPromptMode returns the system prompt injection mode, defaulting to "replace"
func (c *Category) GetSystemPromptMode() string {
	if c.SystemPromptMode == "" {
		return "replace" // Default mode
	}
	return c.SystemPromptMode
}

// GetCategoryByName returns a category by name
func (c *RouterConfig) GetCategoryByName(name string) *Category {
	for i := range c.Categories {
		if c.Categories[i].Name == name {
			return &c.Categories[i]
		}
	}
	return nil
}
