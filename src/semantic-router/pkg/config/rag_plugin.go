package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"

// RAGPluginConfig represents configuration for RAG (Retrieval-Augmented Generation) plugin
type RAGPluginConfig struct {
	// Enable RAG retrieval for this decision
	Enabled bool `json:"enabled" yaml:"enabled"`

	// Retrieval backend type: "milvus", "external_api", "mcp", "openai", "hybrid"
	// - "openai": Use OpenAI's file_search tool with vector stores (Responses API workflow)
	Backend string `json:"backend" yaml:"backend"`

	// Similarity threshold for retrieval (0.0-1.0)
	// Only documents with similarity >= threshold will be retrieved
	SimilarityThreshold *float32 `json:"similarity_threshold,omitempty" yaml:"similarity_threshold,omitempty"`

	// Number of top-k documents to retrieve
	TopK *int `json:"top_k,omitempty" yaml:"top_k,omitempty"`

	// Maximum context length to inject (in characters)
	// If retrieved context exceeds this, it will be truncated
	MaxContextLength *int `json:"max_context_length,omitempty" yaml:"max_context_length,omitempty"`

	// Context injection mode: "tool_role" (default) or "system_prompt"
	// - "tool_role": Inject as tool role messages (compatible with hallucination detection)
	// - "system_prompt": Prepend to system prompt
	InjectionMode string `json:"injection_mode,omitempty" yaml:"injection_mode,omitempty"`

	// Backend-specific configuration
	// Structure depends on Backend type:
	// - "milvus": MilvusRAGConfig
	// - "external_api": ExternalAPIRAGConfig
	// - "mcp": MCPRAGConfig
	// - "openai": OpenAIRAGConfig
	// - "hybrid": HybridRAGConfig
	BackendConfig *StructuredPayload `json:"backend_config,omitempty" yaml:"backend_config,omitempty"`

	// Fallback behavior when retrieval fails
	// - "skip" (default): Continue without context, log warning
	// - "block": Return error response
	// - "warn": Continue with warning header
	OnFailure string `json:"on_failure,omitempty" yaml:"on_failure,omitempty"`

	// Cache retrieved results to avoid redundant searches
	// Uses in-memory cache with TTL
	CacheResults bool `json:"cache_results,omitempty" yaml:"cache_results,omitempty"`

	// TTL for cached retrieval results (seconds)
	// Only used if CacheResults is true
	CacheTTLSeconds *int `json:"cache_ttl_seconds,omitempty" yaml:"cache_ttl_seconds,omitempty"`

	// Minimum confidence threshold for triggering retrieval
	// Only retrieve if signal confidence >= this threshold
	// If not set, retrieval is triggered regardless of confidence
	MinConfidenceThreshold *float32 `json:"min_confidence_threshold,omitempty" yaml:"min_confidence_threshold,omitempty"`
}

// GetRAGConfig returns the RAG plugin configuration for a decision
func (d *Decision) GetRAGConfig() *RAGPluginConfig {
	plugin := d.GetPlugin("rag")
	if plugin == nil || plugin.Configuration == nil {
		return nil
	}

	result := &RAGPluginConfig{}
	if err := UnmarshalPluginConfig(plugin.Configuration, result); err != nil {
		logging.Errorf("Failed to unmarshal RAG config: %v", err)
		return nil
	}
	return result
}
