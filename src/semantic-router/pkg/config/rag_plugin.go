package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RAGPluginConfig represents configuration for RAG (Retrieval-Augmented Generation) plugin
type RAGPluginConfig struct {
	// Enable RAG retrieval for this decision
	Enabled bool `json:"enabled" yaml:"enabled"`

	// Retrieval backend type: "milvus", "external_api", "mcp", "hybrid"
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
	// - "hybrid": HybridRAGConfig
	BackendConfig interface{} `json:"backend_config,omitempty" yaml:"backend_config,omitempty"`

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

// MilvusRAGConfig represents configuration for Milvus-based RAG retrieval
type MilvusRAGConfig struct {
	// Collection name for retrieval
	Collection string `json:"collection" yaml:"collection"`

	// Use existing Milvus cache connection if available
	// If true, reuses the connection from semantic cache
	ReuseCacheConnection bool `json:"reuse_cache_connection,omitempty" yaml:"reuse_cache_connection,omitempty"`

	// Field name containing document content
	// Default: "content"
	ContentField string `json:"content_field,omitempty" yaml:"content_field,omitempty"`

	// Field name containing metadata (optional)
	// If set, metadata will be included in retrieved context
	MetadataField string `json:"metadata_field,omitempty" yaml:"metadata_field,omitempty"`

	// Filter expression for Milvus query (optional)
	// Example: "domain == 'science' && published_date > '2024-01-01'"
	FilterExpression string `json:"filter_expression,omitempty" yaml:"filter_expression,omitempty"`
}

// ExternalAPIRAGConfig represents configuration for external API-based RAG retrieval
type ExternalAPIRAGConfig struct {
	// API endpoint URL
	Endpoint string `json:"endpoint" yaml:"endpoint"`

	// Authentication
	APIKey     string `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	AuthHeader string `json:"auth_header,omitempty" yaml:"auth_header,omitempty"` // e.g., "Authorization", "Api-Key"

	// Request format: "openai", "pinecone", "weaviate", "elasticsearch", "custom"
	RequestFormat string `json:"request_format" yaml:"request_format"`

	// Custom request template (for "custom" format)
	// Supports Go template syntax with variables: {{.Query}}, {{.TopK}}, {{.Threshold}}
	RequestTemplate string `json:"request_template,omitempty" yaml:"request_template,omitempty"`

	// Timeout in seconds
	TimeoutSeconds *int `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`

	// Additional headers to include in request
	Headers map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

// MCPRAGConfig represents configuration for MCP-based RAG retrieval
type MCPRAGConfig struct {
	// MCP server name (must be registered)
	ServerName string `json:"server_name" yaml:"server_name"`

	// Tool name to invoke for retrieval
	ToolName string `json:"tool_name" yaml:"tool_name"`

	// Tool arguments template
	// Supports variable substitution: ${user_content}, ${matched_domains}, ${top_k}
	ToolArguments map[string]interface{} `json:"tool_arguments,omitempty" yaml:"tool_arguments,omitempty"`

	// Timeout in seconds
	TimeoutSeconds *int `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
}

// HybridRAGConfig represents configuration for hybrid RAG with multiple backends
type HybridRAGConfig struct {
	// Primary backend to use first
	Primary string `json:"primary" yaml:"primary"` // "milvus", "external_api", "mcp"

	// Fallback backend if primary fails
	Fallback string `json:"fallback,omitempty" yaml:"fallback,omitempty"`

	// Primary backend configuration
	PrimaryConfig interface{} `json:"primary_config,omitempty" yaml:"primary_config,omitempty"`

	// Fallback backend configuration
	FallbackConfig interface{} `json:"fallback_config,omitempty" yaml:"fallback_config,omitempty"`

	// Strategy: "sequential" (try primary, then fallback) or "parallel" (try both, use best)
	Strategy string `json:"strategy,omitempty" yaml:"strategy,omitempty"`
}

// GetRAGConfig returns the RAG plugin configuration for a decision
func (d *Decision) GetRAGConfig() *RAGPluginConfig {
	config := d.GetPluginConfig("rag")
	if config == nil {
		return nil
	}

	result := &RAGPluginConfig{}
	if err := unmarshalPluginConfig(config, result); err != nil {
		logging.Errorf("Failed to unmarshal RAG config: %v", err)
		return nil
	}

	// Unmarshal backend-specific config based on Backend type
	if result.BackendConfig != nil && result.Backend != "" {
		var backendConfig interface{}
		switch result.Backend {
		case "milvus":
			backendConfig = &MilvusRAGConfig{}
		case "external_api":
			backendConfig = &ExternalAPIRAGConfig{}
		case "mcp":
			backendConfig = &MCPRAGConfig{}
		case "hybrid":
			backendConfig = &HybridRAGConfig{}
		default:
			logging.Warnf("Unknown RAG backend type: %s", result.Backend)
			return result
		}

		if err := unmarshalPluginConfig(result.BackendConfig, backendConfig); err != nil {
			logging.Errorf("Failed to unmarshal RAG backend config for %s: %v", result.Backend, err)
		} else {
			result.BackendConfig = backendConfig
		}
	}

	return result
}

// Validate validates the RAG plugin configuration
func (c *RAGPluginConfig) Validate() error {
	if !c.Enabled {
		return nil // Disabled configs don't need validation
	}

	if c.Backend == "" {
		return fmt.Errorf("RAG backend is required when enabled")
	}

	// Validate backend-specific config
	switch c.Backend {
	case "milvus":
		if milvusConfig, ok := c.BackendConfig.(*MilvusRAGConfig); ok {
			if milvusConfig.Collection == "" {
				return fmt.Errorf("Milvus collection name is required")
			}
		}
	case "external_api":
		if apiConfig, ok := c.BackendConfig.(*ExternalAPIRAGConfig); ok {
			if apiConfig.Endpoint == "" {
				return fmt.Errorf("External API endpoint is required")
			}
			if apiConfig.RequestFormat == "" {
				return fmt.Errorf("Request format is required for external API")
			}
		}
	case "mcp":
		if mcpConfig, ok := c.BackendConfig.(*MCPRAGConfig); ok {
			if mcpConfig.ServerName == "" {
				return fmt.Errorf("MCP server name is required")
			}
			if mcpConfig.ToolName == "" {
				return fmt.Errorf("MCP tool name is required")
			}
		}
	case "hybrid":
		if hybridConfig, ok := c.BackendConfig.(*HybridRAGConfig); ok {
			if hybridConfig.Primary == "" {
				return fmt.Errorf("Primary backend is required for hybrid RAG")
			}
		}
	default:
		return fmt.Errorf("Unknown RAG backend: %s", c.Backend)
	}

	// Validate similarity threshold
	if c.SimilarityThreshold != nil {
		threshold := *c.SimilarityThreshold
		if threshold < 0.0 || threshold > 1.0 {
			return fmt.Errorf("Similarity threshold must be between 0.0 and 1.0, got %.2f", threshold)
		}
	}

	// Validate top-k
	if c.TopK != nil && *c.TopK <= 0 {
		return fmt.Errorf("TopK must be greater than 0, got %d", *c.TopK)
	}

	// Validate injection mode
	if c.InjectionMode != "" && c.InjectionMode != "tool_role" && c.InjectionMode != "system_prompt" {
		return fmt.Errorf("Injection mode must be 'tool_role' or 'system_prompt', got %s", c.InjectionMode)
	}

	// Validate on_failure
	if c.OnFailure != "" && c.OnFailure != "skip" && c.OnFailure != "block" && c.OnFailure != "warn" {
		return fmt.Errorf("OnFailure must be 'skip', 'block', or 'warn', got %s", c.OnFailure)
	}

	return nil
}
