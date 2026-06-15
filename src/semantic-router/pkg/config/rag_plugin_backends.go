package config

import "fmt"

// MilvusRAGConfig represents configuration for Milvus-based RAG retrieval.
type MilvusRAGConfig struct {
	Collection           string `json:"collection" yaml:"collection"`
	ReuseCacheConnection bool   `json:"reuse_cache_connection,omitempty" yaml:"reuse_cache_connection,omitempty"`
	ContentField         string `json:"content_field,omitempty" yaml:"content_field,omitempty"`
	MetadataField        string `json:"metadata_field,omitempty" yaml:"metadata_field,omitempty"`
	FilterExpression     string `json:"filter_expression,omitempty" yaml:"filter_expression,omitempty"`
}

// ExternalAPIRAGConfig represents configuration for external API-based RAG retrieval.
type ExternalAPIRAGConfig struct {
	Endpoint        string            `json:"endpoint" yaml:"endpoint"`
	APIKey          string            `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	AuthHeader      string            `json:"auth_header,omitempty" yaml:"auth_header,omitempty"`
	RequestFormat   string            `json:"request_format" yaml:"request_format"`
	RequestTemplate string            `json:"request_template,omitempty" yaml:"request_template,omitempty"`
	TimeoutSeconds  *int              `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
	Headers         map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

// MCPRAGConfig represents configuration for MCP-based RAG retrieval.
type MCPRAGConfig struct {
	ServerName     string             `json:"server_name" yaml:"server_name"`
	ToolName       string             `json:"tool_name" yaml:"tool_name"`
	ToolArguments  *StructuredPayload `json:"tool_arguments,omitempty" yaml:"tool_arguments,omitempty"`
	TimeoutSeconds *int               `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
}

// OpenAIRAGConfig represents configuration for OpenAI file_search-based RAG retrieval.
type OpenAIRAGConfig struct {
	VectorStoreID  string             `json:"vector_store_id" yaml:"vector_store_id"`
	BaseURL        string             `json:"base_url,omitempty" yaml:"base_url,omitempty"`
	APIKey         string             `json:"api_key" yaml:"api_key"`
	MaxNumResults  *int               `json:"max_num_results,omitempty" yaml:"max_num_results,omitempty"`
	FileIDs        []string           `json:"file_ids,omitempty" yaml:"file_ids,omitempty"`
	Filter         *StructuredPayload `json:"filter,omitempty" yaml:"filter,omitempty"`
	TimeoutSeconds *int               `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
	WorkflowMode   string             `json:"workflow_mode,omitempty" yaml:"workflow_mode,omitempty"`
}

// HybridRAGConfig represents configuration for hybrid RAG with multiple backends.
type HybridRAGConfig struct {
	Primary        string             `json:"primary" yaml:"primary"`
	Fallback       string             `json:"fallback,omitempty" yaml:"fallback,omitempty"`
	PrimaryConfig  *StructuredPayload `json:"primary_config,omitempty" yaml:"primary_config,omitempty"`
	FallbackConfig *StructuredPayload `json:"fallback_config,omitempty" yaml:"fallback_config,omitempty"`
	Strategy       string             `json:"strategy,omitempty" yaml:"strategy,omitempty"`
}

// VectorStoreRAGConfig contains configuration for the local vectorstore RAG backend.
type VectorStoreRAGConfig struct {
	VectorStoreID string   `json:"vector_store_id" yaml:"vector_store_id"`
	FileIDs       []string `json:"file_ids,omitempty" yaml:"file_ids,omitempty"`
}

// QdrantRAGConfig represents configuration for Qdrant-based RAG retrieval.
type QdrantRAGConfig struct {
	Collection           string `json:"collection" yaml:"collection"`
	ReuseCacheConnection bool   `json:"reuse_cache_connection,omitempty" yaml:"reuse_cache_connection,omitempty"`
	ContentField         string `json:"content_field,omitempty" yaml:"content_field,omitempty"`
}

func (c *RAGPluginConfig) MilvusBackendConfig() (*MilvusRAGConfig, error) {
	return decodeRAGBackendConfig[MilvusRAGConfig](c, "milvus")
}

func (c *RAGPluginConfig) QdrantBackendConfig() (*QdrantRAGConfig, error) {
	return decodeRAGBackendConfig[QdrantRAGConfig](c, "qdrant")
}

func (c *RAGPluginConfig) ExternalAPIBackendConfig() (*ExternalAPIRAGConfig, error) {
	return decodeRAGBackendConfig[ExternalAPIRAGConfig](c, "external_api")
}

func (c *RAGPluginConfig) MCPBackendConfig() (*MCPRAGConfig, error) {
	return decodeRAGBackendConfig[MCPRAGConfig](c, "mcp")
}

func (c *RAGPluginConfig) OpenAIBackendConfig() (*OpenAIRAGConfig, error) {
	return decodeRAGBackendConfig[OpenAIRAGConfig](c, "openai")
}

func (c *RAGPluginConfig) VectorStoreBackendConfig() (*VectorStoreRAGConfig, error) {
	return decodeRAGBackendConfig[VectorStoreRAGConfig](c, "vectorstore")
}

func (c *RAGPluginConfig) HybridBackendConfig() (*HybridRAGConfig, error) {
	return decodeRAGBackendConfig[HybridRAGConfig](c, "hybrid")
}

func (c *MCPRAGConfig) ToolArgumentMap() (map[string]interface{}, error) {
	if c == nil || c.ToolArguments == nil {
		return map[string]interface{}{}, nil
	}
	return c.ToolArguments.AsStringMap()
}

func (c *OpenAIRAGConfig) FilterMap() (map[string]interface{}, error) {
	if c == nil || c.Filter == nil {
		return nil, nil
	}
	return c.Filter.AsStringMap()
}

func decodeRAGBackendConfig[T any](cfg *RAGPluginConfig, expectedBackend string) (*T, error) {
	if cfg == nil {
		return nil, fmt.Errorf("RAG config is nil")
	}
	if cfg.Backend != expectedBackend {
		return nil, fmt.Errorf("expected RAG backend %q, got %q", expectedBackend, cfg.Backend)
	}
	if cfg.BackendConfig == nil {
		return nil, fmt.Errorf("BackendConfig is required for backend %q", expectedBackend)
	}
	result := new(T)
	if err := cfg.BackendConfig.DecodeInto(result); err != nil {
		return nil, err
	}
	return result, nil
}
