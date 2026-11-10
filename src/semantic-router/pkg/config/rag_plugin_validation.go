package config

import "fmt"

// validateRAGSource checks a rag source's backend and its retrieval bounds.
func validateRAGSource(c *ContextSource) error {
	if c.Backend == "" {
		return fmt.Errorf("backend is required")
	}
	if err := validateRAGBackendConfig(c); err != nil {
		return err
	}
	if c.MaxContextLength != nil && *c.MaxContextLength < 0 {
		// Zero is a supported explicit value: the injection path truncates a
		// zero budget to the ellipsis marker, and issue #4327 keeps it valid.
		return fmt.Errorf("max_context_length must not be negative, got %d", *c.MaxContextLength)
	}
	if c.CacheTTLSeconds != nil && *c.CacheTTLSeconds < 1 {
		return fmt.Errorf("cache_ttl_seconds must be at least 1, got %d", *c.CacheTTLSeconds)
	}
	switch c.OnFailure {
	case "", RAGOnFailureSkip, RAGOnFailureBlock, RAGOnFailureWarn:
		return nil
	default:
		return fmt.Errorf("on_failure must be skip, block or warn, got %q", c.OnFailure)
	}
}

func validateRAGBackendConfig(c *ContextSource) error {
	switch c.Backend {
	case "milvus":
		return validateMilvusRAGBackend(c)
	case "qdrant":
		return validateQdrantRAGBackend(c)
	case "external_api":
		return validateExternalAPIRAGBackend(c)
	case "mcp":
		return validateMCPRAGBackend(c)
	case "openai":
		return validateOpenAIRAGBackend(c)
	case "hybrid":
		return validateHybridRAGBackend(c)
	case "vectorstore":
		return validateVectorStoreRAGBackend(c)
	default:
		return fmt.Errorf("unknown RAG backend %q", c.Backend)
	}
}

func validateVectorStoreRAGBackend(c *ContextSource) error {
	vectorStoreConfig, err := c.VectorStoreBackendConfig()
	if err != nil {
		return err
	}
	if vectorStoreConfig.VectorStoreID == "" {
		return fmt.Errorf("backend_config.vector_store_id is required for the vectorstore backend")
	}
	return nil
}

func validateMilvusRAGBackend(c *ContextSource) error {
	milvusConfig, err := c.MilvusBackendConfig()
	if err != nil {
		return err
	}
	if milvusConfig.Collection == "" {
		return fmt.Errorf("milvus collection name is required")
	}
	return nil
}

func validateQdrantRAGBackend(c *ContextSource) error {
	qdrantConfig, err := c.QdrantBackendConfig()
	if err != nil {
		return err
	}
	if qdrantConfig.Collection == "" {
		return fmt.Errorf("qdrant collection name is required")
	}
	return nil
}

func validateExternalAPIRAGBackend(c *ContextSource) error {
	apiConfig, err := c.ExternalAPIBackendConfig()
	if err != nil {
		return err
	}
	if apiConfig.Endpoint == "" {
		return fmt.Errorf("external API endpoint is required")
	}
	if apiConfig.RequestFormat == "" {
		return fmt.Errorf("request format is required for external API")
	}
	if apiConfig.MaxResponseBytes < 0 {
		return fmt.Errorf("external API max_response_bytes must be non-negative")
	}
	if apiConfig.Embedding != nil && !apiConfig.EmbedsQuery() {
		return fmt.Errorf("embedding: request format %s searches without a query vector; remove embedding", apiConfig.RequestFormat)
	}
	return nil
}

func validateMCPRAGBackend(c *ContextSource) error {
	mcpConfig, err := c.MCPBackendConfig()
	if err != nil {
		return err
	}
	if mcpConfig.ServerName == "" {
		return fmt.Errorf("MCP server name is required")
	}
	if mcpConfig.ToolName == "" {
		return fmt.Errorf("MCP tool name is required")
	}
	return fmt.Errorf("RAG backend %q is not available: the router has no MCP tool invoker", c.Backend)
}

func validateOpenAIRAGBackend(c *ContextSource) error {
	openaiConfig, err := c.OpenAIBackendConfig()
	if err != nil {
		return err
	}
	if openaiConfig.VectorStoreID == "" {
		return fmt.Errorf("vector store ID is required for OpenAI backend")
	}
	if openaiConfig.APIKey == "" {
		return fmt.Errorf("API key is required for OpenAI backend")
	}
	if openaiConfig.MaxResponseBytes < 0 {
		return fmt.Errorf("OpenAI max_response_bytes must be non-negative")
	}
	return nil
}

func validateHybridRAGBackend(c *ContextSource) error {
	hybridConfig, err := c.HybridBackendConfig()
	if err != nil {
		return err
	}
	if hybridConfig.Primary == "" {
		return fmt.Errorf("primary backend is required for hybrid RAG")
	}
	if hybridConfig.Primary == "mcp" || hybridConfig.Fallback == "mcp" {
		return fmt.Errorf("RAG backend %q is not available: the router has no MCP tool invoker", "mcp")
	}
	return nil
}
