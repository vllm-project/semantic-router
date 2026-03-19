package config

import "fmt"

// Validate validates the RAG plugin configuration.
func (c *RAGPluginConfig) Validate() error {
	if !c.Enabled {
		return nil
	}
	if c.Backend == "" {
		return fmt.Errorf("RAG backend is required when enabled")
	}
	if err := validateRAGBackendConfig(c); err != nil {
		return err
	}
	if err := validateRAGSimilarityThreshold(c.SimilarityThreshold); err != nil {
		return err
	}
	if err := validateRAGTopK(c.TopK); err != nil {
		return err
	}
	if err := validateRAGInjectionMode(c.InjectionMode); err != nil {
		return err
	}
	return validateRAGOnFailure(c.OnFailure)
}

func validateRAGBackendConfig(c *RAGPluginConfig) error {
	switch c.Backend {
	case "milvus":
		return validateMilvusRAGBackend(c)
	case "external_api":
		return validateExternalAPIRAGBackend(c)
	case "mcp":
		return validateMCPRAGBackend(c)
	case "openai":
		return validateOpenAIRAGBackend(c)
	case "hybrid":
		return validateHybridRAGBackend(c)
	default:
		return fmt.Errorf("unknown RAG backend: %s", c.Backend)
	}
}

func validateMilvusRAGBackend(c *RAGPluginConfig) error {
	milvusConfig, err := c.MilvusBackendConfig()
	if err != nil {
		return err
	}
	if milvusConfig.Collection == "" {
		return fmt.Errorf("milvus collection name is required")
	}
	return nil
}

func validateExternalAPIRAGBackend(c *RAGPluginConfig) error {
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
	return nil
}

func validateMCPRAGBackend(c *RAGPluginConfig) error {
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
	return nil
}

func validateOpenAIRAGBackend(c *RAGPluginConfig) error {
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
	return nil
}

func validateHybridRAGBackend(c *RAGPluginConfig) error {
	hybridConfig, err := c.HybridBackendConfig()
	if err != nil {
		return err
	}
	if hybridConfig.Primary == "" {
		return fmt.Errorf("primary backend is required for hybrid RAG")
	}
	return nil
}

func validateRAGSimilarityThreshold(threshold *float32) error {
	if threshold == nil {
		return nil
	}
	if *threshold < 0.0 || *threshold > 1.0 {
		return fmt.Errorf("similarity threshold must be between 0.0 and 1.0, got %.2f", *threshold)
	}
	return nil
}

func validateRAGTopK(topK *int) error {
	if topK == nil || *topK > 0 {
		return nil
	}
	return fmt.Errorf("TopK must be greater than 0, got %d", *topK)
}

func validateRAGInjectionMode(mode string) error {
	if mode == "" || mode == "tool_role" || mode == "system_prompt" {
		return nil
	}
	return fmt.Errorf("injection mode must be 'tool_role' or 'system_prompt', got %s", mode)
}

func validateRAGOnFailure(onFailure string) error {
	if onFailure == "" || onFailure == "skip" || onFailure == "block" || onFailure == "warn" {
		return nil
	}
	return fmt.Errorf("OnFailure must be 'skip', 'block', or 'warn', got %s", onFailure)
}
