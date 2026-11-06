package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Compiler) compilePluginRef(ref *PluginRef) *config.DecisionPlugin {
	// Check if it's a template reference
	if tmpl, ok := c.pluginTemplates[ref.Name]; ok {
		// Merge template fields with override fields
		mergedFields := make(map[string]Value)
		for k, v := range tmpl.Fields {
			mergedFields[k] = v
		}
		if ref.Fields != nil {
			for k, v := range ref.Fields {
				mergedFields[k] = v
			}
		}
		return c.buildDecisionPlugin(tmpl.PluginType, mergedFields)
	}

	// Inline plugin — ref.Name is the plugin type
	fields := ref.Fields
	if fields == nil {
		fields = make(map[string]Value)
	}
	return c.buildDecisionPlugin(ref.Name, fields)
}

func (c *Compiler) compileRAGPlugin(fields map[string]Value) config.RAGPluginConfig {
	cfg := config.RAGPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "backend"); ok {
		cfg.Backend = v
	}
	if v, ok := getIntField(fields, "top_k"); ok {
		cfg.TopK = &v
	}
	if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
		cfg.SimilarityThreshold = &v
	}
	if v, ok := getIntField(fields, "max_context_length"); ok {
		cfg.MaxContextLength = &v
	}
	if v, ok := getStringField(fields, "injection_mode"); ok {
		cfg.InjectionMode = v
	}
	if v, ok := getStringField(fields, "on_failure"); ok {
		cfg.OnFailure = v
	}
	if v, ok := getBoolField(fields, "cache_results"); ok {
		cfg.CacheResults = v
	}
	if v, ok := getIntField(fields, "cache_ttl_seconds"); ok {
		cfg.CacheTTLSeconds = &v
	}
	if v, ok := getFloat32Field(fields, "min_confidence_threshold"); ok {
		cfg.MinConfidenceThreshold = &v
	}
	if raw, exists := fields["rerank"]; exists {
		if obj, ok := raw.(ObjectValue); ok {
			cfg.Rerank = &config.RAGRerankConfig{}
			compilePluginFields(c, obj.Fields, cfg.Rerank)
		} else {
			c.addError(Position{}, "rag rerank must be an object")
		}
	}
	cfg.BackendConfig = c.compileRAGBackendConfig(fields)
	return cfg
}

func (c *Compiler) compileRAGBackendConfig(fields map[string]Value) *config.StructuredPayload {
	obj, ok := fields["backend_config"].(ObjectValue)
	if !ok {
		return nil
	}
	payload, err := config.NewStructuredPayload(fieldsToMap(obj.Fields))
	if err != nil {
		c.addError(Position{}, "failed to encode rag backend_config: %v", err)
		return nil
	}
	return payload
}
