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

func (c *Compiler) compileToolsPlugin(fields map[string]Value) config.ToolsPluginConfig {
	cfg := config.ToolsPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "mode"); ok {
		cfg.Mode = v
	}
	if v, ok := getBoolField(fields, "semantic_selection"); ok {
		cfg.SemanticSelection = &v
	}
	if values, ok := getStringArrayField(fields, "allow_tools"); ok {
		cfg.AllowTools = values
	}
	if values, ok := getStringArrayField(fields, "block_tools"); ok {
		cfg.BlockTools = values
	}
	if v, ok := getStringField(fields, "strategy"); ok {
		cfg.Strategy = v
	}
	if raw, ok := fields["dynamic_retrieval"]; ok {
		if obj, ok := raw.(ObjectValue); ok {
			cfg.DynamicRetrieval = compileDynamicRetrievalConfig(obj.Fields)
		}
	}
	return cfg
}

func compileDynamicRetrievalConfig(fields map[string]Value) *config.DynamicRetrievalConfig {
	cfg := &config.DynamicRetrievalConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "strategy"); ok {
		cfg.Strategy = v
	}
	if v, ok := getIntField(fields, "history_window"); ok {
		cfg.HistoryWindow = v
	}
	if v, ok := getFloat64Field(fields, "min_history_confidence"); ok {
		cfg.MinHistoryConfidence = v
	}
	if v, ok := getBoolField(fields, "fallback_on_low_confidence"); ok {
		cfg.FallbackOnLowConfidence = v
	}
	if raw, ok := fields["weights"]; ok {
		if obj, ok := raw.(ObjectValue); ok {
			cfg.Weights = compileDynamicRetrievalWeights(obj.Fields)
		}
	}
	return cfg
}

func compileDynamicRetrievalWeights(fields map[string]Value) *config.DynamicRetrievalWeights {
	weights := &config.DynamicRetrievalWeights{}
	if v, ok := getFloat64Field(fields, "semantic"); ok {
		weights.Semantic = v
	}
	if v, ok := getFloat64Field(fields, "history"); ok {
		weights.History = v
	}
	if v, ok := getFloat64Field(fields, "decision_prior"); ok {
		weights.DecisionPrior = v
	}
	if v, ok := getFloat64Field(fields, "repetition_penalty"); ok {
		weights.RepetitionPenalty = v
	}
	return weights
}
