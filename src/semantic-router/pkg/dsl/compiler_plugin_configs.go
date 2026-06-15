package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Compiler) buildDecisionPlugin(pluginType string, fields map[string]Value) *config.DecisionPlugin {
	if pluginType == "semantic_cache" {
		pluginType = "semantic-cache"
	}
	dp := &config.DecisionPlugin{Type: pluginType}
	cfg, ok := c.buildPluginConfigValue(pluginType, fields)
	if !ok {
		return nil
	}
	if err := c.attachPluginConfig(dp, pluginType, cfg); err != nil {
		c.addError(Position{}, "failed to encode plugin %q configuration: %v", pluginType, err)
		return nil
	}
	return dp
}

func (c *Compiler) attachPluginConfig(dp *config.DecisionPlugin, pluginType string, cfg interface{}) error {
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		return err
	}
	dp.Configuration = payload
	return nil
}

type pluginConfigCompiler func(*Compiler, map[string]Value) (interface{}, bool)

var pluginConfigCompilers = map[string]pluginConfigCompiler{
	"system_prompt": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileSystemPromptPluginConfig(fields), true
	},
	"semantic_cache": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileSemanticCachePluginConfig(fields), true
	},
	"semantic-cache": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileSemanticCachePluginConfig(fields), true
	},
	"hallucination": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileHallucinationPluginConfig(fields), true
	},
	"memory": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileMemoryPluginConfig(fields), true
	},
	"rag": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileRAGPlugin(fields), true
	},
	"header_mutation": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return config.HeaderMutationPluginConfig{}, true
	},
	"router_replay": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileRouterReplayPluginConfig(fields), true
	},
	"image_gen": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileImageGenPluginConfig(fields), true
	},
	"fast_response": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileFastResponsePluginConfig(fields), true
	},
	"request_params": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileRequestParamsPluginConfig(fields), true
	},
	"tool_selection": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileToolSelectionPluginConfig(fields), true
	},
	"tools": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileToolsPlugin(fields), true
	},
}

func (c *Compiler) buildPluginConfigValue(pluginType string, fields map[string]Value) (interface{}, bool) {
	if fn, ok := pluginConfigCompilers[pluginType]; ok {
		return fn(c, fields)
	}
	c.addError(Position{}, "unknown plugin type %q", pluginType)
	return nil, false
}

func (c *Compiler) compileSystemPromptPluginConfig(fields map[string]Value) config.SystemPromptPluginConfig {
	cfg := config.SystemPromptPluginConfig{}
	if v, ok := getStringField(fields, "system_prompt"); ok {
		cfg.SystemPrompt = v
	}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = &v
	}
	if v, ok := getStringField(fields, "mode"); ok {
		cfg.Mode = v
	}
	return cfg
}

func (c *Compiler) compileSemanticCachePluginConfig(fields map[string]Value) config.SemanticCachePluginConfig {
	cfg := config.SemanticCachePluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
		cfg.SimilarityThreshold = &v
	}
	return cfg
}

func (c *Compiler) compileHallucinationPluginConfig(fields map[string]Value) config.HallucinationPluginConfig {
	cfg := config.HallucinationPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getBoolField(fields, "use_nli"); ok {
		cfg.UseNLI = v
	}
	if v, ok := getStringField(fields, "hallucination_action"); ok {
		cfg.HallucinationAction = v
	}
	return cfg
}

func (c *Compiler) compileMemoryPluginConfig(fields map[string]Value) config.MemoryPluginConfig {
	cfg := config.MemoryPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getIntField(fields, "retrieval_limit"); ok {
		cfg.RetrievalLimit = &v
	}
	if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
		cfg.SimilarityThreshold = &v
	}
	if v, ok := getBoolField(fields, "auto_store"); ok {
		cfg.AutoStore = &v
	}
	return cfg
}

func (c *Compiler) compileRouterReplayPluginConfig(fields map[string]Value) config.RouterReplayPluginConfig {
	cfg := config.RouterReplayPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getIntField(fields, "max_records"); ok {
		cfg.MaxRecords = v
	}
	if v, ok := getBoolField(fields, "capture_request_body"); ok {
		cfg.CaptureRequestBody = v
	}
	if v, ok := getBoolField(fields, "capture_response_body"); ok {
		cfg.CaptureResponseBody = v
	}
	if v, ok := getIntField(fields, "max_body_bytes"); ok {
		cfg.MaxBodyBytes = v
	}
	return cfg
}

func (c *Compiler) compileImageGenPluginConfig(fields map[string]Value) config.ImageGenPluginConfig {
	cfg := config.ImageGenPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "backend"); ok {
		cfg.Backend = v
	}
	return cfg
}

func (c *Compiler) compileFastResponsePluginConfig(fields map[string]Value) config.FastResponsePluginConfig {
	cfg := config.FastResponsePluginConfig{}
	if v, ok := getStringField(fields, "message"); ok {
		cfg.Message = v
	}
	return cfg
}

func (c *Compiler) compileRequestParamsPluginConfig(fields map[string]Value) config.RequestParamsPluginConfig {
	cfg := config.RequestParamsPluginConfig{}
	if v, ok := getStringArrayField(fields, "blocked_params"); ok {
		cfg.BlockedParams = v
	}
	if v, ok := getIntField(fields, "max_tokens_limit"); ok {
		cfg.MaxTokensLimit = &v
	}
	if v, ok := getIntField(fields, "max_n"); ok {
		cfg.MaxN = &v
	}
	if v, ok := getBoolField(fields, "strip_unknown"); ok {
		cfg.StripUnknown = v
	}
	return cfg
}

func (c *Compiler) compileToolSelectionPluginConfig(fields map[string]Value) config.ToolSelectionPluginConfig {
	cfg := config.ToolSelectionPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "mode"); ok {
		cfg.Mode = v
	}
	if v, ok := getIntField(fields, "top_k"); ok {
		cfg.TopK = v
	}
	if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
		cfg.SimilarityThreshold = &v
	}
	if v, ok := getStringField(fields, "tools_db_path"); ok {
		cfg.ToolsDBPath = v
	}
	if v, ok := getFloat32Field(fields, "relevance_threshold"); ok {
		cfg.RelevanceThreshold = &v
	}
	if v, ok := getIntField(fields, "preserve_count"); ok {
		cfg.PreserveCount = v
	}
	if v, ok := getStringField(fields, "strategy"); ok {
		cfg.Strategy = v
	}
	return cfg
}
