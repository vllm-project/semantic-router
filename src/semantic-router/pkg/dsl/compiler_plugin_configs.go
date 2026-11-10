package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Compiler) buildDecisionPlugin(pluginType string, fields map[string]Value) *config.DecisionPlugin {
	pluginType = config.NormalizeDecisionPluginType(pluginType)
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
	"response_cache": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return compileStrictPluginFields(c, "response_cache", fields, &config.ResponseCachePluginConfig{})
	},
	"context": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return compileStrictPluginFields(c, "context", fields, &config.ContextPluginConfig{})
	},
	"prompt_cache": compilePromptCachePluginConfig,
	// masking decodes its fields as written: hand-mapping them would only add
	// a place to forget one (#3566).
	"masking": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return compileStrictPluginFields(c, "masking", fields, &config.MaskingPluginConfig{})
	},
	"response_guard": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return compileStrictPluginFields(c, "response_guard", fields, &config.ResponseGuardPluginConfig{})
	},
	"header_mutation": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileHeaderMutationPluginConfig(fields), true
	},
	"router_replay": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileRouterReplayPluginConfig(fields)
	},
	"fast_response": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileFastResponsePluginConfig(fields), true
	},
	"request_params": func(c *Compiler, fields map[string]Value) (interface{}, bool) {
		return c.compileRequestParamsPluginConfig(fields), true
	},
	"tools": compileToolsPluginConfig,
}

// compileToolsPluginConfig keeps the tools plugin's fields as written, its
// policy, select and match blocks included, once they decode as one.
func compileToolsPluginConfig(c *Compiler, fields map[string]Value) (interface{}, bool) {
	raw := fieldsToMap(fields)
	payload, err := config.NewStructuredPayload(raw)
	if err == nil {
		err = payload.DecodeIntoStrict(&config.ToolsPluginConfig{})
	}
	if err != nil {
		c.addError(Position{}, "invalid tools configuration: %v; the tools plugin takes enabled, policy and select", err)
		return nil, false
	}
	return raw, true
}

func compilePromptCachePluginConfig(
	c *Compiler,
	fields map[string]Value,
) (interface{}, bool) {
	cfg, err := decodePromptCachePluginFields(fields)
	if err != nil {
		c.addError(Position{}, "%v", err)
		return nil, false
	}
	return cfg, true
}

func decodePromptCachePluginFields(
	fields map[string]Value,
) (*config.PromptCachePluginConfig, error) {
	payload, err := config.NewStructuredPayload(fieldsToMap(fields))
	if err != nil {
		return nil, fmt.Errorf("failed to encode plugin fields: %w", err)
	}
	cfg := &config.PromptCachePluginConfig{}
	if err := payload.DecodeIntoStrict(cfg); err != nil {
		return nil, fmt.Errorf("failed to decode plugin fields: %w", err)
	}
	if err := config.ValidatePromptCachePluginConfig(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Compiler) compileHeaderMutationPluginConfig(fields map[string]Value) config.HeaderMutationPluginConfig {
	return config.HeaderMutationPluginConfig{
		Add:    compileHeaderPairs(fields["add"]),
		Update: compileHeaderPairs(fields["update"]),
		Delete: stringArrayValue(fields["delete"]),
	}
}

func compileHeaderPairs(value Value) []config.HeaderPair {
	array, ok := value.(ArrayValue)
	if !ok {
		return nil
	}
	result := make([]config.HeaderPair, 0, len(array.Items))
	for _, item := range array.Items {
		object, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		name, nameOK := getStringField(object.Fields, "name")
		value, valueOK := getStringField(object.Fields, "value")
		if nameOK && valueOK {
			result = append(result, config.HeaderPair{Name: name, Value: value})
		}
	}
	return result
}

func stringArrayValue(value Value) []string {
	array, ok := value.(ArrayValue)
	if !ok {
		return nil
	}
	result := make([]string, 0, len(array.Items))
	for _, item := range array.Items {
		if text, ok := item.(StringValue); ok {
			result = append(result, text.V)
		}
	}
	return result
}

// compileStrictPluginFields decodes a plugin's fields as the Router loads its
// configuration, refusing a field the payload does not declare.
func compileStrictPluginFields(c *Compiler, pluginType string, fields map[string]Value, target interface{}) (interface{}, bool) {
	payload, err := config.NewStructuredPayload(fieldsToMap(fields))
	if err == nil {
		err = payload.DecodeIntoStrict(target)
	}
	if err != nil {
		c.addError(Position{}, "invalid %s configuration: %v", pluginType, err)
		return nil, false
	}
	return target, true
}

func (c *Compiler) buildPluginConfigValue(pluginType string, fields map[string]Value) (interface{}, bool) {
	if fn, ok := pluginConfigCompilers[pluginType]; ok {
		return fn(c, fields)
	}
	if replacement, retired := config.RemovedPluginTypes[pluginType]; retired {
		c.addError(Position{}, "plugin %q was removed; write %s", pluginType, replacement)
		return nil, false
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

func (c *Compiler) compileRouterReplayPluginConfig(fields map[string]Value) (interface{}, bool) {
	// Preserve omission and explicit false/zero overrides. Serializing the
	// effective capture config here would replace inherited global defaults.
	raw := fieldsToMap(fields)
	payload, err := config.NewStructuredPayload(raw)
	if err == nil {
		err = payload.DecodeIntoStrict(&config.RouterReplayPluginConfig{})
	}
	if err != nil {
		c.addError(Position{}, "invalid router_replay configuration: %v", err)
		return nil, false
	}
	return raw, true
}

func (c *Compiler) compileFastResponsePluginConfig(fields map[string]Value) config.FastResponsePluginConfig {
	cfg := config.FastResponsePluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = &v
	}
	if v, ok := getStringField(fields, "message"); ok {
		cfg.Message = v
	}
	return cfg
}

func (c *Compiler) compileRequestParamsPluginConfig(fields map[string]Value) config.RequestParamsPluginConfig {
	cfg := config.RequestParamsPluginConfig{}
	if value, exists := fields["default_max_tokens"]; exists {
		if integer, ok := value.(IntValue); ok && integer.V > 0 {
			cfg.DefaultMaxTokens = config.FixedOutputTokenDefault(integer.V)
		} else if text, ok := value.(StringValue); ok && text.V == "auto" {
			cfg.DefaultMaxTokens = &config.OutputTokenDefault{Auto: true}
		} else {
			c.addError(Position{}, "request_params.default_max_tokens must be a positive integer or auto")
		}
	}
	if v, ok := getStringArrayField(fields, "blocked_params"); ok {
		cfg.BlockedParams = v
	}
	if v, ok := getIntField(fields, "max_tokens_limit"); ok {
		cfg.MaxTokensLimit = &v
	}
	if v, ok := getIntField(fields, "max_n"); ok {
		cfg.MaxN = &v
	}
	return cfg
}
