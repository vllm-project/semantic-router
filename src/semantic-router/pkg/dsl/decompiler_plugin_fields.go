package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type pluginFieldsDecoder func(*config.DecisionPlugin) map[string]Value

var pluginFieldsDecoders = map[string]pluginFieldsDecoder{
	"system_prompt":       pluginFieldsSystemPrompt,
	"response_cache":      pluginFieldsResponseCache,
	"context_compression": pluginFieldsStructuredConfiguration,
	"prompt_cache":        pluginFieldsStructuredConfiguration,
	"router_replay":       pluginFieldsStructuredConfiguration,
	"shadow_dispatch":     pluginFieldsStructuredConfiguration,
	"memory":              pluginFieldsMemory,
	"response_guard":      pluginFieldsStructuredConfiguration,
	"fast_response":       pluginFieldsFastResponse,
	"request_params":      pluginFieldsRequestParams,
	"tools":               pluginFieldsStructuredConfiguration,
	"rag":                 pluginFieldsRAG,
	"header_mutation":     pluginFieldsHeaderMutation,
	"masking":             pluginFieldsStructuredConfiguration,
}

func pluginConfigToFields(p *config.DecisionPlugin) map[string]Value {
	if fn, ok := pluginFieldsDecoders[config.NormalizeDecisionPluginType(p.Type)]; ok {
		return fn(p)
	}
	return map[string]Value{}
}

func pluginFieldsSystemPrompt(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.SystemPromptPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled != nil {
		fields["enabled"] = BoolValue{V: *cfg.Enabled}
	}
	if cfg.SystemPrompt != "" {
		fields["system_prompt"] = StringValue{V: cfg.SystemPrompt}
	}
	if cfg.Mode != "" {
		fields["mode"] = StringValue{V: cfg.Mode}
	}
	return fields
}

func pluginFieldsResponseCache(p *config.DecisionPlugin) map[string]Value {
	return pluginFieldsStructuredConfiguration(p)
}

func pluginFieldsStructuredConfiguration(p *config.DecisionPlugin) map[string]Value {
	object, ok := structuredPayloadObjectValue(p.Configuration)
	if !ok {
		return map[string]Value{}
	}
	return object.Fields
}

func pluginFieldsMemory(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.MemoryPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.RetrievalLimit != nil {
		fields["retrieval_limit"] = IntValue{V: *cfg.RetrievalLimit}
	}
	if cfg.SimilarityThreshold != nil {
		fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
	}
	if cfg.AutoStore != nil {
		fields["auto_store"] = BoolValue{V: *cfg.AutoStore}
	}
	return fields
}

func pluginFieldsFastResponse(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.FastResponsePluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Message != "" {
		fields["message"] = StringValue{V: cfg.Message}
	}
	return fields
}

func pluginFieldsRequestParams(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.RequestParamsPluginConfig](p)
	if !ok {
		return fields
	}
	if len(cfg.BlockedParams) > 0 {
		var items []Value
		for _, s := range cfg.BlockedParams {
			items = append(items, StringValue{V: s})
		}
		fields["blocked_params"] = ArrayValue{Items: items}
	}
	if cfg.DefaultMaxTokens != nil {
		if cfg.DefaultMaxTokens.Auto {
			fields["default_max_tokens"] = StringValue{V: "auto"}
		} else {
			fields["default_max_tokens"] = IntValue{V: cfg.DefaultMaxTokens.Value}
		}
	}
	if cfg.MaxTokensLimit != nil {
		fields["max_tokens_limit"] = IntValue{V: *cfg.MaxTokensLimit}
	}
	if cfg.MaxN != nil {
		fields["max_n"] = IntValue{V: *cfg.MaxN}
	}
	if cfg.StripUnknown {
		fields["strip_unknown"] = BoolValue{V: true}
	}
	return fields
}

func pluginFieldsRAG(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.RAGPluginConfig](p)
	if !ok {
		return fields
	}
	addRAGCoreFields(fields, cfg)
	addRAGBackendAndFailureFields(fields, cfg)
	addRAGCacheFields(fields, cfg)
	return fields
}

func addRAGCoreFields(fields map[string]Value, cfg *config.RAGPluginConfig) {
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.Backend != "" {
		fields["backend"] = StringValue{V: cfg.Backend}
	}
	if cfg.SimilarityThreshold != nil {
		fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
	}
	if cfg.TopK != nil {
		fields["top_k"] = IntValue{V: *cfg.TopK}
	}
	if cfg.MaxContextLength != nil {
		fields["max_context_length"] = IntValue{V: *cfg.MaxContextLength}
	}
	if cfg.InjectionMode != "" {
		fields["injection_mode"] = StringValue{V: cfg.InjectionMode}
	}
}

func addRAGBackendAndFailureFields(fields map[string]Value, cfg *config.RAGPluginConfig) {
	if cfg.Rerank != nil {
		rerank := make(map[string]Value)
		if cfg.Rerank.Deployment != "" {
			rerank["deployment"] = StringValue{V: cfg.Rerank.Deployment}
		}
		if cfg.Rerank.TopK != nil {
			rerank["top_k"] = IntValue{V: *cfg.Rerank.TopK}
		}
		if cfg.Rerank.Layer != 0 {
			rerank["layer"] = IntValue{V: cfg.Rerank.Layer}
		}
		if cfg.Rerank.Dimension != 0 {
			rerank["dimension"] = IntValue{V: cfg.Rerank.Dimension}
		}
		fields["rerank"] = ObjectValue{Fields: rerank}
	}
	if backendConfig, ok := structuredPayloadObjectValue(cfg.BackendConfig); ok {
		fields["backend_config"] = backendConfig
	}
	if cfg.OnFailure != "" {
		fields["on_failure"] = StringValue{V: cfg.OnFailure}
	}
}

func addRAGCacheFields(fields map[string]Value, cfg *config.RAGPluginConfig) {
	if cfg.CacheResults {
		fields["cache_results"] = BoolValue{V: true}
	}
	if cfg.CacheTTLSeconds != nil {
		fields["cache_ttl_seconds"] = IntValue{V: *cfg.CacheTTLSeconds}
	}
	if cfg.MinConfidenceThreshold != nil {
		fields["min_confidence_threshold"] = FloatValue{V: float64(*cfg.MinConfidenceThreshold)}
	}
}

func pluginFieldsHeaderMutation(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.HeaderMutationPluginConfig](p)
	if !ok {
		return fields
	}
	if len(cfg.Add) > 0 {
		var items []Value
		for _, h := range cfg.Add {
			items = append(items, ObjectValue{Fields: map[string]Value{
				"name":  StringValue{V: h.Name},
				"value": StringValue{V: h.Value},
			}})
		}
		fields["add"] = ArrayValue{Items: items}
	}
	if len(cfg.Update) > 0 {
		var items []Value
		for _, h := range cfg.Update {
			items = append(items, ObjectValue{Fields: map[string]Value{
				"name":  StringValue{V: h.Name},
				"value": StringValue{V: h.Value},
			}})
		}
		fields["update"] = ArrayValue{Items: items}
	}
	if len(cfg.Delete) > 0 {
		fields["delete"] = stringsToArray(cfg.Delete)
	}
	return fields
}

func structuredPayloadObjectValue(payload *config.StructuredPayload) (ObjectValue, bool) {
	raw, err := payload.AsStringMap()
	if err != nil || len(raw) == 0 {
		return ObjectValue{}, false
	}
	return interfaceMapObjectValue(raw), true
}

func interfaceMapObjectValue(raw map[string]interface{}) ObjectValue {
	fields := make(map[string]Value, len(raw))
	for key, value := range raw {
		fields[key] = interfaceValueToDSLValue(value)
	}
	return ObjectValue{Fields: fields}
}

func interfaceValueToDSLValue(raw interface{}) Value {
	switch typed := normalizePluginConfigValue(raw).(type) {
	case string:
		return StringValue{V: typed}
	case bool:
		return BoolValue{V: typed}
	case int:
		return IntValue{V: typed}
	case int64:
		return IntValue{V: int(typed)}
	case float32:
		return FloatValue{V: float64(typed)}
	case float64:
		return FloatValue{V: typed}
	case []interface{}:
		items := make([]Value, 0, len(typed))
		for _, item := range typed {
			items = append(items, interfaceValueToDSLValue(item))
		}
		return ArrayValue{Items: items}
	case map[string]interface{}:
		return interfaceMapObjectValue(typed)
	default:
		return StringValue{V: fmt.Sprintf("%v", typed)}
	}
}
