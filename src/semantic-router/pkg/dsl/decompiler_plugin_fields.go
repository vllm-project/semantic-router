package dsl

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type pluginFieldsDecoder func(*config.DecisionPlugin) map[string]Value

var pluginFieldsDecoders = map[string]pluginFieldsDecoder{
	"system_prompt":   pluginFieldsSystemPrompt,
	"semantic-cache":  pluginFieldsSemanticCache,
	"router_replay":   pluginFieldsRouterReplay,
	"memory":          pluginFieldsMemory,
	"hallucination":   pluginFieldsHallucination,
	"image_gen":       pluginFieldsImageGen,
	"fast_response":   pluginFieldsFastResponse,
	"request_params":  pluginFieldsRequestParams,
	"tool_selection":  pluginFieldsToolSelection,
	"tools":           pluginFieldsTools,
	"rag":             pluginFieldsRAG,
	"header_mutation": pluginFieldsHeaderMutation,
}

func pluginConfigToFields(p *config.DecisionPlugin) map[string]Value {
	if fn, ok := pluginFieldsDecoders[p.Type]; ok {
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

func pluginFieldsSemanticCache(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.SemanticCachePluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.SimilarityThreshold != nil {
		fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
	}
	return fields
}

func pluginFieldsRouterReplay(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.RouterReplayPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.MaxRecords != 0 {
		fields["max_records"] = IntValue{V: cfg.MaxRecords}
	}
	if cfg.CaptureRequestBody {
		fields["capture_request_body"] = BoolValue{V: true}
	}
	if cfg.CaptureResponseBody {
		fields["capture_response_body"] = BoolValue{V: true}
	}
	if cfg.MaxBodyBytes != 0 {
		fields["max_body_bytes"] = IntValue{V: cfg.MaxBodyBytes}
	}
	return fields
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

func pluginFieldsHallucination(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.HallucinationPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	return fields
}

func pluginFieldsImageGen(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.ImageGenPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.Backend != "" {
		fields["backend"] = StringValue{V: cfg.Backend}
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

func pluginFieldsToolSelection(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.ToolSelectionPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.Mode != "" {
		fields["mode"] = StringValue{V: cfg.Mode}
	}
	if cfg.ToolsDBPath != "" {
		fields["tools_db_path"] = StringValue{V: cfg.ToolsDBPath}
	}
	if cfg.TopK != 0 {
		fields["top_k"] = IntValue{V: cfg.TopK}
	}
	if cfg.SimilarityThreshold != nil {
		fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
	}
	if cfg.Strategy != "" {
		fields["strategy"] = StringValue{V: cfg.Strategy}
	}
	if cfg.RelevanceThreshold != nil {
		fields["relevance_threshold"] = FloatValue{V: float64(*cfg.RelevanceThreshold)}
	}
	if cfg.PreserveCount != 0 {
		fields["preserve_count"] = IntValue{V: cfg.PreserveCount}
	}
	return fields
}

func pluginFieldsTools(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.ToolsPluginConfig](p)
	if !ok {
		return fields
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.Mode != "" {
		fields["mode"] = StringValue{V: cfg.Mode}
	}
	if cfg.SemanticSelection != nil {
		fields["semantic_selection"] = BoolValue{V: *cfg.SemanticSelection}
	}
	if cfg.Strategy != "" {
		fields["strategy"] = StringValue{V: cfg.Strategy}
	}
	if len(cfg.AllowTools) > 0 {
		fields["allow_tools"] = stringsToArray(cfg.AllowTools)
	}
	if len(cfg.BlockTools) > 0 {
		fields["block_tools"] = stringsToArray(cfg.BlockTools)
	}
	if cfg.DynamicRetrieval != nil {
		fields["dynamic_retrieval"] = dynamicRetrievalObjectValue(cfg.DynamicRetrieval)
	}
	return fields
}

func pluginFieldsRAG(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	cfg, ok := decodePluginConfig[config.RAGPluginConfig](p)
	if !ok {
		return fields
	}
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
	return fields
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
