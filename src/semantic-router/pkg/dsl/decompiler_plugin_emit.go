package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type typedPluginConfigEmitter func(*strings.Builder, *config.DecisionPlugin)

var typedPluginConfigEmitters = map[string]typedPluginConfigEmitter{
	"system_prompt":   emitSystemPromptPluginConfig,
	"semantic-cache":  emitSemanticCachePluginConfig,
	"router_replay":   emitRouterReplayPluginConfig,
	"memory":          emitMemoryPluginConfig,
	"hallucination":   emitHallucinationPluginConfig,
	"image_gen":       emitImageGenPluginConfig,
	"fast_response":   emitFastResponsePluginConfig,
	"request_params":  emitRequestParamsPluginConfig,
	"tool_selection":  emitToolSelectionPluginConfig,
	"tools":           emitToolsPluginConfig,
	"rag":             emitRAGPluginConfig,
	"header_mutation": emitHeaderMutationPluginConfig,
}

func decompilePluginConfig(p *config.DecisionPlugin) string {
	var sb strings.Builder
	emitTypedPluginConfig(&sb, p)
	if rawMap, ok := normalizePluginConfigMap(p.Configuration); ok {
		extraFields := filterPluginConfigMap(rawMap, knownPluginConfigKeys(p))
		if len(extraFields) > 0 {
			writePluginConfigMap(&sb, extraFields, "    ")
		}
	}
	return sb.String()
}

func emitTypedPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	if fn, ok := typedPluginConfigEmitters[p.Type]; ok {
		fn(sb, p)
	}
}

func emitSystemPromptPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.SystemPromptPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled != nil {
		fmt.Fprintf(sb, "    enabled: %v\n", *cfg.Enabled)
	}
	if cfg.SystemPrompt != "" {
		fmt.Fprintf(sb, "    system_prompt: %q\n", cfg.SystemPrompt)
	}
	if cfg.Mode != "" {
		fmt.Fprintf(sb, "    mode: %q\n", cfg.Mode)
	}
}

func emitSemanticCachePluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.SemanticCachePluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.SimilarityThreshold != nil {
		fmt.Fprintf(sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
	}
}

func emitRouterReplayPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.RouterReplayPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.MaxRecords != 0 {
		fmt.Fprintf(sb, "    max_records: %d\n", cfg.MaxRecords)
	}
	if cfg.CaptureRequestBody {
		fmt.Fprintf(sb, "    capture_request_body: true\n")
	}
	if cfg.CaptureResponseBody {
		fmt.Fprintf(sb, "    capture_response_body: true\n")
	}
	if cfg.MaxBodyBytes != 0 {
		fmt.Fprintf(sb, "    max_body_bytes: %d\n", cfg.MaxBodyBytes)
	}
}

func emitMemoryPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.MemoryPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.RetrievalLimit != nil {
		fmt.Fprintf(sb, "    retrieval_limit: %d\n", *cfg.RetrievalLimit)
	}
	if cfg.SimilarityThreshold != nil {
		fmt.Fprintf(sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
	}
	if cfg.AutoStore != nil {
		fmt.Fprintf(sb, "    auto_store: %v\n", *cfg.AutoStore)
	}
}

func emitHallucinationPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.HallucinationPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.UseNLI {
		fmt.Fprintf(sb, "    use_nli: true\n")
	}
	if cfg.HallucinationAction != "" {
		fmt.Fprintf(sb, "    hallucination_action: %q\n", cfg.HallucinationAction)
	}
}

func emitImageGenPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.ImageGenPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.Backend != "" {
		fmt.Fprintf(sb, "    backend: %q\n", cfg.Backend)
	}
}

func emitFastResponsePluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.FastResponsePluginConfig](p)
	if !ok {
		return
	}
	if cfg.Message != "" {
		fmt.Fprintf(sb, "    message: %q\n", cfg.Message)
	}
}

func emitRequestParamsPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.RequestParamsPluginConfig](p)
	if !ok {
		return
	}
	if len(cfg.BlockedParams) > 0 {
		fmt.Fprintf(sb, "    blocked_params: %s\n", formatStringArray(cfg.BlockedParams))
	}
	if cfg.MaxTokensLimit != nil {
		fmt.Fprintf(sb, "    max_tokens_limit: %d\n", *cfg.MaxTokensLimit)
	}
	if cfg.MaxN != nil {
		fmt.Fprintf(sb, "    max_n: %d\n", *cfg.MaxN)
	}
	if cfg.StripUnknown {
		fmt.Fprintf(sb, "    strip_unknown: true\n")
	}
}

func emitToolSelectionPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.ToolSelectionPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.Mode != "" {
		fmt.Fprintf(sb, "    mode: %q\n", cfg.Mode)
	}
	if cfg.ToolsDBPath != "" {
		fmt.Fprintf(sb, "    tools_db_path: %q\n", cfg.ToolsDBPath)
	}
	if cfg.TopK != 0 {
		fmt.Fprintf(sb, "    top_k: %d\n", cfg.TopK)
	}
	if cfg.SimilarityThreshold != nil {
		fmt.Fprintf(sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
	}
	if cfg.Strategy != "" {
		fmt.Fprintf(sb, "    strategy: %q\n", cfg.Strategy)
	}
	if cfg.FallbackToEmpty != nil && *cfg.FallbackToEmpty {
		fmt.Fprintf(sb, "    fallback_to_empty: true\n")
	}
	if cfg.RelevanceThreshold != nil {
		fmt.Fprintf(sb, "    relevance_threshold: %v\n", *cfg.RelevanceThreshold)
	}
	if cfg.PreserveCount != 0 {
		fmt.Fprintf(sb, "    preserve_count: %d\n", cfg.PreserveCount)
	}
}

func emitToolsPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.ToolsPluginConfig](p)
	if !ok {
		return
	}
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.Mode != "" {
		fmt.Fprintf(sb, "    mode: %q\n", cfg.Mode)
	}
	if cfg.SemanticSelection != nil {
		fmt.Fprintf(sb, "    semantic_selection: %v\n", *cfg.SemanticSelection)
	}
	if cfg.Strategy != "" {
		fmt.Fprintf(sb, "    strategy: %q\n", cfg.Strategy)
	}
	if len(cfg.AllowTools) > 0 {
		fmt.Fprintf(sb, "    allow_tools: %s\n", formatStringArray(cfg.AllowTools))
	}
	if len(cfg.BlockTools) > 0 {
		fmt.Fprintf(sb, "    block_tools: %s\n", formatStringArray(cfg.BlockTools))
	}
	if cfg.DynamicRetrieval != nil {
		fmt.Fprintf(sb, "    dynamic_retrieval: %s\n", formatPluginConfigValue(dynamicRetrievalConfigMap(cfg.DynamicRetrieval)))
	}
}

func emitRAGPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.RAGPluginConfig](p)
	if !ok {
		return
	}
	emitRAGCorePluginConfig(sb, cfg)
	emitRAGBackendAndFailureConfig(sb, cfg)
	emitRAGCacheConfig(sb, cfg)
}

func emitRAGCorePluginConfig(sb *strings.Builder, cfg *config.RAGPluginConfig) {
	if cfg.Enabled {
		fmt.Fprintf(sb, "    enabled: true\n")
	}
	if cfg.Backend != "" {
		fmt.Fprintf(sb, "    backend: %q\n", cfg.Backend)
	}
	if cfg.SimilarityThreshold != nil {
		fmt.Fprintf(sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
	}
	if cfg.TopK != nil {
		fmt.Fprintf(sb, "    top_k: %d\n", *cfg.TopK)
	}
	if cfg.MaxContextLength != nil {
		fmt.Fprintf(sb, "    max_context_length: %d\n", *cfg.MaxContextLength)
	}
	if cfg.InjectionMode != "" {
		fmt.Fprintf(sb, "    injection_mode: %q\n", cfg.InjectionMode)
	}
}

func emitRAGBackendAndFailureConfig(sb *strings.Builder, cfg *config.RAGPluginConfig) {
	if backendConfig, ok := normalizePluginConfigMap(cfg.BackendConfig); ok && len(backendConfig) > 0 {
		fmt.Fprintf(sb, "    backend_config: %s\n", formatPluginConfigValue(backendConfig))
	}
	if cfg.OnFailure != "" {
		fmt.Fprintf(sb, "    on_failure: %q\n", cfg.OnFailure)
	}
}

func emitRAGCacheConfig(sb *strings.Builder, cfg *config.RAGPluginConfig) {
	if cfg.CacheResults {
		fmt.Fprintf(sb, "    cache_results: true\n")
	}
	if cfg.CacheTTLSeconds != nil {
		fmt.Fprintf(sb, "    cache_ttl_seconds: %d\n", *cfg.CacheTTLSeconds)
	}
	if cfg.MinConfidenceThreshold != nil {
		fmt.Fprintf(sb, "    min_confidence_threshold: %v\n", *cfg.MinConfidenceThreshold)
	}
}

func emitHeaderMutationPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	cfg, ok := decodePluginConfig[config.HeaderMutationPluginConfig](p)
	if !ok {
		return
	}
	if len(cfg.Add) > 0 {
		fmt.Fprintf(sb, "    add: [")
		for i, h := range cfg.Add {
			if i > 0 {
				fmt.Fprintf(sb, ", ")
			}
			fmt.Fprintf(sb, "{ name: %q, value: %q }", h.Name, h.Value)
		}
		fmt.Fprintf(sb, "]\n")
	}
	if len(cfg.Update) > 0 {
		fmt.Fprintf(sb, "    update: [")
		for i, h := range cfg.Update {
			if i > 0 {
				fmt.Fprintf(sb, ", ")
			}
			fmt.Fprintf(sb, "{ name: %q, value: %q }", h.Name, h.Value)
		}
		fmt.Fprintf(sb, "]\n")
	}
	if len(cfg.Delete) > 0 {
		fmt.Fprintf(sb, "    delete: %s\n", formatStringArray(cfg.Delete))
	}
}
