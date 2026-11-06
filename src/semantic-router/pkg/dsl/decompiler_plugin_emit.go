package dsl

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type typedPluginConfigEmitter func(*strings.Builder, *config.DecisionPlugin)

var typedPluginConfigEmitters = map[string]typedPluginConfigEmitter{
	"system_prompt":       emitSystemPromptPluginConfig,
	"response_cache":      emitResponseCachePluginConfig,
	"context_compression": emitStructuredPluginConfig,
	"prompt_cache":        emitStructuredPluginConfig,
	"router_replay":       emitRouterReplayPluginConfig,
	"shadow_dispatch":     emitStructuredPluginConfig,
	"memory":              emitMemoryPluginConfig,
	"response_guard":      emitStructuredPluginConfig,
	"fast_response":       emitFastResponsePluginConfig,
	"request_params":      emitRequestParamsPluginConfig,
	"tools":               emitStructuredPluginConfig,
	"rag":                 emitRAGPluginConfig,
	"header_mutation":     emitHeaderMutationPluginConfig,
	"masking":             emitStructuredPluginConfig,
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
	if fn, ok := typedPluginConfigEmitters[config.NormalizeDecisionPluginType(p.Type)]; ok {
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

func emitResponseCachePluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	emitStructuredPluginConfig(sb, p)
}

func emitStructuredPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	raw, ok := normalizePluginConfigMap(p.Configuration)
	if !ok {
		return
	}
	writePluginConfigMap(sb, raw, "    ")
}

func emitRouterReplayPluginConfig(sb *strings.Builder, p *config.DecisionPlugin) {
	raw, ok := normalizePluginConfigMap(p.Configuration)
	if !ok {
		return
	}
	// Keep the canonical field order while retaining raw omission, false and
	// zero; an effective capture struct would erase the inheritance contract.
	typ := reflect.TypeOf(config.RouterReplayPluginConfig{})
	for index := 0; index < typ.NumField(); index++ {
		key := strings.Split(typ.Field(index).Tag.Get("json"), ",")[0]
		if value, exists := raw[key]; exists {
			fmt.Fprintf(sb, "    %s: %s\n", key, formatPluginConfigValue(value))
			delete(raw, key)
		}
	}
	writePluginConfigMap(sb, raw, "    ")
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
	if cfg.DefaultMaxTokens != nil {
		if cfg.DefaultMaxTokens.Auto {
			fmt.Fprintln(sb, "    default_max_tokens: \"auto\"")
		} else {
			fmt.Fprintf(sb, "    default_max_tokens: %d\n", cfg.DefaultMaxTokens.Value)
		}
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
	if cfg.Rerank != nil {
		fmt.Fprintf(sb, "    rerank: %s\n", formatRAGRerank(cfg.Rerank))
	}
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

// formatRAGRerank writes a rag plugin's rerank block: the deployment that
// scores the passages, how many it keeps, and the pair scorer's layer and
// dimension.
func formatRAGRerank(rerank *config.RAGRerankConfig) string {
	var parts []string
	if rerank.Deployment != "" {
		parts = append(parts, fmt.Sprintf("deployment: %q", rerank.Deployment))
	}
	if rerank.TopK != nil {
		parts = append(parts, fmt.Sprintf("top_k: %d", *rerank.TopK))
	}
	if rerank.Layer != 0 {
		parts = append(parts, fmt.Sprintf("layer: %d", rerank.Layer))
	}
	if rerank.Dimension != 0 {
		parts = append(parts, fmt.Sprintf("dimension: %d", rerank.Dimension))
	}
	if len(parts) == 0 {
		return "{}"
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}
