package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func loadClassifierMappings(cfg *config.RouterConfig) (*classifierMappings, error) {
	mappings := &classifierMappings{}
	var err error

	if cfg.NeedsCategoryMappingForRouting() {
		mappings.categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "category_mapping_loaded", map[string]interface{}{
			"count": mappings.categoryMapping.GetCategoryCount(),
		})
	}

	if cfg.NeedsPIIMappingForRouting() {
		mappings.piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "pii_mapping_loaded", map[string]interface{}{
			"count": mappings.piiMapping.GetPIITypeCount(),
		})
	}

	if cfg.NeedsJailbreakMappingForRouting() {
		mappings.jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "jailbreak_mapping_loaded", map[string]interface{}{
			"count": mappings.jailbreakMapping.GetJailbreakTypeCount(),
		})
	}

	return mappings, nil
}

func createSemanticCache(cfg *config.RouterConfig) (cache.CacheBackend, error) {
	semanticCacheCfg := cfg.SemanticCache
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(semanticCacheCfg.BackendType),
		Enabled:             semanticCacheCfg.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          semanticCacheCfg.MaxEntries,
		TTLSeconds:          semanticCacheCfg.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(semanticCacheCfg.EvictionPolicy),
		Redis:               semanticCacheCfg.Redis,
		Valkey:              semanticCacheCfg.Valkey,
		Milvus:              semanticCacheCfg.Milvus,
		Qdrant:              semanticCacheCfg.Qdrant,
		EmbeddingModel:      detectSemanticCacheEmbeddingModel(cfg),
	}

	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.ComponentEvent("extproc", "semantic_cache_initialized", map[string]interface{}{
			"backend":              cacheConfig.BackendType,
			"similarity_threshold": cacheConfig.SimilarityThreshold,
			"ttl_seconds":          cacheConfig.TTLSeconds,
			"max_entries":          cacheConfig.MaxEntries,
		})
	} else {
		logging.ComponentEvent("extproc", "semantic_cache_disabled", map[string]interface{}{
			"backend": cacheConfig.BackendType,
		})
	}

	return semanticCache, nil
}

func detectSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	semanticCacheCfg := cfg.SemanticCache
	embeddingModels := cfg.EmbeddingModels
	embeddingModel := semanticCacheCfg.EmbeddingModel
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case embeddingModels.MmBertModelPath != "":
		return "mmbert"
	case embeddingModels.MultiModalModelPath != "":
		return "multimodal"
	case embeddingModels.Qwen3ModelPath != "":
		return "qwen3"
	case embeddingModels.GemmaModelPath != "":
		return "gemma"
	default:
		logging.ComponentWarnEvent("extproc", "semantic_cache_embedding_fallback", map[string]interface{}{
			"fallback_model": "bert",
		})
		return "bert"
	}
}

func createToolsDatabase(cfg *config.RouterConfig, plan embedding.RuntimePlan) (*tools.ToolsDatabase, error) {
	embeddingModels := cfg.EmbeddingModels
	toolsThreshold := embeddingModels.MinSimilarityThreshold()
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	var provider embedding.Provider
	if cfg.Tools.Enabled {
		var err error
		provider, err = toolsEmbeddingProvider(cfg, plan)
		if err != nil {
			return nil, err
		}
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		ModelType:           plan.ModelType,
		TargetDimension:     embeddingModels.EmbeddingConfig.TargetDimension,
		Provider:            provider,
	})

	if toolsDatabase.IsEnabled() {
		logging.ComponentEvent("extproc", "tools_database_initialized", map[string]interface{}{
			"similarity_threshold": toolsThreshold,
			"top_k":                cfg.Tools.TopK,
		})
	} else {
		logging.ComponentEvent("extproc", "tools_database_disabled", map[string]interface{}{})
	}

	return toolsDatabase, nil
}

func toolsEmbeddingProvider(cfg *config.RouterConfig, plan embedding.RuntimePlan) (embedding.Provider, error) {
	if cfg == nil || plan.Backend != config.EmbeddingBackendOpenAICompatible {
		return nil, nil
	}
	provider, err := embedding.NewProvider(cfg.EmbeddingModels, embedding.ProviderOptions{
		BackendOverride: plan.Backend,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create tools embedding provider: %w", err)
	}
	return provider, nil
}

func resolveRouterEmbeddingRuntimePlan(cfg *config.RouterConfig) (embedding.RuntimePlan, error) {
	if cfg == nil {
		return embedding.RuntimePlan{}, fmt.Errorf("embedding runtime config is nil")
	}
	return embedding.ResolveRuntimePlan(
		cfg.EmbeddingModels,
		embedding.BackendOverrideFromEnv(),
		embedding.ModelTypeOverrideFromEnv(),
	)
}

func (r *OpenAIRouter) resolvedEmbeddingRuntimePlan() (embedding.RuntimePlan, error) {
	if r != nil && r.embeddingRuntimePlan.Backend != "" {
		return r.embeddingRuntimePlan, nil
	}
	if r == nil {
		return embedding.RuntimePlan{}, fmt.Errorf("embedding router is nil")
	}
	return resolveRouterEmbeddingRuntimePlan(r.Config)
}

func createRouterClassifier(
	cfg *config.RouterConfig,
	mappings *classifierMappings,
) (*classification.Classifier, *services.ClassificationService, error) {
	classifier, err := classification.BuildClassifier(
		cfg,
		mappings.categoryMapping,
		mappings.piiMapping,
		mappings.jailbreakMapping,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build classifier: %w", err)
	}

	if err := classifier.InitializeRuntime(); err != nil {
		return nil, nil, fmt.Errorf("failed to initialize classifier runtime: %w", err)
	}

	classificationService := services.NewClassificationService(classifier, cfg)
	return classifier, classificationService, nil
}

func createResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}

	responseStore, err := createResponseStore(cfg)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "response_api_store_init_failed", map[string]interface{}{
			"backend":              cfg.ResponseAPI.StoreBackend,
			"error_type":           safeErrorForLog(err),
			"response_api_enabled": false,
		})
		return nil
	}

	logging.ComponentEvent("extproc", "response_api_initialized", map[string]interface{}{
		"backend": cfg.ResponseAPI.StoreBackend,
	})
	return NewResponseAPIFilter(responseStore)
}
