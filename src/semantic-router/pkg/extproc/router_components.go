package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func loadClassifierMappings(cfg *config.RouterConfig) (*classifierMappings, error) {
	mappings := &classifierMappings{}
	var err error

	if cfg.CategoryMappingPath != "" {
		mappings.categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "category_mapping_loaded", map[string]interface{}{
			"count": mappings.categoryMapping.GetCategoryCount(),
		})
	}

	if cfg.PIIMappingPath != "" {
		mappings.piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "pii_mapping_loaded", map[string]interface{}{
			"count": mappings.piiMapping.GetPIITypeCount(),
		})
	}

	if cfg.IsPromptGuardEnabled() {
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

func createToolsDatabase(cfg *config.RouterConfig) *tools.ToolsDatabase {
	embeddingModels := cfg.EmbeddingModels
	toolsThreshold := embeddingModels.MinSimilarityThreshold()
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		ModelType:           embeddingModels.EmbeddingConfig.ModelType,
		TargetDimension:     embeddingModels.EmbeddingConfig.TargetDimension,
	})

	if toolsDatabase.IsEnabled() {
		logging.ComponentEvent("extproc", "tools_database_initialized", map[string]interface{}{
			"similarity_threshold": toolsThreshold,
			"top_k":                cfg.Tools.TopK,
		})
	} else {
		logging.ComponentEvent("extproc", "tools_database_disabled", map[string]interface{}{})
	}

	return toolsDatabase
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
			"error":                err.Error(),
			"response_api_enabled": false,
		})
		return nil
	}

	logging.ComponentEvent("extproc", "response_api_initialized", map[string]interface{}{
		"backend": cfg.ResponseAPI.StoreBackend,
	})
	return NewResponseAPIFilter(responseStore)
}
