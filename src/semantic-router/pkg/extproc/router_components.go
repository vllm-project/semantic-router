package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func createSemanticCache(cfg *config.RouterConfig, sets ...*embedding.Set) (cache.CacheBackend, error) {
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
		PolarityGuard: cache.PolarityGuardOptions{
			UseNLI:                 semanticCacheCfg.PolarityGuard.UsesNLI(),
			ContradictionThreshold: semanticCacheCfg.PolarityGuard.EffectiveContradictionThreshold(),
		},
	}

	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	if cacheConfig.Enabled && len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(cacheConfig.EmbeddingModel, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("semantic cache embedding: %w", err)
		}
		cacheConfig.EmbeddingProvider = provider
	}
	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}
	if err := cache.ValidateBackendEmbedding(context.Background(), semanticCache); err != nil {
		_ = semanticCache.Close()
		return nil, fmt.Errorf("failed to prepare semantic cache embedding: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.ComponentEvent("extproc", "semantic_cache_initialized", map[string]interface{}{
			"backend":              cacheConfig.BackendType,
			"similarity_threshold": cacheConfig.SimilarityThreshold,
			"ttl_seconds":          cacheConfig.TTLSeconds,
			"max_entries":          cacheConfig.MaxEntries,
			"polarity_guard_mode":  semanticCacheCfg.PolarityGuard.NormalizedMode(),
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

func createToolsDatabase(cfg *config.RouterConfig, provider embedding.Provider) (*tools.ToolsDatabase, error) {
	embeddingModels := cfg.EmbeddingModels
	toolsThreshold := embeddingModels.MinSimilarityThreshold()
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	if !cfg.Tools.Enabled {
		provider = nil
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		ModelType:           embeddingModels.EmbeddingConfig.ModelType,
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

func toolsEmbeddingProvider(cfg *config.RouterConfig, sets ...*embedding.Set) (embedding.Provider, error) {
	if len(sets) > 0 && sets[0] != nil {
		return sets[0].Get("", cfg.EmbeddingConfig.TargetDimension, 0)
	}

	if cfg == nil || !cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return nil, nil
	}
	provider, err := embedding.NewProvider(cfg.EmbeddingModels, embedding.ProviderOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create tools embedding provider: %w", err)
	}
	return provider, nil
}

func createRouterClassifier(
	cfg *config.RouterConfig,
	runtimeOptions ...classification.RecipeRuntimeOptions,
) (*classification.RecipeClassifiers, *classification.Classifier, *services.ClassificationService, error) {
	classifiers, err := classification.BuildRecipeClassifiers(
		cfg,
		nil,
		nil,
		nil,
		runtimeOptions...,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to build recipe classifiers: %w", err)
	}

	if err := classifiers.InitializeRuntime(); err != nil {
		_ = classifiers.Close()
		return nil, nil, nil, fmt.Errorf("failed to initialize recipe classifiers: %w", err)
	}

	defaultClassifier := classifiers.Default()
	if defaultClassifier == nil {
		_ = classifiers.Close()
		return nil, nil, nil, fmt.Errorf("default routing recipe classifier is unavailable")
	}
	classificationService := services.NewRecipeClassificationService(classifiers, cfg)
	return classifiers, defaultClassifier, classificationService, nil
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
