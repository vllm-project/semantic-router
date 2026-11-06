package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func createSemanticCache(cfg *config.RouterConfig, sets ...*embedding.Set) (cache.CacheBackend, string, error) {
	semanticCacheCfg := cfg.SemanticCache
	storeNeeded, semanticNeeded := cfg.ResponseCacheDemand()
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(semanticCacheCfg.BackendType),
		Enabled:             storeNeeded,
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

	if semanticNeeded && len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(cacheConfig.EmbeddingModel, 0, 0)
		if err != nil {
			return nil, "", fmt.Errorf("semantic cache embedding: %w", err)
		}
		cacheConfig.EmbeddingProvider = provider
	}
	identity := ""
	if semanticNeeded {
		var err error
		cacheConfig, identity, err = cache.PrepareEmbeddingNamespace(cacheConfig, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			return embedding.ResolveProviderIdentity(cacheConfig.EmbeddingProvider, settings)
		})
		if err != nil {
			return nil, "", fmt.Errorf("bind semantic cache embedding: %w", err)
		}
	}
	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create semantic cache: %w", err)
	}
	if semanticNeeded {
		if err := cache.ValidateBackendEmbedding(context.Background(), semanticCache); err != nil {
			_ = semanticCache.Close()
			return nil, "", fmt.Errorf("failed to prepare semantic cache embedding: %w", err)
		}
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

	return semanticCache, identity, nil
}

func detectSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	return config.SemanticCacheEmbeddingModel(cfg)
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
