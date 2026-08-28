package cache

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type cacheBackendRegistration struct {
	capabilities BackendCapabilities
	validate     func(CacheConfig) error
	build        func(CacheConfig) (LegacyCacheBackend, error)
}

var cacheBackendRegistry = map[CacheBackendType]cacheBackendRegistration{
	InMemoryCacheType: {
		capabilities: BackendCapabilities{
			Backend:          InMemoryCacheType,
			Exact:            true,
			Semantic:         true,
			PerEntryTTL:      true,
			Stats:            true,
			Invalidate:       true,
			Flush:            true,
			ConsistencyModel: "process-local",
		},
		validate: validateInMemoryCacheConfig,
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			logging.Debugf("Creating in-memory response cache backend")
			return NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             config.Enabled,
				SimilarityThreshold: config.SimilarityThreshold,
				MaxEntries:          config.MaxEntries,
				TTLSeconds:          config.TTLSeconds,
				EvictionPolicy:      config.EvictionPolicy,
				UseHNSW:             config.UseHNSW,
				HNSWM:               config.HNSWM,
				HNSWEfConstruction:  config.HNSWEfConstruction,
				EmbeddingModel:      config.EmbeddingModel,
				PolarityGuard:       config.PolarityGuard,
			}), nil
		},
	},
	RedisCacheType: {
		capabilities: sharedCapabilities(RedisCacheType, "eventual"),
		validate: func(config CacheConfig) error {
			if config.Redis == nil {
				return fmt.Errorf("redis configuration is required for Redis cache backend")
			}
			return nil
		},
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			return NewRedisCache(RedisCacheOptions{
				Enabled:             config.Enabled,
				SimilarityThreshold: config.SimilarityThreshold,
				TTLSeconds:          config.TTLSeconds,
				Config:              config.Redis,
				EmbeddingModel:      config.EmbeddingModel,
			})
		},
	},
	ValkeyCacheType: {
		capabilities: sharedCapabilities(ValkeyCacheType, "eventual"),
		validate: func(config CacheConfig) error {
			if config.Valkey == nil {
				return fmt.Errorf("valkey configuration is required for Valkey cache backend")
			}
			return nil
		},
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			return NewValkeyCache(ValkeyCacheOptions{
				Enabled:             config.Enabled,
				SimilarityThreshold: config.SimilarityThreshold,
				TTLSeconds:          config.TTLSeconds,
				Config:              config.Valkey,
				EmbeddingModel:      config.EmbeddingModel,
			})
		},
	},
	MilvusCacheType: {
		capabilities: sharedCapabilities(MilvusCacheType, "session"),
		validate: func(config CacheConfig) error {
			if config.Milvus == nil {
				return fmt.Errorf("milvus configuration is required for Milvus cache backend")
			}
			return nil
		},
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			return NewMilvusCache(MilvusCacheOptions{
				Enabled:             config.Enabled,
				SimilarityThreshold: config.SimilarityThreshold,
				TTLSeconds:          config.TTLSeconds,
				Config:              config.Milvus,
				EmbeddingModel:      config.EmbeddingModel,
			})
		},
	},
	QdrantCacheType: {
		capabilities: sharedCapabilities(QdrantCacheType, "eventual"),
		validate: func(config CacheConfig) error {
			return validateQdrantCacheConfig(config.Qdrant)
		},
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			return NewQdrantCache(QdrantCacheOptions{
				Enabled:             config.Enabled,
				SimilarityThreshold: config.SimilarityThreshold,
				TTLSeconds:          config.TTLSeconds,
				Config:              config.Qdrant,
				EmbeddingModel:      config.EmbeddingModel,
			})
		},
	},
	HybridCacheType: {
		capabilities: sharedCapabilities(HybridCacheType, "session"),
		validate: func(config CacheConfig) error {
			if config.Milvus == nil {
				return fmt.Errorf("milvus configuration is required for hybrid cache backend")
			}
			return nil
		},
		build: func(config CacheConfig) (LegacyCacheBackend, error) {
			return NewHybridCache(hybridCacheOptionsFromConfig(config))
		},
	},
}

func sharedCapabilities(
	backend CacheBackendType,
	consistency string,
) BackendCapabilities {
	return BackendCapabilities{
		Backend:          backend,
		Exact:            true,
		Semantic:         true,
		PerEntryTTL:      true,
		Stats:            true,
		Invalidate:       true,
		Flush:            true,
		Shared:           true,
		ConsistencyModel: consistency,
	}
}

func normalizedBackendType(backend CacheBackendType) CacheBackendType {
	if backend == "" {
		return InMemoryCacheType
	}
	return backend
}

func CapabilitiesForBackend(backend CacheBackendType) BackendCapabilities {
	registration, ok := cacheBackendRegistry[normalizedBackendType(backend)]
	if !ok {
		return BackendCapabilities{Backend: backend}
	}
	return registration.capabilities
}

func validateRegisteredBackend(config CacheConfig) error {
	registration, ok := cacheBackendRegistry[normalizedBackendType(config.BackendType)]
	if !ok {
		return fmt.Errorf("unsupported cache backend type: %s", config.BackendType)
	}
	if err := validateSelectedBackendFields(config); err != nil {
		return err
	}
	return registration.validate(config)
}

func buildRegisteredBackend(config CacheConfig) (LegacyCacheBackend, error) {
	registration, ok := cacheBackendRegistry[normalizedBackendType(config.BackendType)]
	if !ok {
		return nil, fmt.Errorf("unsupported cache backend type: %s", config.BackendType)
	}
	return registration.build(config)
}

func validateSelectedBackendFields(config CacheConfig) error {
	selected := normalizedBackendType(config.BackendType)
	configured := map[CacheBackendType]bool{
		RedisCacheType:  config.Redis != nil,
		ValkeyCacheType: config.Valkey != nil,
		MilvusCacheType: config.Milvus != nil,
		QdrantCacheType: config.Qdrant != nil,
	}
	if selected == HybridCacheType {
		delete(configured, MilvusCacheType)
	}
	for backend, present := range configured {
		if present && backend != selected {
			return fmt.Errorf("%s configuration is not valid when backend_type is %s", backend, selected)
		}
	}
	return nil
}
