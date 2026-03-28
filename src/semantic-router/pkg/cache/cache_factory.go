package cache

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewCacheBackend creates a cache backend instance from the provided configuration
func NewCacheBackend(config CacheConfig) (CacheBackend, error) {
	if err := ValidateCacheConfig(config); err != nil {
		return nil, fmt.Errorf("invalid cache config: %w", err)
	}

	if !config.Enabled {
		// Create a disabled cache backend
		logging.Debugf("Cache disabled - creating disabled in-memory cache backend")
		return NewInMemoryCache(InMemoryCacheOptions{
			Enabled: false,
		}), nil
	}

	switch config.BackendType {
	case InMemoryCacheType, "":
		// Use in-memory cache as the default backend
		logging.Debugf("Creating in-memory cache backend - MaxEntries: %d, TTL: %ds, Threshold: %.3f, EmbeddingModel: %s, UseHNSW: %t",
			config.MaxEntries, config.TTLSeconds, config.SimilarityThreshold, config.EmbeddingModel, config.UseHNSW)

		options := InMemoryCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			MaxEntries:          config.MaxEntries,
			TTLSeconds:          config.TTLSeconds,
			EvictionPolicy:      config.EvictionPolicy,
			UseHNSW:             config.UseHNSW,
			HNSWM:               config.HNSWM,
			HNSWEfConstruction:  config.HNSWEfConstruction,
			EmbeddingModel:      config.EmbeddingModel,
		}
		return NewInMemoryCache(options), nil

	case MilvusCacheType:
		logging.Debugf("Creating Milvus cache backend - Config: %v, TTL: %ds, Threshold: %.3f, EmbeddingModel: %s",
			config.Milvus, config.TTLSeconds, config.SimilarityThreshold, config.EmbeddingModel)
		return NewMilvusCache(MilvusCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			TTLSeconds:          config.TTLSeconds,
			Config:              config.Milvus,
			EmbeddingModel:      config.EmbeddingModel,
		})

	case RedisCacheType:
		logging.Debugf("Creating Redis cache backend - Config: %v, TTL: %ds, Threshold: %.3f, EmbeddingModel: %s",
			config.Redis, config.TTLSeconds, config.SimilarityThreshold, config.EmbeddingModel)
		return NewRedisCache(RedisCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			TTLSeconds:          config.TTLSeconds,
			Config:              config.Redis,
			EmbeddingModel:      config.EmbeddingModel,
		})

	case ValkeyCacheType:
		if config.Valkey == nil {
			return nil, fmt.Errorf("valkey configuration is required for Valkey cache backend")
		}
		logging.Debugf("Creating Valkey cache backend - Config: %v, TTL: %ds, Threshold: %.3f, EmbeddingModel: %s",
			config.Valkey, config.TTLSeconds, config.SimilarityThreshold, config.EmbeddingModel)
		options := ValkeyCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			TTLSeconds:          config.TTLSeconds,
			Config:              config.Valkey,
			EmbeddingModel:      config.EmbeddingModel,
		}
		return NewValkeyCache(options)

	case HybridCacheType:
		logging.Debugf("Creating Hybrid cache backend - Config: %v, TTL: %ds, Threshold: %.3f, EmbeddingModel: %s",
			config.Milvus, config.TTLSeconds, config.SimilarityThreshold, config.EmbeddingModel)
		return NewHybridCache(hybridCacheOptionsFromConfig(config))

	default:
		logging.Debugf("Unsupported cache backend type: %s", config.BackendType)
		return nil, fmt.Errorf("unsupported cache backend type: %s", config.BackendType)
	}
}

func hybridCacheOptionsFromConfig(config CacheConfig) HybridCacheOptions {
	return HybridCacheOptions{
		Enabled:             config.Enabled,
		SimilarityThreshold: config.SimilarityThreshold,
		TTLSeconds:          config.TTLSeconds,
		MaxMemoryEntries:    config.MaxMemoryEntries,
		HNSWM:               config.HNSWM,
		HNSWEfConstruction:  config.HNSWEfConstruction,
		Milvus:              config.Milvus,
		EmbeddingModel:      config.EmbeddingModel,
	}
}

// ValidateCacheConfig validates cache configuration parameters
func ValidateCacheConfig(config CacheConfig) error {
	if !config.Enabled {
		return nil // Skip validation for disabled cache
	}

	// Check similarity threshold range
	if config.SimilarityThreshold < 0.0 || config.SimilarityThreshold > 1.0 {
		return fmt.Errorf("similarity_threshold must be between 0.0 and 1.0, got: %f", config.SimilarityThreshold)
	}

	// Check TTL value
	if config.TTLSeconds < 0 {
		return fmt.Errorf("ttl_seconds cannot be negative, got: %d", config.TTLSeconds)
	}

	// Check backend-specific requirements
	switch config.BackendType {
	case InMemoryCacheType, "":
		if config.MaxEntries < 0 {
			return fmt.Errorf("max_entries cannot be negative for in-memory cache, got: %d", config.MaxEntries)
		}
		// Validate eviction policy
		switch config.EvictionPolicy {
		case "", FIFOEvictionPolicyType, LRUEvictionPolicyType, LFUEvictionPolicyType:
			// "" is allowed, treated as FIFO by default
		default:
			return fmt.Errorf("unsupported eviction_policy: %s", config.EvictionPolicy)
		}
	case MilvusCacheType:
		if config.Milvus == nil {
			return fmt.Errorf("milvus configuration is required for Milvus cache backend")
		}
		logging.Debugf("Milvus configuration: %+v", config.Milvus)
	case RedisCacheType:
		if config.Redis == nil {
			return fmt.Errorf("redis configuration is required for Redis cache backend")
		}
	case HybridCacheType:
		if config.Milvus == nil {
			return fmt.Errorf("milvus configuration is required for hybrid cache backend")
		}
	case ValkeyCacheType:
		if config.Valkey == nil {
			return fmt.Errorf("valkey configuration is required for Valkey cache backend")
		}
	}

	return nil
}

// GetDefaultCacheConfig provides sensible default cache configuration values
func GetDefaultCacheConfig() CacheConfig {
	return CacheConfig{
		BackendType:         InMemoryCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.8,
		MaxEntries:          1000,
		TTLSeconds:          3600,
	}
}

// CacheBackendInfo describes the capabilities and features of a cache backend
type CacheBackendInfo struct {
	Type        CacheBackendType `json:"type"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Features    []string         `json:"features"`
}

// GetAvailableCacheBackends returns metadata for all supported cache backends
func GetAvailableCacheBackends() []CacheBackendInfo {
	return []CacheBackendInfo{
		{
			Type:        InMemoryCacheType,
			Name:        "In-Memory Cache",
			Description: "High-performance in-memory semantic cache with BERT embeddings",
			Features: []string{
				"Fast access",
				"No external dependencies",
				"Automatic memory management",
				"TTL support",
				"Entry limit support",
			},
		},
		{
			Type:        MilvusCacheType,
			Name:        "Milvus Vector Database",
			Description: "Enterprise-grade semantic cache powered by Milvus vector database",
			Features: []string{
				"Highly scalable",
				"Persistent storage",
				"Distributed architecture",
				"Advanced indexing",
				"High availability",
				"TTL support",
			},
		},
		{
			Type:        RedisCacheType,
			Name:        "Redis Vector Database",
			Description: "High-performance semantic cache powered by Redis with vector search",
			Features: []string{
				"Fast in-memory performance",
				"Persistent storage with AOF/RDB",
				"Scalable with Redis Cluster",
				"HNSW and FLAT indexing",
				"Wide ecosystem support",
				"TTL support",
			},
		},
		{
			Type:        ValkeyCacheType,
			Name:        "Valkey Vector Database",
			Description: "High-performance semantic cache powered by Valkey with vector search",
			Features: []string{
				"Fast in-memory performance",
				"Persistent storage with AOF/RDB",
				"Scalable with Valkey Cluster",
				"HNSW and FLAT indexing",
				"Redis-compatible protocol",
				"TTL support",
			},
		},
	}
}
