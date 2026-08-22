package cache

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewCacheBackend creates a cache backend instance from the provided configuration
func NewCacheBackend(config CacheConfig) (LegacyCacheBackend, error) {
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

	return buildRegisteredBackend(config)
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
		return nil
	}
	if config.SimilarityThreshold < 0.0 || config.SimilarityThreshold > 1.0 {
		return fmt.Errorf("similarity_threshold must be between 0.0 and 1.0, got: %f", config.SimilarityThreshold)
	}
	if config.TTLSeconds < 0 {
		return fmt.Errorf("ttl_seconds cannot be negative, got: %d", config.TTLSeconds)
	}
	return validateCacheBackend(config)
}

func validateCacheBackend(config CacheConfig) error {
	return validateRegisteredBackend(config)
}

func validateInMemoryCacheConfig(config CacheConfig) error {
	if config.MaxEntries < 0 {
		return fmt.Errorf("max_entries cannot be negative for in-memory cache, got: %d", config.MaxEntries)
	}
	switch config.EvictionPolicy {
	case "", FIFOEvictionPolicyType, LRUEvictionPolicyType, LFUEvictionPolicyType:
	default:
		return fmt.Errorf("unsupported eviction_policy: %s", config.EvictionPolicy)
	}
	return nil
}

func validateQdrantCacheConfig(cfg *config.QdrantConfig) error {
	if cfg == nil {
		return fmt.Errorf("qdrant configuration is required for Qdrant cache backend")
	}
	if cfg.Host == "" {
		return fmt.Errorf("qdrant.host is required for Qdrant cache backend")
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
	Type         CacheBackendType    `json:"type"`
	Name         string              `json:"name"`
	Description  string              `json:"description"`
	Features     []string            `json:"features"`
	Capabilities BackendCapabilities `json:"capabilities"`
}

// GetAvailableCacheBackends returns metadata for all supported cache backends
func GetAvailableCacheBackends() []CacheBackendInfo {
	return []CacheBackendInfo{
		{
			Type:         InMemoryCacheType,
			Name:         "In-Memory Cache",
			Description:  "High-performance in-memory semantic cache with BERT embeddings",
			Capabilities: CapabilitiesForBackend(InMemoryCacheType),
			Features: []string{
				"Fast access",
				"No external dependencies",
				"Automatic memory management",
				"TTL support",
				"Entry limit support",
			},
		},
		{
			Type:         MilvusCacheType,
			Name:         "Milvus Vector Database",
			Description:  "Enterprise-grade semantic cache powered by Milvus vector database",
			Capabilities: CapabilitiesForBackend(MilvusCacheType),
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
			Type:         RedisCacheType,
			Name:         "Redis Vector Database",
			Description:  "High-performance semantic cache powered by Redis with vector search",
			Capabilities: CapabilitiesForBackend(RedisCacheType),
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
			Type:         ValkeyCacheType,
			Name:         "Valkey Vector Database",
			Description:  "High-performance semantic cache powered by Valkey with vector search",
			Capabilities: CapabilitiesForBackend(ValkeyCacheType),
			Features: []string{
				"Fast in-memory performance",
				"Persistent storage with AOF/RDB",
				"Scalable with Valkey Cluster",
				"HNSW and FLAT indexing",
				"Redis-compatible protocol",
				"TTL support",
			},
		},
		{
			Type:         QdrantCacheType,
			Name:         "Qdrant Vector Database",
			Description:  "Distributed response cache backed by Qdrant vector search",
			Features:     []string{"Exact lookup", "Semantic lookup", "TTL support"},
			Capabilities: CapabilitiesForBackend(QdrantCacheType),
		},
		{
			Type:         HybridCacheType,
			Name:         "Hybrid HNSW and Milvus",
			Description:  "Process-local HNSW hot tier backed by shared Milvus storage",
			Features:     []string{"Hot tier", "Shared persistence", "TTL support"},
			Capabilities: CapabilitiesForBackend(HybridCacheType),
		},
	}
}
